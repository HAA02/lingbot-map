"""Background lingbot-map inference for the realtime server.

- Holds a rolling buffer of preprocessed frames (518-wide, H rounded to /14).
- Every `interval_s` seconds, runs `model.inference_streaming` on the latest
  window of `window_size` frames and pushes the result to listeners via the
  injected `broadcast_fn`.

The model load is heavy (~5 s + 11 GB peak) so we load once at startup.
"""
from __future__ import annotations

import asyncio
import logging
import struct
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch

from lingbot_map.models.gct_stream import GCTStream
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import (
    closed_form_inverse_se3_general,
    unproject_depth_map_to_point_map,
)

log = logging.getLogger("realtime.infer")


def _voxel_downsample(xyz: np.ndarray, rgb: np.ndarray, voxel_size: float = 0.025):
    """Average points that fall in the same voxel cell (pure numpy, no Open3D)."""
    keys = np.floor(xyz / voxel_size).astype(np.int32)
    # Pack (ix, iy, iz) into a single int64 key for fast unique lookup.
    # Shift assumes coordinates stay within ±32768 voxels (~820m at 2.5cm).
    packed = (keys[:, 0].astype(np.int64) * 65536 + keys[:, 1].astype(np.int64)) * 65536 + keys[:, 2].astype(np.int64)
    _, inv, counts = np.unique(packed, return_inverse=True, return_counts=True)
    n = len(counts)
    xyz_out = np.zeros((n, 3), np.float64)
    rgb_out = np.zeros((n, 3), np.float64)
    np.add.at(xyz_out, inv, xyz)
    np.add.at(rgb_out, inv, rgb.astype(np.float64))
    c = counts[:, None].astype(np.float64)
    return (xyz_out / c).astype(np.float32), (rgb_out / c).clip(0, 255).astype(np.uint8)


def _voxel_fuse_weighted(
    xyz: np.ndarray, rgb: np.ndarray, conf: np.ndarray,
    voxel_size: float = 0.025, min_obs: int = 1,
):
    """Confidence-weighted voxel fusion — multi-view observations of the same
    surface collapse into one point at the weighted-mean position, with color
    drawn from the highest-confidence observation in the cell.

    Args:
        xyz: (N, 3) world-frame points.
        rgb: (N, 3) uint8 colors.
        conf: (N,) per-point depth confidence (e.g., from depth_conf head).
        voxel_size: cell edge in meters.
        min_obs: minimum number of observations per voxel to keep (1 = keep all).

    Returns:
        (M, 3) xyz, (M, 3) rgb, (M,) accumulated weight per voxel.
    """
    keys = np.floor(xyz / voxel_size).astype(np.int32)
    packed = (keys[:, 0].astype(np.int64) * 65536 + keys[:, 1].astype(np.int64)) * 65536 + keys[:, 2].astype(np.int64)
    _, inv, counts = np.unique(packed, return_inverse=True, return_counts=True)
    n = len(counts)

    w = np.clip(conf, 1e-3, None).astype(np.float64)
    sumw = np.zeros(n, dtype=np.float64)
    sumxyz = np.zeros((n, 3), dtype=np.float64)
    np.add.at(sumw, inv, w)
    np.add.at(sumxyz, inv, xyz.astype(np.float64) * w[:, None])
    xyz_out = sumxyz / sumw[:, None]

    # Color: take the highest-confidence observation in each voxel
    # (more robust than mean — preserves edge sharpness).
    # Vectorized argmax-per-group via sort + reduce-last-occurrence.
    order = np.argsort(w, kind="stable")  # ascending; best (largest w) appears last in each group
    inv_sorted = inv[order]
    # For each voxel, the last appearance of inv_sorted == voxel gives the highest-w index.
    best_idx_in_order = np.full(n, -1, dtype=np.int64)
    best_idx_in_order[inv_sorted] = np.arange(len(order))  # overwrites earlier with later
    rgb_out = rgb[order[best_idx_in_order]]

    if min_obs > 1:
        keep = counts >= min_obs
        xyz_out = xyz_out[keep]; rgb_out = rgb_out[keep]; sumw = sumw[keep]
    return xyz_out.astype(np.float32), rgb_out, sumw.astype(np.float32)


def _radius_outlier_filter(xyz: np.ndarray, rgb: np.ndarray,
                           radius: float = 0.05, min_neighbors: int = 4):
    """Drop points that have fewer than `min_neighbors` other points within
    `radius`. Removes floating noise from multi-view depth disagreements.
    Uses scipy cKDTree for C-speed neighbor counting."""
    if xyz.shape[0] < min_neighbors + 1:
        return xyz, rgb
    from scipy.spatial import cKDTree
    tree = cKDTree(xyz)
    # count_neighbors against self → each point counts itself too, so we
    # require min_neighbors + 1 from the query.
    counts = tree.query_ball_point(xyz, r=radius, return_length=True)
    keep = counts >= (min_neighbors + 1)
    return xyz[keep], rgb[keep]


def _preprocess_bgr(bgr: np.ndarray, target_w: int = 518, patch: int = 14) -> torch.Tensor:
    """BGR uint8 (H, W, 3) -> RGB float tensor (3, H', W'), values in [0,1].

    Mirrors demo.py's `load_and_preprocess_images(mode="crop")`:
      1. width set to `target_w`
      2. height = round(h * (target_w/w) / patch) * patch
      3. if new_h > target_w, center-crop to target_w (square cap)
    Frame rotation must be applied by caller before this — videos with
    container rotation metadata need cv2.CAP_PROP_ORIENTATION_AUTO=1 or
    manual cv2.rotate.
    """
    h, w = bgr.shape[:2]
    new_w = target_w
    new_h = max(patch, round(h * (new_w / w) / patch) * patch)
    img = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    # Center-crop tall frames so the model sees an aspect within its training
    # distribution (max height == target_w == 518).
    if new_h > target_w:
        start = (new_h - target_w) // 2
        img = img[start:start + target_w]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return t


def _poses_from_pose_enc(pose_enc: torch.Tensor, img_hw: tuple[int, int]):
    """Return (w2c, c2w, K). w2c is what pose_encoding_to_extri_intri emits natively
    (OpenCV "cam-from-world"); c2w is its inverse, for camera placement display."""
    w2c, K = pose_encoding_to_extri_intri(pose_enc, img_hw)  # (..., 3, 4), (..., 3, 3)
    ext44 = torch.zeros((*w2c.shape[:-2], 4, 4), device=w2c.device, dtype=w2c.dtype)
    ext44[..., :3, :4] = w2c
    ext44[..., 3, 3] = 1.0
    c2w = closed_form_inverse_se3_general(ext44)[..., :3, :4]
    return w2c, c2w, K


class InferenceWorker:
    def __init__(
        self,
        model_path: str,
        broadcast_fn: Callable[[bytes], Any],
        *,
        device: str = "cuda",
        image_size: int = 518,
        patch_size: int = 14,
        window_size: int = 32,         # streaming: 32 frames max per tick (was 64)
        interval_s: float = 5.0,       # streaming tick interval (was 2.5 — caused pileup)
        max_buffer: int = 32,
        max_points_per_frame: int = 30000,
        conf_threshold: float = 1.5,     # demo's absolute conf cutoff (vis_threshold)
        num_scale_frames: int = 16,      # more anchor frames → better global consistency
        output_mode: str = "mesh",       # "points" (LBP2) or "mesh" (LBM1, Tier 2 TSDF)
        tsdf_voxel: float = 0.025,       # TSDF voxel size (m)
        tsdf_trunc: float = 0.10,        # truncation distance (m)
    ) -> None:
        self.broadcast_fn = broadcast_fn
        self.device = device
        self.image_size = image_size
        self.patch_size = patch_size
        self.window_size = window_size
        self.interval_s = interval_s
        self.max_points_per_frame = max_points_per_frame
        self.conf_threshold = conf_threshold
        self.num_scale_frames = num_scale_frames
        self.output_mode = output_mode
        self.tsdf_voxel = tsdf_voxel
        self.tsdf_trunc = tsdf_trunc

        # Anchor frames pin the world origin + depth scale across ticks: the first
        # `num_scale_frames` frames are kept forever and re-fed as scale frames on
        # every inference call, so `inference_streaming`'s bidirectional scale
        # block produces an identical world frame each tick.
        self.anchor: list[tuple[torch.Tensor, np.ndarray, int]] = []
        # Rolling tail of recent frames (excludes anchor); sliced on each tick.
        self.buffer: deque[tuple[torch.Tensor, np.ndarray, int]] = deque(
            maxlen=max(1, max_buffer - num_scale_frames)
        )
        self._lock = asyncio.Lock()
        self._frame_seq = 0
        self._last_emit_seq = -1
        self._last_push_ts = 0.0

        log.info("loading lingbot-map model on %s …", device)
        t0 = time.time()
        self.model = GCTStream(
            img_size=image_size,
            patch_size=patch_size,
            enable_3d_rope=True,
            max_frame_num=1024,
            kv_cache_sliding_window=64,
            kv_cache_scale_frames=8,
            kv_cache_cross_frame_special=True,
            kv_cache_include_scale_frames=True,
            use_sdpa=True,
            camera_num_iterations=4,
            enable_point=False,   # checkpoint missing point_head weights → use depth unprojection
        )
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        log.info("checkpoint loaded  missing=%d unexpected=%d", len(missing), len(unexpected))
        self.model.aggregator.to(dtype=torch.bfloat16)
        self.model.to(device).eval()
        log.info("model ready in %.1fs", time.time() - t0)

        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        """Spawn the periodic inference loop. Idempotent."""
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self._loop())
            log.info("inference loop spawned (interval %.1fs)", self.interval_s)

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await self._task

    async def process_video_file(
        self, path: Path, *, target_frames: int = 32,
        multi_window_threshold: int = 300, window_overlap: int = 8,
    ) -> tuple[bytes | None, list[bytes], dict]:
        """Sample frames from the video, run inference, broadcast the result.

        For long videos (`total_frames > multi_window_threshold`), runs multiple
        overlapping inference windows and aligns them via Procrustes registration
        on the overlapping camera centers (Tier 3). Otherwise single-window path.
        """
        loop = asyncio.get_running_loop()
        # Quick probe to decide single vs multi-window
        meta_probe = await loop.run_in_executor(None, self._probe_video, str(path))
        total = meta_probe.get("total_frames", 0)
        use_long = total > multi_window_threshold

        if use_long:
            # Multi-window: split into K windows of `target_frames` each
            num_windows = max(2, min(6, total // (target_frames * 2)))
            payload, thumbs_jpeg, info = await self._process_video_multiwindow(
                str(path), target_frames=target_frames,
                num_windows=num_windows, overlap=window_overlap,
            )
            info.update(meta_probe)
            info["mode"] = "multi-window"
            info["windows"] = num_windows
            return payload, thumbs_jpeg, info

        snap, meta = await loop.run_in_executor(
            None, self._sample_video_sync, str(path), target_frames
        )
        if not snap:
            return None, [], {"error": "no frames extracted", **meta}
        payload = await loop.run_in_executor(None, self._infer_sync, snap)
        # JPEG-encode each frame's thumbnail (the same pixels the model saw) so
        # the viewer can paint them on each camera frustum.
        thumbs_jpeg: list[bytes] = []
        for _, thumb_rgb, _ in snap:
            small = cv2.resize(thumb_rgb, (128, 128), interpolation=cv2.INTER_AREA)
            bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            thumbs_jpeg.append(buf.tobytes() if ok else b"")
        info = {**meta, "frames_used": len(snap),
                "payload_bytes": len(payload) if payload else 0,
                "thumbs_bytes": sum(len(t) for t in thumbs_jpeg)}
        if payload is not None:
            try:
                await self.broadcast_fn(payload, thumbs_jpeg)
            except Exception as e:
                log.warning("broadcast failed: %s", e)
                info["broadcast_error"] = str(e)
        return payload, thumbs_jpeg, info

    def _probe_video(self, video_path: str) -> dict:
        cap = cv2.VideoCapture(video_path)
        info = {
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "fps": round(float(cap.get(cv2.CAP_PROP_FPS) or 0.0), 2),
            "rotation": int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0),
        }
        cap.release()
        return info

    async def _process_video_multiwindow(
        self, video_path: str, target_frames: int, num_windows: int, overlap: int,
    ) -> tuple[bytes | None, list[bytes], dict]:
        """Tier 3: Sample windows from the video, infer each, align via Procrustes,
        and run the final fusion (Tier1 + TSDF if enabled) on the merged data."""
        loop = asyncio.get_running_loop()
        info: dict = {}

        # Decide window bounds in raw-frame indices, with overlap
        meta = self._probe_video(video_path)
        total = meta["total_frames"]
        # frame-range of each window
        # span = window's source frame range; consecutive windows share `overlap` virtual frames
        # We define virtual-frame indices [0..target_frames*num_windows - overlap*(num_windows-1)]
        # mapped uniformly into [0..total-1].
        eff_len = target_frames * num_windows - overlap * (num_windows - 1)
        virt_indices = np.linspace(0, total - 1, eff_len).astype(int)
        win_virt_ranges = []
        for k in range(num_windows):
            start_v = k * (target_frames - overlap)
            end_v = start_v + target_frames
            win_virt_ranges.append((start_v, end_v))

        # Sample per-window: extract `target_frames` real-frame indices for each window
        windows_data = []
        thumbs_per_window: list[list[np.ndarray]] = []
        for k, (s, e) in enumerate(win_virt_ranges):
            real_idx = [int(virt_indices[i]) for i in range(s, e)]
            snap = await loop.run_in_executor(
                None, self._sample_video_by_indices, video_path, real_idx, meta["rotation"]
            )
            if not snap:
                continue
            win_raw = await loop.run_in_executor(None, self._run_window_raw, snap)
            if win_raw is None:
                continue
            windows_data.append(win_raw)
            thumbs_per_window.append([t[1] for t in snap])

        if not windows_data:
            return None, [], {"error": "no windows produced output"}
        info["windows_succeeded"] = len(windows_data)

        # Align windows via Procrustes on overlap cam centers
        from registration import chain_windows
        try:
            merged = chain_windows(windows_data, overlap=overlap)
            info["alignment_scales"] = merged.get("scales_to_ref")
        except Exception as exc:
            log.exception("multi-window alignment failed: %s", exc)
            return None, [], {"error": f"alignment failed: {exc}"}

        # Run final fusion (Tier 1 + Tier 2) on merged data
        payload = await loop.run_in_executor(None, self._fuse_merged, merged)

        # Build thumbs JPEG (drop overlap dupes to match merged frame count)
        thumbs_flat: list[np.ndarray] = list(thumbs_per_window[0])
        for k in range(1, len(thumbs_per_window)):
            thumbs_flat.extend(thumbs_per_window[k][overlap:])
        thumbs_jpeg: list[bytes] = []
        for thumb_rgb in thumbs_flat:
            small = cv2.resize(thumb_rgb, (128, 128), interpolation=cv2.INTER_AREA)
            bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            thumbs_jpeg.append(buf.tobytes() if ok else b"")

        if payload is not None:
            try:
                await self.broadcast_fn(payload, thumbs_jpeg)
            except Exception as e:
                log.warning("broadcast failed: %s", e)

        info.update({
            "frames_used": merged["frame_count"],
            "payload_bytes": len(payload) if payload else 0,
            "thumbs_bytes": sum(len(t) for t in thumbs_jpeg),
        })
        return payload, thumbs_jpeg, info

    def _sample_video_by_indices(self, video_path: str, indices: list[int], rotation: int):
        """Like _sample_video_sync but for a specific list of frame indices."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        try:
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1.0)
        except Exception:
            pass
        rot_map = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
        rot_code = rot_map.get(rotation)
        snap = []
        seq = 0
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            if rot_code is not None and rotation in (90, 270) and bgr.shape[1] > bgr.shape[0]:
                bgr = cv2.rotate(bgr, rot_code)
            elif rot_code == cv2.ROTATE_180:
                bgr = cv2.rotate(bgr, rot_code)
            tensor = _preprocess_bgr(bgr, target_w=self.image_size, patch=self.patch_size)
            thumb = (tensor.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            seq += 1
            snap.append((tensor, thumb, seq))
        cap.release()
        return snap

    def _run_window_raw(self, snap):
        """Run inference on one window; return per-window raw data (no fusion yet).
        Used by multi-window pipeline before Procrustes alignment.
        Returns dict or None if depth missing."""
        if not snap:
            return None
        tensors = [t[0] for t in snap]
        thumbs = [t[1] for t in snap]
        seqs = [t[2] for t in snap]
        images = torch.stack(tensors, dim=0).unsqueeze(0).to(self.device)
        N, _, H, W = images.shape[1:]
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            preds = self.model.inference_streaming(
                images.squeeze(0),
                num_scale_frames=self.num_scale_frames,
                keyframe_interval=1,
                output_device=None,
            )
        for k in list(preds.keys()):
            if isinstance(preds[k], torch.Tensor) and preds[k].dim() >= 4 and preds[k].shape[0] == 1:
                preds[k] = preds[k][0]
        w2c, c2w, K = _poses_from_pose_enc(preds["pose_enc"], (H, W))
        if w2c.dim() == 4 and w2c.shape[0] == 1: w2c = w2c[0]
        if c2w.dim() == 4 and c2w.shape[0] == 1: c2w = c2w[0]
        if K.dim()   == 4 and K.shape[0]   == 1: K   = K[0]
        w2c_np = w2c.detach().cpu().numpy().astype(np.float32)
        c2w_np = c2w.detach().cpu().numpy().astype(np.float32)
        K_np   = K.detach().cpu().numpy().astype(np.float32)
        depth = preds.get("depth")
        if depth is None:
            return None
        depth_np = depth.detach().cpu().numpy().astype(np.float32)
        if depth_np.ndim == 3:
            depth_np = depth_np[..., None]
        wp = unproject_depth_map_to_point_map(depth_np, w2c_np, K_np)
        dc = preds.get("depth_conf")
        wpc = dc.detach().cpu().numpy() if dc is not None else None

        all_xyz, all_rgb, all_conf = [], [], []
        for i in range(N):
            pts = wp[i].reshape(-1, 3)
            cols = thumbs[i].reshape(-1, 3).astype(np.uint8)
            conf = wpc[i].reshape(-1) if wpc is not None else None
            mask = conf > self.conf_threshold if conf is not None else np.ones(pts.shape[0], dtype=bool)
            if mask.sum() == 0:
                continue
            idx = np.where(mask)[0]
            if idx.size > self.max_points_per_frame:
                idx = np.random.default_rng(int(seqs[i])).choice(idx, self.max_points_per_frame, replace=False)
            all_xyz.append(pts[idx]); all_rgb.append(cols[idx])
            all_conf.append(conf[idx] if conf is not None else np.ones(idx.size, dtype=np.float32))
        if not all_xyz:
            return None
        return {
            "xyz":  np.concatenate(all_xyz,  axis=0).astype(np.float32),
            "rgb":  np.concatenate(all_rgb,  axis=0).astype(np.uint8),
            "conf": np.concatenate(all_conf, axis=0).astype(np.float32),
            "c2w":  c2w_np,
            "w2c":  w2c_np,
            "K":    K_np,
            "depth": depth_np.squeeze(-1),  # (N, H, W)
            "thumbs": thumbs,
        }

    def _fuse_merged(self, merged: dict) -> bytes | None:
        """Run Tier 1 + Tier 2 fusion on a merged multi-window result."""
        xyz, rgb, conf = merged["xyz"], merged["rgb"], merged["conf"]
        c2w_np, w2c_np, K_np = merged["c2w"], merged["w2c"], merged["K"]
        depth_hw = merged["depth"]
        thumbs = merged["thumbs"]

        # Tier 1 — voxel fusion
        n_before = xyz.shape[0]
        xyz, rgb, _ = _voxel_fuse_weighted(xyz, rgb, conf, voxel_size=0.015, min_obs=1)
        xyz, rgb = _radius_outlier_filter(xyz, rgb, radius=0.04, min_neighbors=3)
        log.info("Tier1+3 fusion: merged %d → voxel+outlier %d points (windows=%d)",
                 n_before, xyz.shape[0], merged.get("n_windows", 1))

        seq = 0
        # Tier 2 — TSDF if enabled
        if self.output_mode == "mesh":
            try:
                from tsdf import build_tsdf_volume, extract_mesh, color_vertices
                t0 = time.time()
                vol = build_tsdf_volume(
                    depth_hw, w2c_np, K_np, None,  # no per-pixel conf in merged path; use 1
                    voxel_size=self.tsdf_voxel, trunc=self.tsdf_trunc,
                    conf_floor=0.0,
                )
                verts, faces, normals = extract_mesh(vol, min_weight=1.0)
                imgs = np.stack(thumbs, axis=0)
                v_rgb = color_vertices(verts, imgs, w2c_np, K_np)
                log.info("Tier2+3 TSDF: verts=%d faces=%d in %.2fs",
                         verts.shape[0], faces.shape[0], time.time() - t0)
                if verts.shape[0] > 0 and faces.shape[0] > 0:
                    header = struct.pack("<4sIIIII", b"LBM1", 1,
                                         verts.shape[0], faces.shape[0],
                                         c2w_np.shape[0], seq)
                    body = (verts.astype(np.float32).tobytes()
                            + v_rgb.astype(np.uint8).tobytes()
                            + faces.astype(np.uint32).tobytes()
                            + c2w_np.astype(np.float32).tobytes()
                            + K_np[0].astype(np.float32).tobytes())
                    return header + body
                log.warning("TSDF empty mesh → fall back to LBP2")
            except Exception:
                log.exception("TSDF failed in merged path; falling back to points")

        # LBP2 point cloud fallback
        header = struct.pack("<4sIIII", b"LBP2", 1, xyz.shape[0], c2w_np.shape[0], seq)
        body = (xyz.tobytes() + rgb.tobytes()
                + c2w_np.astype(np.float32).tobytes()
                + K_np[0].astype(np.float32).tobytes())
        return header + body

    def _sample_video_sync(self, video_path: str, target_frames: int):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], {"error": "cannot open video"}
        # Honor container rotation metadata. Phone videos store landscape pixels
        # + Rotate=90/180/270; without this the model sees a sideways scene and
        # the entire point cloud comes out rotated.
        rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)
        try:
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1.0)
        except Exception:
            pass
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        meta = {"total_frames": total, "fps": round(fps, 2), "rotation": rotation}
        if total <= 0:
            cap.release()
            return [], {**meta, "error": "no frames"}
        n = min(target_frames, total)
        # uniformly spaced indices
        if n == 1:
            indices = [total // 2]
        else:
            indices = [int(round(i * (total - 1) / (n - 1))) for i in range(n)]
        # Fallback manual rotation when ORIENTATION_AUTO is silently ignored
        # by the backend (depends on cv2 build + ffmpeg version).
        rot_map = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }
        rot_code = rot_map.get(rotation)
        snap = []
        seq = 0
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            # If auto-rotate didn't fire, the frame still has the source aspect.
            # We detect by checking against the expected post-rotation aspect:
            # for 90/270 rotations, expected H >= W (portrait). If we got W > H
            # while metadata says rotated, rotate manually.
            if rot_code is not None and rotation in (90, 270) and bgr.shape[1] > bgr.shape[0]:
                bgr = cv2.rotate(bgr, rot_code)
            elif rot_code == cv2.ROTATE_180:
                # 180 doesn't change aspect, so we can't auto-detect — apply unconditionally
                # only when rotation==180 was reported (rare on phones).
                bgr = cv2.rotate(bgr, rot_code)
            tensor = _preprocess_bgr(bgr, target_w=self.image_size, patch=self.patch_size)
            # Build the color thumbnail from the *same* pixels the model sees,
            # not from the original frame — otherwise the center-crop in
            # _preprocess_bgr misaligns color (stretched full view) vs depth
            # (cropped strip) and colors come from the wrong scene region.
            thumb = (tensor.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            seq += 1
            snap.append((tensor, thumb, seq))
        cap.release()
        return snap, meta

    async def push_frame(self, bgr: np.ndarray) -> None:
        """Called from the WebRTC track consumer.

        Throttled to ~10 fps — WebRTC delivers 30 fps but we run inference every
        few seconds, so preprocessing every frame wastes CPU. The buffer's deque
        maxlen further trims old frames.
        """
        now = time.time()
        if now - self._last_push_ts < 0.1:
            return
        self._last_push_ts = now
        tensor = _preprocess_bgr(bgr, target_w=self.image_size, patch=self.patch_size)
        rgb_thumb = (tensor.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        self._frame_seq += 1
        item = (tensor, rgb_thumb, self._frame_seq)
        async with self._lock:
            if len(self.anchor) < self.num_scale_frames:
                self.anchor.append(item)
            else:
                self.buffer.append(item)

    async def _loop(self) -> None:
        """Periodic inference. Skips ticks while one inference is still running
        to prevent GPU pile-up that previously caused the 100% GPU lockup.
        Streaming path always emits LBP2 points (TSDF mesh is too slow for live)."""
        log.info("inference loop running (every %.1fs)", self.interval_s)
        loop = asyncio.get_running_loop()
        busy = False
        # Force point output during streaming regardless of self.output_mode.
        prev_mode = self.output_mode
        while not self._stop.is_set():
            await asyncio.sleep(self.interval_s)
            if busy:
                log.debug("skip tick — previous inference still running")
                continue
            async with self._lock:
                if len(self.anchor) + len(self.buffer) < 4:
                    continue
                tail_cap = max(0, self.window_size - len(self.anchor))
                snap = list(self.anchor) + list(self.buffer)[-tail_cap:]
            busy = True
            self.output_mode = "points"  # never run TSDF in the live path
            try:
                t0 = time.time()
                payload = await loop.run_in_executor(None, self._infer_sync, snap)
                log.info("tick: N=%d in %.2fs", len(snap), time.time() - t0)
            except Exception as e:
                log.exception("inference failed: %s", e)
                payload = None
            finally:
                self.output_mode = prev_mode
                busy = False
            if payload is not None:
                try:
                    await self.broadcast_fn(payload)
                except Exception as e:
                    log.warning("broadcast failed: %s", e)

    def _infer_sync(self, snap: list[tuple[torch.Tensor, np.ndarray, int]]) -> bytes | None:
        # Re-infer the *entire* current buffer every tick so all cameras + points
        # share one anchor (lingbot resets KV cache on each call; cross-batch
        # frames otherwise live in different world frames).
        tensors = [t[0] for t in snap]
        thumbs = [t[1] for t in snap]
        seqs = [t[2] for t in snap]
        self._last_emit_seq = max(seqs)

        images = torch.stack(tensors, dim=0).unsqueeze(0).to(self.device)  # (1, N, 3, H, W)
        N, _, H, W = images.shape[1:]

        t0 = time.time()
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            preds = self.model.inference_streaming(
                images.squeeze(0),  # model wants (N,3,H,W)
                num_scale_frames=self.num_scale_frames,
                keyframe_interval=1,
                output_device=None,
            )
        dt = time.time() - t0

        # squeeze possible batch dim
        for k in list(preds.keys()):
            if isinstance(preds[k], torch.Tensor) and preds[k].dim() >= 4 and preds[k].shape[0] == 1:
                preds[k] = preds[k][0]

        # poses — strip batch dim if present
        w2c, c2w, K = _poses_from_pose_enc(preds["pose_enc"], (H, W))
        if w2c.dim() == 4 and w2c.shape[0] == 1: w2c = w2c[0]
        if c2w.dim() == 4 and c2w.shape[0] == 1: c2w = c2w[0]
        if K.dim()   == 4 and K.shape[0]   == 1: K   = K[0]
        w2c_np = w2c.detach().cpu().numpy().astype(np.float32)
        c2w_np = c2w.detach().cpu().numpy().astype(np.float32)
        K_np   = K.detach().cpu().numpy().astype(np.float32)

        # Build world points from depth + w2c + K. The lingbot-map-long.pt
        # checkpoint ships without point_head weights, so the model's native
        # world_points are noise in a different coordinate frame than the camera
        # poses. Depth + pose unprojection guarantees a shared frame.
        depth = preds.get("depth")
        if depth is None:
            log.warning("model did not produce depth; skipping")
            return None
        depth_np = depth.detach().cpu().numpy().astype(np.float32)
        if depth_np.ndim == 3:
            depth_np = depth_np[..., None]
        wp = unproject_depth_map_to_point_map(depth_np, w2c_np, K_np)
        dc = preds.get("depth_conf")
        wpc = dc.detach().cpu().numpy() if dc is not None else None

        # Demo-style filtering: keep points with depth_conf > conf_threshold (1.5).
        # Carry confidence through to fusion as observation weight.
        all_xyz, all_rgb, all_conf = [], [], []
        for i in range(N):
            pts = wp[i].reshape(-1, 3)
            cols = thumbs[i].reshape(-1, 3).astype(np.uint8)
            conf = wpc[i].reshape(-1) if wpc is not None else None
            if conf is not None:
                mask = conf > self.conf_threshold
            else:
                mask = np.ones(pts.shape[0], dtype=bool)
            if mask.sum() == 0:
                continue
            idx = np.where(mask)[0]
            if idx.size > self.max_points_per_frame:
                idx = np.random.default_rng(int(seqs[i])).choice(
                    idx, self.max_points_per_frame, replace=False)
            all_xyz.append(pts[idx])
            all_rgb.append(cols[idx])
            if conf is not None:
                all_conf.append(conf[idx])
            else:
                all_conf.append(np.ones(idx.size, dtype=np.float32))

        if not all_xyz:
            return None
        xyz  = np.concatenate(all_xyz,  axis=0).astype(np.float32)
        rgb  = np.concatenate(all_rgb,  axis=0).astype(np.uint8)
        conf = np.concatenate(all_conf, axis=0).astype(np.float32)

        # Tier 1 — confidence-weighted voxel fusion at 1.5 cm.
        # Multi-view observations of the same surface collapse to one point.
        # Keep all voxels (min_obs=1) so unique observations survive; the
        # radius filter below handles isolated noise instead.
        n_before = xyz.shape[0]
        xyz, rgb, weights = _voxel_fuse_weighted(xyz, rgb, conf,
                                                  voxel_size=0.015, min_obs=1)
        n_voxel = xyz.shape[0]
        # Radius outlier filter: drop floaters with <3 neighbors in 4 cm.
        xyz, rgb = _radius_outlier_filter(xyz, rgb, radius=0.04, min_neighbors=3)
        log.info("Tier1 fusion: %d → voxel %d → outlier %d points",
                 n_before, n_voxel, xyz.shape[0])

        # No coordinate transform — pass OpenCV world coords directly (same as demo/viser).
        # The Three.js viewer handles orientation via camera positioning.

        # diagnostic: compare camera trajectory extent vs point cloud extent
        cam_centers = c2w_np[:, :, 3]  # (N, 3)
        cam_span = (cam_centers.max(0) - cam_centers.min(0))
        cam_centroid = cam_centers.mean(0)
        pts_span = (xyz.max(0) - xyz.min(0))
        pts_centroid = xyz.mean(0)
        log.info(
            "infer N=%d pts=%d in %.2fs | conf>%.2f "
            "| cam_span=[%.2f,%.2f,%.2f] @ %s "
            "| pts_span=[%.2f,%.2f,%.2f] @ %s",
            N, xyz.shape[0], dt, self.conf_threshold,
            *cam_span, np.round(cam_centroid, 2).tolist(),
            *pts_span, np.round(pts_centroid, 2).tolist(),
        )

        # Tier 2 — TSDF fusion + Marching Cubes mesh
        if self.output_mode == "mesh":
            try:
                from tsdf import build_tsdf_volume, extract_mesh, color_vertices
                # depth_hw: (N, H, W), wpc: (N, H, W) confidence map
                depth_hw = depth_np.squeeze(-1) if depth_np.ndim == 4 else depth_np
                t_tsdf = time.time()
                vol = build_tsdf_volume(
                    depth_hw, w2c_np, K_np, wpc,
                    voxel_size=self.tsdf_voxel, trunc=self.tsdf_trunc,
                    conf_floor=self.conf_threshold,
                )
                verts, faces, normals = extract_mesh(vol, min_weight=1.0)
                # Color vertices from images: thumbs is (N, H, W, 3) uint8
                imgs = np.stack(thumbs, axis=0)
                v_rgb = color_vertices(verts, imgs, w2c_np, K_np)
                log.info("Tier2 TSDF: verts=%d faces=%d in %.2fs (vol dims=%s)",
                         verts.shape[0], faces.shape[0], time.time() - t_tsdf,
                         vol["tsdf"].shape)

                # LBM1 binary: magic + flags + num_verts + num_faces + num_poses + seq
                #   verts f32(V*3) | rgb u8(V*3) | faces u32(F*3) | poses f32(N*12) | K f32(9)
                if verts.shape[0] > 0 and faces.shape[0] > 0:
                    header = struct.pack("<4sIIIII", b"LBM1", 1,
                                         verts.shape[0], faces.shape[0],
                                         c2w_np.shape[0], self._frame_seq)
                    body = (
                        verts.astype(np.float32).tobytes()
                        + v_rgb.astype(np.uint8).tobytes()
                        + faces.astype(np.uint32).tobytes()
                        + c2w_np.astype(np.float32).tobytes()
                        + K_np[0].astype(np.float32).tobytes()
                    )
                    return header + body
                log.warning("TSDF produced empty mesh — falling back to LBP2 points")
            except Exception as e:
                log.exception("TSDF mesh extraction failed, falling back to LBP2: %s", e)

        # Binary v2 — full snapshot for this tick. Viewer must replace its scene.
        #   magic 'LBP2' (4) | flags u32(1=replace) | num_pts u32 | num_poses u32 | seq u32
        #   xyz f32 (num_pts*3) | rgb u8 (num_pts*3)
        #   poses f32 (num_poses*12 row-major 3x4 c2w) | K0 f32 9
        flags = 1
        header = struct.pack("<4sIIII", b"LBP2", flags, xyz.shape[0], c2w_np.shape[0], self._frame_seq)
        body = (
            xyz.tobytes()
            + rgb.tobytes()
            + c2w_np.astype(np.float32).tobytes()
            + K_np[0].astype(np.float32).tobytes()
        )
        return header + body

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
        window_size: int = 64,         # max frames per inference; whole buffer is used
        interval_s: float = 2.5,
        max_buffer: int = 64,
        max_points_per_frame: int = 30000,
        conf_threshold: float = 1.5,     # demo's absolute conf cutoff (vis_threshold)
        num_scale_frames: int = 16,      # more anchor frames → better global consistency
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
        # Streaming loop is disabled in upload mode — model stays warm but no
        # periodic inference runs. Use `process_video_file` instead.
        return

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await self._task

    async def process_video_file(
        self, path: Path, *, target_frames: int = 32
    ) -> tuple[bytes | None, list[bytes], dict]:
        """Sample `target_frames` evenly from the video, run inference once,
        broadcast the LBP2 payload + per-frame thumbnails, and return
        (payload, thumbs_jpeg, info)."""
        loop = asyncio.get_running_loop()
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
        """Called from the WebRTC track consumer."""
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
        log.info("inference loop running (every %.1fs)", self.interval_s)
        loop = asyncio.get_running_loop()
        while not self._stop.is_set():
            await asyncio.sleep(self.interval_s)
            async with self._lock:
                if len(self.anchor) + len(self.buffer) < 4:
                    continue
                tail_cap = max(0, self.window_size - len(self.anchor))
                snap = list(self.anchor) + list(self.buffer)[-tail_cap:]
            try:
                payload = await loop.run_in_executor(None, self._infer_sync, snap)
            except Exception as e:
                log.exception("inference failed: %s", e)
                continue
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
        # Same as viser_wrapper.py's `(conf >= threshold_val) & (conf > 0.1)`.
        all_xyz, all_rgb = [], []
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

        if not all_xyz:
            return None
        xyz = np.concatenate(all_xyz, axis=0).astype(np.float32)
        rgb = np.concatenate(all_rgb, axis=0).astype(np.uint8)

        # Voxel downsample at 8 mm: dedupes overlapping observations from different
        # frames hitting the same surface, but small enough to preserve detail.
        xyz, rgb = _voxel_downsample(xyz, rgb, voxel_size=0.008)

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

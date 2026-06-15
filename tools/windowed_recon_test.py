"""Compare multi-window merge: OLD (per-window streaming + Procrustes/ICP chain)
vs NEW (upstream model.inference_windowed). Same source frames, same fusion.

Renders orthographic z-buffered captures + camera trajectory per video so the
merge quality difference is visible. Writes PNGs + stats JSON to --out.

Usage:
    .venv/bin/python tools/windowed_recon_test.py \
        --model ckpts/lingbot-map-long.pt \
        --videos video/pxx.mp4 realtime/_uploads/upload_1779361465494.mp4 \
        --out reports/windowed_compare
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "realtime"))

from lingbot_map.models.gct_stream_window import GCTStream
from lingbot_map.utils.geometry import unproject_depth_map_to_point_map
from inference_worker import (
    _preprocess_bgr, _poses_from_pose_enc,
    _voxel_fuse_weighted, _radius_outlier_keep_mask,
)
from registration import chain_windows

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONF_THRESHOLD = 1.5
MAX_PTS_PER_FRAME = 270000
TARGET_PER_WINDOW = 48        # OLD worker constant
OVERLAP = 16                  # OLD worker constant
NUM_SCALE = 16               # both paths


# ----------------------------------------------------------------------------- sampling
def probe(path):
    cap = cv2.VideoCapture(path)
    info = dict(
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        fps=round(float(cap.get(cv2.CAP_PROP_FPS) or 0), 2),
        rotation=int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0),
    )
    cap.release()
    return info


def sample_by_indices(path, indices, rotation, image_size=518, patch=14):
    """Mirror inference_worker._sample_video_by_indices preprocessing+rotation."""
    cap = cv2.VideoCapture(path)
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1.0)
    except Exception:
        pass
    rot_map = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
               270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    rc = rot_map.get(rotation)
    out = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue
        if rc is not None and rotation in (90, 270) and bgr.shape[1] > bgr.shape[0]:
            bgr = cv2.rotate(bgr, rc)
        elif rc == cv2.ROTATE_180:
            bgr = cv2.rotate(bgr, rc)
        t = _preprocess_bgr(bgr, target_w=image_size, patch=patch)
        thumb = (t.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        out.append((t, thumb, int(idx) + 1))
    cap.release()
    return out


# ----------------------------------------------------------------------------- per-frame -> points
def _points_from_preds(preds, snap, H, W):
    """Unproject depth -> world points + colors + conf + source. Shared by both paths."""
    for k in list(preds.keys()):
        if isinstance(preds[k], torch.Tensor) and preds[k].dim() >= 4 and preds[k].shape[0] == 1:
            preds[k] = preds[k][0]
    w2c, c2w, K = _poses_from_pose_enc(preds["pose_enc"], (H, W))
    if w2c.dim() == 4 and w2c.shape[0] == 1: w2c = w2c[0]
    if c2w.dim() == 4 and c2w.shape[0] == 1: c2w = c2w[0]
    if K.dim() == 4 and K.shape[0] == 1: K = K[0]
    w2c_np = w2c.cpu().numpy().astype(np.float32)
    c2w_np = c2w.cpu().numpy().astype(np.float32)
    K_np = K.cpu().numpy().astype(np.float32)
    depth = preds["depth"].cpu().numpy().astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., None]
    wp = unproject_depth_map_to_point_map(depth, w2c_np, K_np)
    dc = preds.get("depth_conf")
    wpc = dc.cpu().numpy() if dc is not None else None
    thumbs = [s[1] for s in snap]
    seqs = [s[2] for s in snap]
    N = len(snap)
    xs, rs, cs, ss = [], [], [], []
    for i in range(N):
        pts = wp[i].reshape(-1, 3)
        cols = thumbs[i].reshape(-1, 3).astype(np.uint8)
        conf = wpc[i].reshape(-1) if wpc is not None else None
        mask = conf > CONF_THRESHOLD if conf is not None else np.ones(pts.shape[0], bool)
        if mask.sum() == 0:
            continue
        idx = np.where(mask)[0]
        if idx.size > MAX_PTS_PER_FRAME:
            idx = np.random.default_rng(int(seqs[i])).choice(idx, MAX_PTS_PER_FRAME, replace=False)
        xs.append(pts[idx]); rs.append(cols[idx])
        cs.append(conf[idx] if conf is not None else np.ones(idx.size, np.float32))
        ss.append(np.full(idx.size, int(seqs[i]), np.uint32))
    return dict(
        xyz=np.concatenate(xs).astype(np.float32),
        rgb=np.concatenate(rs).astype(np.uint8),
        conf=np.concatenate(cs).astype(np.float32),
        source=np.concatenate(ss).astype(np.uint32),
        frame_ids=np.asarray(seqs, np.uint32),
        c2w=c2w_np, w2c=w2c_np, K=K_np,
        depth=depth.squeeze(-1), thumbs=thumbs,
    )


def _fuse(xyz, rgb, conf, source):
    """Tier1 fusion identical to inference_worker._fuse_merged (fair to both)."""
    xyz, rgb, _, source = _voxel_fuse_weighted(xyz, rgb, conf, voxel_size=0.015,
                                               min_obs=1, labels=source)
    keep = _radius_outlier_keep_mask(xyz, radius=0.04, min_neighbors=3)
    return xyz[keep], rgb[keep], source[keep]


# ----------------------------------------------------------------------------- pipelines
def run_old(model, path, info, device):
    """Worker multi-window: per-window inference_streaming + chain_windows."""
    total = info["total"]
    num_windows = max(2, min(4, total // (TARGET_PER_WINDOW * 2)))
    eff_len = TARGET_PER_WINDOW * num_windows - OVERLAP * (num_windows - 1)
    virt = np.linspace(0, total - 1, eff_len).astype(int)
    windows = []
    for k in range(num_windows):
        s = k * (TARGET_PER_WINDOW - OVERLAP)
        real_idx = [int(virt[i]) for i in range(s, s + TARGET_PER_WINDOW)]
        snap = sample_by_indices(path, real_idx, info["rotation"])
        if not snap:
            continue
        tens = torch.stack([s_[0] for s_ in snap]).to(device)
        H, W = tens.shape[-2:]
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            preds = model.inference_streaming(tens, num_scale_frames=NUM_SCALE,
                                              keyframe_interval=1, output_device=None)
        windows.append(_points_from_preds(preds, snap, H, W))
    merged = chain_windows(windows, overlap=OVERLAP)
    xyz, rgb, src = _fuse(merged["xyz"], merged["rgb"], merged["conf"],
                          merged.get("source", np.zeros(merged["xyz"].shape[0], np.uint32)))
    return dict(xyz=xyz, rgb=rgb, c2w=merged["c2w"], n_windows=num_windows,
                scales=merged.get("scales_to_ref"))


def run_new(model, path, info, device):
    """Upstream model.inference_windowed on the same frame set."""
    total = info["total"]
    num_windows = max(2, min(4, total // (TARGET_PER_WINDOW * 2)))
    eff_len = TARGET_PER_WINDOW * num_windows - OVERLAP * (num_windows - 1)
    virt = np.linspace(0, total - 1, eff_len).astype(int)
    snap = sample_by_indices(path, [int(v) for v in virt], info["rotation"])
    tens = torch.stack([s_[0] for s_ in snap]).to(device)
    H, W = tens.shape[-2:]
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        preds = model.inference_windowed(
            tens, window_size=32, overlap_size=OVERLAP,
            num_scale_frames=NUM_SCALE, keyframe_interval=1, output_device=None)
    chunk_scales = preds.get("chunk_scales")
    d = _points_from_preds(preds, snap, H, W)
    xyz, rgb, src = _fuse(d["xyz"], d["rgb"], d["conf"], d["source"])
    return dict(xyz=xyz, rgb=rgb, c2w=d["c2w"],
                chunk_scales=chunk_scales.flatten().tolist() if chunk_scales is not None else None)


# ----------------------------------------------------------------------------- rendering
def ortho(xyz, rgb, ax_a, ax_b, depth_ax, res=900):
    """Orthographic z-buffered splat onto (ax_a, ax_b); nearest depth wins."""
    p2 = xyz[:, [ax_a, ax_b]].astype(np.float64)
    z = xyz[:, depth_ax].astype(np.float64)
    mn, mx = p2.min(0), p2.max(0)
    span = np.maximum(mx - mn, 1e-6)
    scale = (res * 0.92) / span.max()
    px = ((p2 - (mn + mx) / 2) * scale + res / 2)
    x = px[:, 0].astype(int)
    y = (res - 1 - px[:, 1]).astype(int)
    m = (x >= 0) & (x < res) & (y >= 0) & (y < res)
    x, y, z, c = x[m], y[m], z[m], rgb[m]
    order = np.argsort(-z)            # far first; near painted last (overwrites)
    x, y, c = x[order], y[order], c[order]
    img = np.full((res, res, 3), 250, np.uint8)
    for dx in (0, 1):                 # 2x2 splat for density
        for dy in (0, 1):
            xx = np.clip(x + dx, 0, res - 1); yy = np.clip(y + dy, 0, res - 1)
            img[yy, xx] = c
    return img


def downsample(xyz, rgb, n=180000):
    if xyz.shape[0] <= n:
        return xyz, rgb
    idx = np.random.default_rng(0).choice(xyz.shape[0], n, replace=False)
    return xyz[idx], rgb[idx]


def gravity_basis(c2w):
    """Estimate a [right, up, forward] world basis from camera poses so we can
    render a gravity-aligned floor plan (top-down) + elevation. In OpenCV cams,
    +Y is image-down, so world-up ~= -mean(camera Y axis in world)."""
    R = c2w[:, :3, :3]
    up = -R[:, :, 1].mean(0)
    up = up / (np.linalg.norm(up) + 1e-9)
    fwd = R[:, :, 2].mean(0)                    # mean viewing direction
    fwd = fwd - up * (fwd @ up)
    if np.linalg.norm(fwd) < 1e-6:
        fwd = np.array([1.0, 0.0, 0.0]) - up * up[0]
    fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
    right = np.cross(fwd, up); right = right / (np.linalg.norm(right) + 1e-9)
    return np.stack([right, up, fwd], axis=1)  # (3,3), columns are basis vectors


def render_compare(name, old, new, out_dir):
    """One figure: rows OLD/NEW, cols = gravity-aligned top-down + elevation."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 13))
    for row, (tag, rec) in enumerate([("OLD  (Procrustes+ICP chain)", old),
                                      ("NEW  (upstream inference_windowed)", new)]):
        xyz, rgb = downsample(rec["xyz"], rec["rgb"])
        B = gravity_basis(rec["c2w"])
        loc = xyz @ B                            # cols: 0=right 1=up 2=forward
        # top-down (floor plan): right vs forward, depth=up ; elevation: right vs up, depth=forward
        views = [(0, 2, 1, "top-down (floor plan)"), (0, 1, 2, "elevation")]
        for col, (a, b, d, label) in enumerate(views):
            ax = axes[row, col]
            ax.imshow(ortho(loc, rgb, a, b, d))
            ax.set_title(f"{tag}\n{label}  |  {rec['xyz'].shape[0]:,} pts", fontsize=10)
            ax.axis("off")
    fig.suptitle(f"{name} — multi-window merge comparison (gravity-aligned)",
                 fontsize=14, y=0.995)
    fig.tight_layout()
    p = out_dir / f"{name}_compare.png"
    fig.savefig(p, dpi=90, bbox_inches="tight")
    plt.close(fig)
    return p


def render_traj(name, old, new, out_dir):
    """Camera-center trajectory: bad stitching folds/jumps at window seams."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (tag, rec) in zip(axes, [("OLD", old), ("NEW", new)]):
        B = gravity_basis(rec["c2w"])
        c = rec["c2w"][:, :3, 3] @ B           # floor plane = right(0) vs forward(2)
        a, b = 0, 2
        ax.scatter(c[:, a], c[:, b], c=np.arange(len(c)), cmap="viridis", s=18)
        ax.plot(c[:, a], c[:, b], "-", color="gray", lw=0.6, alpha=0.6)
        ax.set_title(f"{tag} camera trajectory ({len(c)} frames)")
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
    fig.suptitle(f"{name} — camera trajectory (color = frame order)", fontsize=13)
    fig.tight_layout()
    p = out_dir / f"{name}_traj.png"
    fig.savefig(p, dpi=90, bbox_inches="tight")
    plt.close(fig)
    return p


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ckpts/lingbot-map-long.pt")
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--out", default="reports/windowed_compare")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    print("loading model (gct_stream_window) ...")
    t0 = time.time()
    model = GCTStream(img_size=518, patch_size=14, enable_3d_rope=True, max_frame_num=1024,
                      kv_cache_sliding_window=64, kv_cache_scale_frames=8,
                      kv_cache_cross_frame_special=True, kv_cache_include_scale_frames=True,
                      use_sdpa=True, camera_num_iterations=4, enable_point=False)
    ck = torch.load(args.model, map_location=device, weights_only=False)
    sd = ck.get("model", ck)
    miss, unexp = model.load_state_dict(sd, strict=False)
    model.aggregator.to(dtype=torch.bfloat16); model.to(device).eval()
    print(f"  model ready missing={len(miss)} unexpected={len(unexp)} in {time.time()-t0:.1f}s")

    stats = []
    for vp in args.videos:
        name = Path(vp).stem
        info = probe(vp)
        print(f"\n=== {name}  frames={info['total']} fps={info['fps']} rot={info['rotation']} ===")
        t = time.time(); old = run_old(model, vp, info, device)
        print(f"  OLD done {time.time()-t:.1f}s  pts={old['xyz'].shape[0]:,}  scales={old.get('scales')}")
        t = time.time(); new = run_new(model, vp, info, device)
        print(f"  NEW done {time.time()-t:.1f}s  pts={new['xyz'].shape[0]:,}  chunk_scales={new.get('chunk_scales')}")
        p1 = render_compare(name, old, new, out_dir)
        p2 = render_traj(name, old, new, out_dir)
        print(f"  rendered: {p1.name}, {p2.name}")

        def bbox(r): return (r["xyz"].max(0) - r["xyz"].min(0)).round(2).tolist()
        stats.append(dict(
            name=name, frames=info["total"], fps=info["fps"], rotation=info["rotation"],
            old_pts=int(old["xyz"].shape[0]), new_pts=int(new["xyz"].shape[0]),
            old_bbox=bbox(old), new_bbox=bbox(new),
            old_scales=old.get("scales"), new_chunk_scales=new.get("chunk_scales"),
            compare_png=str(p1), traj_png=str(p2),
        ))

    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nstats.json + PNGs written to {out_dir}")


if __name__ == "__main__":
    main()

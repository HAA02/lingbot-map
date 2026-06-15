"""Analyze a capture video on the latest-main upstream pipeline.

Mirrors demo.py's default streaming path (single forward pass -> one globally
consistent world frame, depth unprojection), then renders gravity-aligned
captures + camera trajectory + stats for the reconstruction. Self-contained:
uses only lingbot_map core (no realtime/, no viser, no export module).

Usage:
    .venv/bin/python tools/analyze_pxx.py \
        --model ckpts/lingbot-map-long.pt \
        --video video/pxx.mp4 --fps 10 --out reports/pxx_analysis
"""
from __future__ import annotations
import argparse, json, os, tempfile, time
from pathlib import Path

import cv2
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lingbot_map.models.gct_stream import GCTStream
from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import (
    closed_form_inverse_se3_general, unproject_depth_map_to_point_map,
)

CONF_THRESHOLD = 1.5     # demo vis_threshold
NUM_SCALE = 8            # demo default


# --------------------------------------------------------------------------- io
def extract_frames(video, out_dir, fps):
    cap = cv2.VideoCapture(video)
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1.0)
    except Exception:
        pass
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    interval = max(1, round(src / fps))
    os.makedirs(out_dir, exist_ok=True)
    paths, idx, saved = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % interval == 0:
            p = os.path.join(out_dir, f"{saved:06d}.jpg")
            cv2.imwrite(p, frame)
            paths.append(p); saved += 1
        idx += 1
    cap.release()
    return paths, dict(total_frames=total, src_fps=round(src, 2), interval=interval)


# --------------------------------------------------------------------- rendering
def gravity_basis(c2w):
    R = c2w[:, :3, :3]
    up = -R[:, :, 1].mean(0); up /= (np.linalg.norm(up) + 1e-9)
    fwd = R[:, :, 2].mean(0); fwd = fwd - up * (fwd @ up)
    if np.linalg.norm(fwd) < 1e-6:
        fwd = np.array([1.0, 0, 0]) - up * up[0]
    fwd /= (np.linalg.norm(fwd) + 1e-9)
    right = np.cross(fwd, up); right /= (np.linalg.norm(right) + 1e-9)
    return np.stack([right, up, fwd], axis=1)


def ortho(xyz, rgb, a, b, d, res=1000):
    p2 = xyz[:, [a, b]].astype(np.float64)
    z = xyz[:, d].astype(np.float64)
    mn, mx = p2.min(0), p2.max(0)
    span = np.maximum(mx - mn, 1e-6)
    scale = (res * 0.92) / span.max()
    px = ((p2 - (mn + mx) / 2) * scale + res / 2)
    x = px[:, 0].astype(int); y = (res - 1 - px[:, 1]).astype(int)
    m = (x >= 0) & (x < res) & (y >= 0) & (y < res)
    x, y, z, c = x[m], y[m], z[m], rgb[m]
    order = np.argsort(-z)
    x, y, c = x[order], y[order], c[order]
    img = np.full((res, res, 3), 250, np.uint8)
    for dx in (0, 1):
        for dy in (0, 1):
            img[np.clip(y + dy, 0, res - 1), np.clip(x + dx, 0, res - 1)] = c
    return img


def downsample(xyz, rgb, n=220000):
    if xyz.shape[0] <= n:
        return xyz, rgb
    i = np.random.default_rng(0).choice(xyz.shape[0], n, replace=False)
    return xyz[i], rgb[i]


def voxel_mean(xyz, rgb, v=0.01):
    """Average points per voxel cell (cleans dense multi-view depth)."""
    keys = np.floor(xyz / v).astype(np.int64)
    packed = (keys[:, 0] * 1000003 + keys[:, 1]) * 1000003 + keys[:, 2]
    _, inv, counts = np.unique(packed, return_inverse=True, return_counts=True)
    n = len(counts)
    sx = np.zeros((n, 3)); sc = np.zeros((n, 3))
    np.add.at(sx, inv, xyz); np.add.at(sc, inv, rgb.astype(np.float64))
    c = counts[:, None]
    return (sx / c).astype(np.float32), (sc / c).clip(0, 255).astype(np.uint8)


def radius_keep(xyz, r=0.04, min_nbr=4):
    from scipy.spatial import cKDTree
    if xyz.shape[0] < min_nbr + 1:
        return np.ones(xyz.shape[0], bool)
    counts = cKDTree(xyz).query_ball_point(xyz, r=r, return_length=True)
    return counts >= (min_nbr + 1)


def render_views(name, xyz, rgb, c2w, out_dir):
    B = gravity_basis(c2w)
    loc = xyz @ B
    ds_loc, ds_rgb = downsample(loc, rgb)
    fig, ax = plt.subplots(1, 2, figsize=(15, 7.6))
    ax[0].imshow(ortho(ds_loc, ds_rgb, 0, 2, 1)); ax[0].set_title(f"top-down (floor plan) | {xyz.shape[0]:,} pts", fontsize=11); ax[0].axis("off")
    ax[1].imshow(ortho(ds_loc, ds_rgb, 0, 1, 2)); ax[1].set_title("elevation", fontsize=11); ax[1].axis("off")
    fig.suptitle(f"{name} — point cloud (gravity-aligned, latest-main streaming)", fontsize=14)
    fig.tight_layout()
    p1 = out_dir / f"{name}_cloud.png"; fig.savefig(p1, dpi=95, bbox_inches="tight"); plt.close(fig)

    # oblique 3D
    fig = plt.figure(figsize=(9, 8)); ax3 = fig.add_subplot(111, projection="3d")
    o_loc, o_rgb = downsample(loc, rgb, 90000)
    ax3.scatter(o_loc[:, 0], o_loc[:, 2], o_loc[:, 1], c=o_rgb / 255.0, s=0.6, marker=".", linewidths=0)
    ax3.view_init(elev=22, azim=-60); ax3.set_box_aspect((1, 1, 0.5))
    ax3.set_xlabel("right"); ax3.set_ylabel("forward"); ax3.set_zlabel("up")
    ax3.set_title(f"{name} — oblique 3D view")
    p2 = out_dir / f"{name}_oblique.png"; fig.savefig(p2, dpi=95, bbox_inches="tight"); plt.close(fig)

    # trajectory (floor plane)
    cc = c2w[:, :3, 3] @ B
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(cc[:, 0], cc[:, 2], c=np.arange(len(cc)), cmap="viridis", s=22)
    ax.plot(cc[:, 0], cc[:, 2], "-", color="gray", lw=0.7, alpha=0.6)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_title(f"{name} — camera trajectory ({len(cc)} frames, color=order)")
    fig.colorbar(sc, ax=ax, label="frame index")
    p3 = out_dir / f"{name}_traj.png"; fig.savefig(p3, dpi=95, bbox_inches="tight"); plt.close(fig)
    return [p1, p2, p3]


# -------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ckpts/lingbot-map-long.pt")
    ap.add_argument("--video", default="video/pxx.mp4")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", default="reports/pxx_analysis")
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(args.video).stem
    device = "cuda"

    with tempfile.TemporaryDirectory(prefix="pxx_frames_") as fdir:
        print("extracting frames ...")
        paths, vmeta = extract_frames(args.video, fdir, args.fps)
        print(f"  {len(paths)} frames @ fps={args.fps} (src {vmeta['src_fps']}, total {vmeta['total_frames']}, interval {vmeta['interval']})")
        images = load_and_preprocess_images(paths, mode="crop", image_size=518, patch_size=14)
    S, _, H, W = images.shape
    print(f"  preprocessed: {S} x {W}x{H}")

    print("loading model (streaming, latest main) ...")
    t0 = time.time()
    model = GCTStream(img_size=518, patch_size=14, enable_3d_rope=True, max_frame_num=1024,
                      kv_cache_sliding_window=64, kv_cache_scale_frames=NUM_SCALE,
                      kv_cache_cross_frame_special=True, kv_cache_include_scale_frames=True,
                      use_sdpa=True, camera_num_iterations=4, enable_point=False)
    ck = torch.load(args.model, map_location=device, weights_only=False)
    sd = ck.get("model", ck); miss, unexp = model.load_state_dict(sd, strict=False)
    model.aggregator.to(dtype=torch.bfloat16); model.to(device).eval()
    print(f"  ready missing={len(miss)} unexpected={len(unexp)} in {time.time()-t0:.1f}s")

    kf = 1 if S <= 320 else (S + 319) // 320
    print(f"streaming inference (S={S}, keyframe_interval={kf}) ...")
    t0 = time.time()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        preds = model.inference_streaming(images.to(device), num_scale_frames=NUM_SCALE,
                                          keyframe_interval=kf, output_device=None)
    print(f"  done in {time.time()-t0:.1f}s")

    for k in list(preds.keys()):
        if isinstance(preds[k], torch.Tensor) and preds[k].dim() >= 4 and preds[k].shape[0] == 1:
            preds[k] = preds[k][0]
    w2c, K = pose_encoding_to_extri_intri(preds["pose_enc"], (H, W))
    if w2c.dim() == 4 and w2c.shape[0] == 1: w2c = w2c[0]
    if K.dim() == 4 and K.shape[0] == 1: K = K[0]
    w2c_np = w2c.detach().cpu().numpy().astype(np.float32)
    K_np = K.detach().cpu().numpy().astype(np.float32)
    # camera centers via c2w
    e44 = np.tile(np.eye(4, dtype=np.float32)[None], (w2c_np.shape[0], 1, 1)); e44[:, :3, :4] = w2c_np
    c2w_np = closed_form_inverse_se3_general(torch.from_numpy(e44)).numpy()[:, :3, :]

    depth = preds["depth"].detach().cpu().numpy().astype(np.float32)
    if depth.ndim == 3: depth = depth[..., None]
    wp = unproject_depth_map_to_point_map(depth, w2c_np, K_np)   # w2c per docstring
    conf = preds["depth_conf"].detach().cpu().numpy()
    imgs = (images.permute(0, 2, 3, 1).numpy() * 255).clip(0, 255).astype(np.uint8)

    xs, rs = [], []
    for i in range(S):
        m = conf[i].reshape(-1) > CONF_THRESHOLD
        if m.sum() == 0: continue
        xs.append(wp[i].reshape(-1, 3)[m]); rs.append(imgs[i].reshape(-1, 3)[m])
    xyz = np.concatenate(xs).astype(np.float32); rgb = np.concatenate(rs).astype(np.uint8)
    raw_pts = int(xyz.shape[0])
    print(f"points: {raw_pts:,} (conf>{CONF_THRESHOLD})")

    # Clean for visualization: voxel-average (1cm) then drop floaters.
    xyz, rgb = voxel_mean(xyz, rgb, v=0.01)
    keep = radius_keep(xyz, r=0.04, min_nbr=4)
    xyz, rgb = xyz[keep], rgb[keep]
    print(f"cleaned: {xyz.shape[0]:,} (voxel 1cm + radius outlier)")

    figs = render_views(name, xyz, rgb, c2w_np, out_dir)
    cc = c2w_np[:, :3, 3]
    traj_len = float(np.linalg.norm(np.diff(cc, axis=0), axis=1).sum())
    stats = dict(
        name=name, **vmeta, frames_used=S, hw=[int(H), int(W)],
        keyframe_interval=int(kf), points_raw=raw_pts, points_clean=int(xyz.shape[0]),
        conf_threshold=CONF_THRESHOLD,
        bbox_m=(xyz.max(0) - xyz.min(0)).round(3).tolist(),
        trajectory_len_m=round(traj_len, 3),
        captures=[str(f) for f in figs],
    )
    (out_dir / f"{name}_stats.json").write_text(json.dumps(stats, indent=2))
    print("stats:", json.dumps(stats, indent=2))
    print(f"\nwrote {len(figs)} captures + stats to {out_dir}")


if __name__ == "__main__":
    main()

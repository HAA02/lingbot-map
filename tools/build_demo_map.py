"""Dense, demo-style point-cloud map straight from video — uses the SAME path as
demo.py (model.inference_streaming + depth unprojection, no voxel-fusion / no
180k subsample that the realtime coverage worker applies). Produces a far denser,
more photoreal cloud, then writes the self-contained interactive map HTML.

Usage:
    .venv/bin/python tools/build_demo_map.py \
        --model ckpts/lingbot-map-long.pt \
        --videos video/a.mp4:cam1 video/b.mp4:cam2 \
        --fps 12 --keep 500000 --out reports/pointcloud_map/demo_map.html
"""
from __future__ import annotations
import argparse, base64, json, os, tempfile, time
from pathlib import Path
import cv2, numpy as np, torch

from lingbot_map.models.gct_stream import GCTStream
from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import closed_form_inverse_se3_general, unproject_depth_map_to_point_map
from build_pointcloud_map_html import TEMPLATE   # reuse the viewer template

CONF = 1.5
NUM_SCALE = 8
PALETTE = ["#5aa7ff", "#ff7eb6", "#7CFFB2", "#ffd166"]


def extract(video, d, fps):
    cap = cv2.VideoCapture(video)
    try: cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1.0)
    except Exception: pass
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, round(src / fps))
    os.makedirs(d, exist_ok=True)
    paths, idx, k = [], 0, 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        if idx % interval == 0:
            p = os.path.join(d, f"{k:06d}.jpg"); cv2.imwrite(p, fr); paths.append(p); k += 1
        idx += 1
    cap.release(); return paths


def radius_keep(xyz, r=0.05, min_nbr=4):
    from scipy.spatial import cKDTree
    if xyz.shape[0] < min_nbr + 1: return np.ones(xyz.shape[0], bool)
    return cKDTree(xyz).query_ball_point(xyz, r=r, return_length=True) >= (min_nbr + 1)


def reconstruct(model, video, fps, keep, device, rng):
    import glob
    if os.path.isdir(video):
        paths = sorted(sum([glob.glob(os.path.join(video, f"*{e}"))
                            for e in (".png", ".jpg", ".jpeg", ".PNG", ".JPG")], []))
        images = load_and_preprocess_images(paths, mode="crop", image_size=518, patch_size=14)
    else:
        with tempfile.TemporaryDirectory(prefix="dm_") as fd:
            paths = extract(video, fd, fps)
            images = load_and_preprocess_images(paths, mode="crop", image_size=518, patch_size=14)
    S, _, H, W = images.shape
    kf = 1 if S <= 320 else (S + 319) // 320
    t0 = time.time()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        preds = model.inference_streaming(images.to(device), num_scale_frames=NUM_SCALE,
                                          keyframe_interval=kf, output_device=None)
    dt = time.time() - t0
    for k in list(preds.keys()):
        if isinstance(preds[k], torch.Tensor) and preds[k].dim() >= 4 and preds[k].shape[0] == 1:
            preds[k] = preds[k][0]
    w2c, K = pose_encoding_to_extri_intri(preds["pose_enc"], (H, W))
    if w2c.dim() == 4 and w2c.shape[0] == 1: w2c = w2c[0]
    if K.dim() == 4 and K.shape[0] == 1: K = K[0]
    w2c_np = w2c.cpu().numpy().astype(np.float32); K_np = K.cpu().numpy().astype(np.float32)
    e44 = np.tile(np.eye(4, dtype=np.float32)[None], (w2c_np.shape[0], 1, 1)); e44[:, :3, :4] = w2c_np
    c2w = closed_form_inverse_se3_general(torch.from_numpy(e44)).numpy()[:, :3, :].astype(np.float32)
    depth = preds["depth"].cpu().numpy().astype(np.float32)
    if depth.ndim == 3: depth = depth[..., None]
    wp = unproject_depth_map_to_point_map(depth, w2c_np, K_np)
    conf = preds["depth_conf"].cpu().numpy()
    imgs = (images.permute(0, 2, 3, 1).numpy() * 255).clip(0, 255).astype(np.uint8)
    xs, rs = [], []
    for i in range(S):
        m = conf[i].reshape(-1) > CONF
        if m.sum() == 0: continue
        xs.append(wp[i].reshape(-1, 3)[m]); rs.append(imgs[i].reshape(-1, 3)[m])
    xyz = np.concatenate(xs).astype(np.float32); rgb = np.concatenate(rs).astype(np.uint8)
    raw = xyz.shape[0]
    # keep it dense + demo-like: light pre-cap, drop floaters, final cap
    if xyz.shape[0] > keep * 2:
        idx = rng.choice(xyz.shape[0], keep * 2, replace=False); xyz, rgb = xyz[idx], rgb[idx]
    k = radius_keep(xyz, r=0.05, min_nbr=4); xyz, rgb = xyz[k], rgb[k]
    if xyz.shape[0] > keep:
        idx = rng.choice(xyz.shape[0], keep, replace=False); xyz, rgb = xyz[idx], rgb[idx]
    poses = [[float(v) for v in c2w[i].reshape(-1)] for i in range(c2w.shape[0])]
    print(f"  {Path(video).name}: frames={S} kf={kf} raw={raw:,} -> kept={xyz.shape[0]:,} in {dt:.1f}s")
    return xyz, rgb, poses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ckpts/lingbot-map-long.pt")
    ap.add_argument("--videos", nargs="+", required=True, help="path[:label] ...")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--keep", type=int, default=500000)
    ap.add_argument("--out", default="reports/pointcloud_map/demo_map.html")
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda"; rng = np.random.default_rng(0)

    print("loading model (streaming, demo pipeline) ...")
    t0 = time.time()
    model = GCTStream(img_size=518, patch_size=14, enable_3d_rope=True, max_frame_num=1024,
                      kv_cache_sliding_window=64, kv_cache_scale_frames=NUM_SCALE,
                      kv_cache_cross_frame_special=True, kv_cache_include_scale_frames=True,
                      use_sdpa=True, camera_num_iterations=4, enable_point=False)
    ck = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(ck.get("model", ck), strict=False)
    model.aggregator.to(dtype=torch.bfloat16); model.to(device).eval()
    print(f"  ready in {time.time()-t0:.1f}s")

    datasets = []
    for i, spec in enumerate(args.videos):
        vp, _, label = spec.partition(":"); label = label or Path(vp).stem
        print(f"reconstruct {vp} ...")
        xyz, rgb, poses = reconstruct(model, vp, args.fps, args.keep, device, rng)
        datasets.append({"label": label, "up": Path(vp).stem, "n": int(xyz.shape[0]),
                         "color": PALETTE[i % len(PALETTE)],
                         "xyz": base64.b64encode(xyz.tobytes()).decode("ascii"),
                         "rgb": base64.b64encode(rgb.tobytes()).decode("ascii"),
                         "poses": poses})
    html = TEMPLATE.replace("__DATASETS__", json.dumps(datasets))
    out.write_text(html, encoding="utf-8")
    print(f"\nwrote {out}  ({out.stat().st_size/1e6:.1f} MB, {len(datasets)} datasets)")


if __name__ == "__main__":
    main()

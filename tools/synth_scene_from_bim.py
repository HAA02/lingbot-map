"""Generate a synthetic scene_export.json + identity correspondences from a BIM.

Used to dry-run the lingbot->BIM pipeline before real inference is available
(no GPU / no lingbot checkpoint). Cameras are placed inside the BIM bounding
box and sweep through it; correspondences are taken from BIM object centers
so the resulting Sim(3) is identity (alignment validates against itself).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def look_at_c2w(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0, 0, 1.0])) -> np.ndarray:
    """Right-handed look-at producing a camera-to-world matrix.

    Camera convention: +Z forward, +X right, +Y down (OpenCV-style).
    """
    z = target - eye
    z = z / max(np.linalg.norm(z), 1e-8)
    x = np.cross(z, up)
    nx = np.linalg.norm(x)
    if nx < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    else:
        x = x / nx
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])  # world axes expressed in camera basis -> c2w rotation
    c2w = np.zeros((3, 4))
    c2w[:3, :3] = R
    c2w[:3, 3] = eye
    return c2w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bim", required=True, help="lingbot BIM metadata JSON")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-frames", type=int, default=24)
    ap.add_argument("--image-size", type=int, default=518)
    ap.add_argument("--focal", type=float, default=400.0)
    ap.add_argument("--zone", type=str, default=None,
                    help="restrict camera placement near objects in this zone/level (e.g. '레벨 7')")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.bim) as f:
        bim = json.load(f)

    objs = bim["objects"]
    if args.zone:
        zoned = [o for o in objs if o.get("zone") == args.zone]
        if zoned:
            objs_for_extent = zoned
            print(f"[scene] restricting to zone={args.zone} ({len(zoned)} objects)")
        else:
            print(f"[scene] zone {args.zone!r} not found, using all objects")
            objs_for_extent = objs
    else:
        objs_for_extent = objs

    # extent from centers
    centers = []
    for o in objs_for_extent:
        c = o.get("center")
        if c is None and o.get("start") and o.get("end"):
            c = [0.5 * (a + b) for a, b in zip(o["start"], o["end"])]
        if c is not None:
            centers.append(c)
    centers = np.array(centers, dtype=np.float64)
    mn, mx = centers.min(axis=0), centers.max(axis=0)
    ctr = 0.5 * (mn + mx)
    span = (mx - mn)
    print(f"[scene] extent min={mn.round(1)} max={mx.round(1)} center={ctr.round(1)} span={span.round(1)}")

    # circular trajectory in XY around the centroid, slight z bob
    radius_xy = 0.35 * max(span[0], span[1])
    z0 = ctr[2]
    N = args.n_frames
    H = W = args.image_size
    K = np.array([[args.focal, 0, W / 2], [0, args.focal, H / 2], [0, 0, 1]], dtype=np.float64)

    frames = []
    rng = np.random.default_rng(0)
    for i in range(N):
        theta = 2 * math.pi * (i / N)
        eye = np.array([
            ctr[0] + radius_xy * math.cos(theta),
            ctr[1] + radius_xy * math.sin(theta),
            z0 + 0.05 * span[2] * math.sin(2 * theta),
        ])
        c2w = look_at_c2w(eye, ctr)
        frames.append({
            "frame_number": i,
            "timestamp": float(i) / 10.0,
            "image_path": None,
            "intrinsic": K.tolist(),
            "extrinsic_c2w": c2w.tolist(),
            "confidence": 0.8,
        })

    scene = {
        "scene_id": f"synthetic-{os.path.basename(args.bim)}",
        "coordinate_system": "lingbot_world",
        "schema_version": "1.0",
        "source": {
            "synthetic": True,
            "image_size": [H, W],
            "focal": args.focal,
            "extent_center": ctr.tolist(),
            "extent_span": span.tolist(),
        },
        "frames": frames,
        "point_cloud": {},
    }
    scene_path = os.path.join(args.output_dir, "scene_export.json")
    with open(scene_path, "w") as f:
        json.dump(scene, f, indent=2)
    print(f"[scene] wrote {scene_path} (N={N} frames)")

    # identity correspondences: pick 6 BIM object centers spread across the extent.
    picks_idx = np.linspace(0, len(centers) - 1, 6, dtype=int)
    picks = centers[picks_idx]
    pairs = [{"lingbot": p.tolist(), "bim": p.tolist(), "label": f"obj_{i}"}
             for i, p in enumerate(picks)]
    corr_path = os.path.join(args.output_dir, "correspondences.json")
    with open(corr_path, "w") as f:
        json.dump({"pairs": pairs}, f, indent=2)
    print(f"[scene] wrote {corr_path} ({len(pairs)} identity pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Standardized scene export for lingbot-map inference results.

Produces a JSON + companion NPZ/PLY package so that downstream pipelines
(e.g. BIM alignment, progress analysis) can consume pose/pointcloud results
without depending on lingbot-map internals.

Coordinate convention: `extrinsic` is **camera-to-world (c2w)** 3x4 matrices,
matching `demo.postprocess()`. Intrinsic is a standard pinhole 3x3 in pixel
units of the model's working image resolution.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = "1.0"


@dataclass
class LingbotSceneExport:
    scene_id: str
    coordinate_system: str = "lingbot_world"
    schema_version: str = SCHEMA_VERSION
    source: dict[str, Any] = field(default_factory=dict)
    frames: list[dict[str, Any]] = field(default_factory=list)
    point_cloud: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_numpy(arr) -> np.ndarray:
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr)


def _sample_pointcloud(world_points: np.ndarray,
                       colors: np.ndarray | None,
                       confidence: np.ndarray | None,
                       max_points: int,
                       conf_threshold: float) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Flatten per-frame world_points (N,H,W,3) and downsample.

    Filters points with confidence < `conf_threshold` if confidence is given.
    """
    pts = world_points.reshape(-1, 3)
    cols = colors.reshape(-1, 3) if colors is not None else None
    conf = confidence.reshape(-1) if confidence is not None else None

    if conf is not None:
        mask = conf >= conf_threshold
        pts = pts[mask]
        if cols is not None:
            cols = cols[mask]
        conf = conf[mask]

    if pts.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
        if cols is not None:
            cols = cols[idx]
        if conf is not None:
            conf = conf[idx]

    return pts, cols, conf


def _write_ply(path: str, points: np.ndarray, colors: np.ndarray | None) -> None:
    n = points.shape[0]
    has_color = colors is not None
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_color:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header.append("end_header")
    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        if has_color:
            cols_u8 = np.clip(colors, 0, 255).astype(np.uint8) if colors.dtype != np.uint8 else colors
            for p, c in zip(points, cols_u8):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        else:
            for p in points:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def export_scene(
    predictions: dict[str, Any],
    output_dir: str,
    scene_id: str,
    *,
    image_paths: Iterable[str] | None = None,
    timestamps: Iterable[float] | None = None,
    source_info: dict[str, Any] | None = None,
    pointcloud_max_points: int = 200_000,
    pointcloud_conf_threshold: float = 0.1,
) -> LingbotSceneExport:
    """Serialize an inference result dict to `output_dir`.

    Expected keys in `predictions` (per `demo.postprocess`):
      - `extrinsic`: (N, 3, 4) camera-to-world
      - `intrinsic`: (N, 3, 3)
      - `world_points`: (N, H, W, 3)              [optional, but recommended]
      - `world_points_conf`: (N, H, W)            [optional]
      - `images`: (N, 3, H, W) or (N, H, W, 3)    [optional, for color]
    """
    os.makedirs(output_dir, exist_ok=True)

    extr = _to_numpy(predictions["extrinsic"])  # (N,3,4)
    intr = _to_numpy(predictions["intrinsic"])  # (N,3,3)
    n = extr.shape[0]

    image_paths = list(image_paths) if image_paths is not None else [None] * n
    timestamps = list(timestamps) if timestamps is not None else [float(i) for i in range(n)]

    frame_conf = None
    if "world_points_conf" in predictions:
        wpc = _to_numpy(predictions["world_points_conf"])
        # per-frame mean confidence as a coarse summary
        frame_conf = wpc.reshape(n, -1).mean(axis=1)

    frames: list[dict[str, Any]] = []
    for i in range(n):
        frames.append({
            "frame_number": int(i),
            "timestamp": float(timestamps[i]) if i < len(timestamps) else float(i),
            "image_path": image_paths[i] if i < len(image_paths) else None,
            "intrinsic": intr[i].tolist(),
            "extrinsic_c2w": extr[i].tolist(),
            "confidence": float(frame_conf[i]) if frame_conf is not None else None,
        })

    pc_info: dict[str, Any] = {}
    if "world_points" in predictions:
        wp = _to_numpy(predictions["world_points"])  # (N,H,W,3)
        wpc = _to_numpy(predictions["world_points_conf"]) if "world_points_conf" in predictions else None
        colors = None
        if "images" in predictions:
            imgs = _to_numpy(predictions["images"])
            # accept (N,3,H,W) or (N,H,W,3); normalize to (N,H,W,3) uint8-ish range
            if imgs.ndim == 4 and imgs.shape[1] == 3:
                imgs = np.transpose(imgs, (0, 2, 3, 1))
            if imgs.dtype != np.uint8:
                lo, hi = float(imgs.min()), float(imgs.max())
                if hi <= 1.0 + 1e-3:
                    imgs = np.clip(imgs * 255.0, 0, 255).astype(np.uint8)
                else:
                    imgs = np.clip(imgs, 0, 255).astype(np.uint8)
            colors = imgs

        pts, cols, conf = _sample_pointcloud(
            wp, colors, wpc, pointcloud_max_points, pointcloud_conf_threshold
        )

        ply_path = os.path.join(output_dir, "point_cloud_sampled.ply")
        _write_ply(ply_path, pts, cols)
        conf_path = os.path.join(output_dir, "point_confidence.npy")
        if conf is not None:
            np.save(conf_path, conf)

        pc_info = {
            "path": os.path.basename(ply_path),
            "confidence_path": os.path.basename(conf_path) if conf is not None else None,
            "num_points": int(pts.shape[0]),
            "conf_threshold": pointcloud_conf_threshold,
        }

    export = LingbotSceneExport(
        scene_id=scene_id,
        source=source_info or {},
        frames=frames,
        point_cloud=pc_info,
    )

    json_path = os.path.join(output_dir, "scene_export.json")
    with open(json_path, "w") as f:
        json.dump(export.to_dict(), f, indent=2)

    return export


def load_scene(path: str) -> dict[str, Any]:
    """Load a scene_export.json file."""
    with open(path) as f:
        return json.load(f)

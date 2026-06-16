"""Auto pipe-path: the camera walks UNDER the main fire-pipe run looking up, so the
trajectory ≈ the dominant FXX pipe run centerline. Auto-detect it (no manual
waypoints) — replaces hand-estimated --gt-path on repetitive-pipe corridors.

run_centerline: PCA dominant direction of the pipe XZ cloud → densest perpendicular
lane (the main run among parallel runs) → centerline endpoints along that lane.
"""
from __future__ import annotations

import numpy as np

from scan2bim.dtdx_geometry import decode_geometry


def run_centerline(xz, *, lane_width: float = 1.0, pct=(2, 98)) -> np.ndarray:
    """Dominant straight run through a 2D point cloud → [[x0,z0],[x1,z1]] endpoints."""
    xz = np.asarray(xz, float)
    c = xz.mean(0)
    X = xz - c
    _, V = np.linalg.eigh(X.T @ X)
    dirv, perp = V[:, -1], V[:, 0]
    off = X @ perp
    edges = np.histogram_bin_edges(off, bins=40)
    hist, _ = np.histogram(off, bins=edges)
    k = int(np.argmax(hist))
    peak = float((edges[k] + edges[k + 1]) / 2)         # densest perpendicular lane = main run
    lane = np.abs(off - peak) < lane_width
    along = X[lane] @ dirv
    a0, a1 = np.percentile(along, pct)
    return np.array([c + dirv * a0 + perp * peak, c + dirv * a1 + perp * peak])


def main_pipe_run(dtdx_path, *, flip_x: bool = True, lane_width: float = 1.0) -> np.ndarray:
    """Centerline of the dominant pipe run in a discipline file (model XZ, display frame)."""
    d = decode_geometry(dtdx_path)
    P = np.concatenate([m["positions"] for m in d["meshes"] if len(m["positions"])]).astype(float)
    if flip_x:
        P[:, 0] *= -1.0
    return run_centerline(P[:, [0, 2]], lane_width=lane_width)

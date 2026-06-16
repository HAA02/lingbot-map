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


def _fxx_xz(dtdx_path, flip_x):
    d = decode_geometry(dtdx_path)
    P = np.concatenate([m["positions"] for m in d["meshes"] if len(m["positions"])]).astype(float)
    if flip_x:
        P[:, 0] *= -1.0
    return P[:, [0, 2]]


def main_pipe_run(dtdx_path, *, flip_x: bool = True, lane_width: float = 1.0) -> np.ndarray:
    """Centerline of the dominant pipe run in a discipline file (model XZ, display frame)."""
    return run_centerline(_fxx_xz(dtdx_path, flip_x), lane_width=lane_width)


def run_L_polyline(xz, *, lane_width: float = 1.0, branch_min_span: float = 4.0) -> np.ndarray:
    """Main run + its strongest perpendicular branch → L-polyline [start, corner,
    branch_end] (model XZ). Resolves walk direction (start = run end FAR from the
    branch — you turn at the branch, near the walk's end) and the turn corner.
    Falls back to the straight run if no clear branch."""
    xz = np.asarray(xz, float)
    A, B = run_centerline(xz, lane_width=lane_width)
    dirv = B - A; L = float(np.linalg.norm(dirv)); dirv = dirv / (L + 1e-9)
    perp = np.array([-dirv[1], dirv[0]])
    rel = xz - A
    s = rel @ dirv            # along run
    o = rel @ perp            # perpendicular offset
    inrun = (s > 0) & (s < L)
    # branch = band along the run with the widest perpendicular spread (off the main lane)
    best = None
    for c in np.arange(2.0, L - 2.0, 1.0):
        band = inrun & (np.abs(s - c) < 1.0) & (np.abs(o) > lane_width)
        if band.sum() < 200:
            continue
        span = float(np.percentile(np.abs(o[band]), 90))
        if span >= branch_min_span and (best is None or span > best[1]):
            best = (c, span, float(np.median(o[band])))
    if best is None:
        return np.array([A, B])                                   # no branch → straight run
    corner_s, span, side = best
    corner = A + dirv * corner_s
    branch_end = corner + perp * (span if side > 0 else -span)
    start = A if abs(corner_s - 0) > abs(corner_s - L) else B     # far end from the corner
    return np.array([start, corner, branch_end])


def main_pipe_run_L(dtdx_path, *, flip_x: bool = True, lane_width: float = 1.0) -> np.ndarray:
    """L-polyline [start, corner, branch_end] for the dominant FXX run + its branch."""
    return run_L_polyline(_fxx_xz(dtdx_path, flip_x), lane_width=lane_width)

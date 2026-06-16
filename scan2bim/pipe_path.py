"""Auto pipe-path: the camera walks UNDER the main fire-pipe run looking up, so the
trajectory ≈ the dominant FXX pipe run centerline. Auto-detect it (no manual
waypoints) — replaces hand-estimated --gt-path on repetitive-pipe corridors.

run_centerline: PCA dominant direction of the pipe XZ cloud → densest perpendicular
lane (the main run among parallel runs) → centerline endpoints along that lane.
"""
from __future__ import annotations

import numpy as np

from scan2bim.dtdx_geometry import decode_geometry


def _resample(pts, n):
    pts = np.asarray(pts, float)
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    if d[-1] < 1e-9:
        return np.repeat(pts[:1], n, axis=0)
    u = np.linspace(0, d[-1], n)
    return np.column_stack([np.interp(u, d, pts[:, k]) for k in range(pts.shape[1])])


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


def _line_resid(pts):
    if len(pts) < 2:
        return 0.0
    X = pts - pts.mean(0)
    return float(np.linalg.eigvalsh(X.T @ X)[0])      # smallest eigenvalue = perpendicular scatter


def _seg_dir(pts):
    X = pts - pts.mean(0)
    return np.linalg.eigh(X.T @ X)[1][:, -1]


def trajectory_turn_fraction(xz, *, n_samp: int = 60, min_frac: float = 0.15):
    """Main corner of a trajectory via the best two-straight-segment split → (fraction
    0..1, angle°). Robust to end-hooks/jitter (a small artifact adds little fit
    residual, so the MAIN corridor turn wins, not the sharpest instant)."""
    xz = _resample(np.asarray(xz, float), n_samp)
    n = len(xz)
    arclen = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xz, axis=0), axis=1))])
    lo, hi = int(n * min_frac), int(n * (1 - min_frac))
    best_i, best_r = n // 2, None
    for i in range(max(3, lo), min(n - 3, hi)):
        r = _line_resid(xz[:i + 1]) + _line_resid(xz[i:])
        if best_r is None or r < best_r:
            best_r, best_i = r, i
    d1, d2 = _seg_dir(xz[:best_i + 1]), _seg_dir(xz[best_i:])
    ang = float(np.degrees(np.arccos(np.clip(abs(d1 @ d2), -1, 1))))
    return float(arclen[best_i] / (arclen[-1] + 1e-9)), 180 - ang if ang > 90 else ang


def run_L_polyline(xz, *, lane_width: float = 1.0, branch_min_span: float = 4.0,
                   turn_fraction=None) -> np.ndarray:
    """Main run + a perpendicular branch → L-polyline [start, corner, branch_end].
    start = run end FAR from the branch cluster (you turn near the walk's end).
    turn_fraction (from the recon trajectory): pick the branch whose corner sits at
    that fraction along the L — disambiguates among parallel branches. Without it,
    pick the strongest branch. Falls back to the straight run if no branch."""
    xz = np.asarray(xz, float)
    A, B = run_centerline(xz, lane_width=lane_width)
    dirv = B - A; L = float(np.linalg.norm(dirv)); dirv = dirv / (L + 1e-9)
    perp = np.array([-dirv[1], dirv[0]])
    rel = xz - A
    s = rel @ dirv
    o = rel @ perp
    inrun = (s > 0) & (s < L)
    cands = []
    for c in np.arange(2.0, L - 2.0, 1.0):
        band = inrun & (np.abs(s - c) < 1.0) & (np.abs(o) > lane_width)
        if band.sum() < 200:
            continue
        span = float(np.percentile(np.abs(o[band]), 90))
        if span >= branch_min_span:
            cands.append((c, span, float(np.median(o[band]))))
    if not cands:
        return np.array([A, B])
    start_at_A = np.mean([c for c, _, _ in cands]) > L / 2     # branches near B → start at A
    def cfrac(c, span):
        cs = c if start_at_A else (L - c)
        return cs / (cs + span + 1e-9)
    if turn_fraction is not None:
        c, span, side = min(cands, key=lambda k: abs(cfrac(k[0], k[1]) - turn_fraction))
    else:
        c, span, side = max(cands, key=lambda k: k[1])
    corner = A + dirv * c
    branch_end = corner + perp * (span if side > 0 else -span)
    start = A if start_at_A else B
    return np.array([start, corner, branch_end])


def main_pipe_run_L(dtdx_path, *, flip_x: bool = True, lane_width: float = 1.0,
                    turn_fraction=None) -> np.ndarray:
    """L-polyline [start, corner, branch_end] for the dominant FXX run + its branch."""
    return run_L_polyline(_fxx_xz(dtdx_path, flip_x), lane_width=lane_width, turn_fraction=turn_fraction)

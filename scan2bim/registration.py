"""FR-2.1 Sim3 registration (scale + rotation + translation) without
correspondences — places the (metric) scan onto the design model frame.

Open3D has no Python 3.14 wheel, so this uses PCA coarse alignment (with a
proper-rotation flip search to resolve axis sign ambiguity) followed by an
Umeyama-based ICP refine, reusing the project's verified Sim3 solver.

Intended target = the design's structural shell (non-repetitive); the scan's
structural surfaces register against it. Robust to partial overlap via the
inlier-gated ICP.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from lingbot_map.bim.alignment import solve_sim3_umeyama

# proper-rotation sign flips of the 3 principal axes (det = +1)
_FLIPS = [(1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1)]


def _apply(s, R, t, pts):
    return s * (pts @ R.T) + t


def _pca(pts):
    c = pts.mean(0)
    X = pts - c
    _u, _s, Vt = np.linalg.svd(X, full_matrices=False)
    return c, Vt


def _rms_radius(pts, c):
    return float(np.sqrt(((pts - c) ** 2).sum(1).mean()))


def _coarse(src, dst, flip):
    cs, Vs = _pca(src)
    cd, Vd = _pca(dst)
    rs, rd = _rms_radius(src, cs), _rms_radius(dst, cd)
    s = (rd / rs) if rs > 1e-9 else 1.0
    R = Vd.T @ np.diag(flip).astype(float) @ Vs
    if np.linalg.det(R) < 0:  # keep proper rotation
        F = np.diag(flip).astype(float)
        F[2, 2] *= -1
        R = Vd.T @ F @ Vs
    t = cd - s * (R @ cs)
    return s, R, t


def _icp(src, dst, tree, s, R, t, iters=40):
    for _ in range(iters):
        d, idx = tree.query(_apply(s, R, t, src), workers=-1)
        thr = max(float(np.median(d)) * 3.0, 1e-6)
        m = d < thr
        if int(m.sum()) < 3:
            break
        res = solve_sim3_umeyama(src[m], dst[idx[m]])
        ns = float(res.transform.scale)
        nR = np.asarray(res.transform.rotation, dtype=np.float64)
        nt = np.asarray(res.transform.translation, dtype=np.float64)
        converged = abs(ns - s) < 1e-9 and np.allclose(nR, R, atol=1e-9)
        s, R, t = ns, nR, nt
        if converged:
            break
    d, _ = tree.query(_apply(s, R, t, src), workers=-1)
    rmse = float(np.sqrt((d ** 2).mean()))
    inl = float((d < max(float(np.median(d)) * 3.0, 0.05)).mean())
    return s, R, t, rmse, inl


def _subsample(a, n):
    a = np.asarray(a, dtype=np.float64)
    if len(a) > n:
        a = a[np.linspace(0, len(a) - 1, n).astype(int)]
    return a


def register_sim3(source, target, *, max_points: int = 4000) -> dict:
    """Register source point cloud onto target. Returns Sim3 + fit metrics."""
    src = _subsample(source, max_points)
    dst = _subsample(target, max_points)
    tree = cKDTree(dst)
    best = None
    for flip in _FLIPS:
        s0, R0, t0 = _coarse(src, dst, flip)
        s, R, t, rmse, inl = _icp(src, dst, tree, s0, R0, t0)
        if best is None or rmse < best[3]:
            best = (s, R, t, rmse, inl)
    s, R, t, rmse, inl = best
    return {
        "scale": float(s),
        "rotation": R.tolist(),
        "translation": t.tolist(),
        "rmse": rmse,
        "inlier_ratio": inl,
    }

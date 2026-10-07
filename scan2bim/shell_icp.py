"""Rigid SE(3) trimmed point-to-point ICP — scan-shell refiner, numpy/scipy only.

Candidate refiner for a scan shell against a BIM shell AFTER another stage has
already placed the cloud. Does not decide that a match is accepted.

Column-vector convention: x' = R @ x + t. init is 4x4 [R|t; 0 0 0 1] (X' = init @ X).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

_MIN_INLIERS = 6
_EPS_T = 1e-6
_EPS_R = 1e-6


def _empty_result() -> dict:
    return {
        "rotation": np.eye(3).tolist(),
        "translation": [0.0, 0.0, 0.0],
        "rmse": float("inf"),
        "inlier_ratio": 0.0,
        "accepted": False,
        "backend": "numpy",
    }


def _as_nx3(pts) -> np.ndarray:
    a = np.asarray(pts, dtype=np.float64)
    if a.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    a = np.atleast_2d(a)
    if a.shape[1] != 3:
        a = a.reshape(-1, 3)
    return np.ascontiguousarray(a, dtype=np.float64)


def _subsample(a: np.ndarray, n: int) -> np.ndarray:
    if len(a) > n:
        a = a[np.linspace(0, len(a) - 1, n).astype(int)]
    return a


def _parse_init(init) -> tuple[np.ndarray, np.ndarray]:
    """Column-vector SE(3): x' = R @ x + t, with t = init[:3, 3]."""
    if init is None:
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    T = np.asarray(init, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError("init must be a 4x4 array or None")
    return T[:3, :3].copy(), T[:3, 3].copy()


def _apply(R: np.ndarray, t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ R.T + t


def _rot_angle(R: np.ndarray) -> float:
    c = (np.trace(R) - 1.0) * 0.5
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def _kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid R, t with x' = R @ x + t (no scale). Proper rotation only."""
    cs = src.mean(axis=0)
    ct = dst.mean(axis=0)
    H = (src - cs).T @ (dst - ct)
    U, _s, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0.0:
        Vt = Vt.copy()
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = ct - R @ cs
    return R, t


def refine_rigid_icp(
    source,
    target,
    init=None,
    *,
    max_points: int = 4000,
    max_corr_dist: float = 0.5,
    iters: int = 30,
) -> dict:
    """Trimmed point-to-point ICP. x' = R @ x + t (column vectors). Never accepts."""
    src = _subsample(_as_nx3(source), max_points)
    dst = _subsample(_as_nx3(target), max_points)
    if len(src) < _MIN_INLIERS or len(dst) < _MIN_INLIERS:
        return _empty_result()

    R, t = _parse_init(init)
    tree = cKDTree(dst)

    for _ in range(int(iters)):
        d, idx = tree.query(_apply(R, t, src), workers=-1)
        mask = d < max_corr_dist
        if int(mask.sum()) < _MIN_INLIERS:
            break
        nR, nt = _kabsch(src[mask], dst[idx[mask]])
        dR = nR @ R.T
        if np.linalg.norm(nt - t) < _EPS_T and _rot_angle(dR) < _EPS_R:
            R, t = nR, nt
            break
        R, t = nR, nt

    d, _idx = tree.query(_apply(R, t, src), workers=-1)
    mask = d < max_corr_dist
    n_inl = int(mask.sum())
    if n_inl == 0:
        rmse = float("inf")
    else:
        rmse = float(np.sqrt(np.mean(d[mask] ** 2)))
    return {
        "rotation": np.asarray(R, dtype=np.float64).tolist(),
        "translation": np.asarray(t, dtype=np.float64).reshape(3).tolist(),
        "rmse": rmse,
        "inlier_ratio": float(n_inl / len(src)),
        "accepted": False,
        "backend": "numpy",
    }

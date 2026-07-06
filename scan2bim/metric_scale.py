"""Multi-anchor metric scale for monocular recon->model placement.

A SINGLE anchor (e.g. ceiling-height vertical extent) is fragile: if the scan
doesn't capture the full floor-to-ceiling span, it silently over/under-shoots
(observed both ways in practice). Combine independent anchors and flag
disagreement instead of trusting one blindly.
"""
from __future__ import annotations

import numpy as np


def estimate_floor_level(y: np.ndarray, cam_y: float | None = None, pct: float = 3.0) -> float | None:
    """Low-percentile of the vertical extent BELOW the camera = floor (gravity-
    aligned frame, recon/unmetric units) — mirrors the existing ceiling estimate
    (97th percentile) for consistency. A density-peak estimator was tried first
    but on real "looking up at pipes" footage it locks onto whatever non-floor
    surface (pipe/wall texture) happens to be denser than the sparsely-captured
    floor; percentile is robust to that since it only needs the low tail, not a
    density mode. Restricting to below cam_y avoids the ceiling band entirely."""
    y = np.asarray(y, dtype=np.float64)
    if cam_y is not None:
        y = y[y < cam_y]
    if y.size == 0:
        return None
    return float(np.percentile(y, pct))


def camera_height_scale(cam_y: np.ndarray, floor_y: float, assumed_height: float = 1.5) -> float | None:
    """Scale from assumed real camera-carry height (eye level) vs the recon's own
    camera-to-floor distance. Independent of ceiling capture completeness."""
    h = float(np.median(np.asarray(cam_y, dtype=np.float64)) - floor_y)
    if h <= 1e-6:
        return None
    return assumed_height / h


def fuse_scale_estimates(estimates: list[float | None], *, disagreement_ratio: float = 1.15) -> tuple[float, dict]:
    """Combine scale estimates from independent anchors. The FIRST estimate is
    the primary/validated anchor: when the others agree, average them in
    (cross-validation, mild smoothing); when they disagree, keep the primary
    unchanged rather than silently dragging it toward an unconfirmed anchor —
    a lone secondary anchor can itself be wrong (e.g. floor rarely visible in
    upward-looking footage), so disagreement should surface as a warning for
    review, not quietly change the result. disagreement_ratio: max/min above
    which anchors are flagged as disagreeing."""
    vals = [float(v) for v in estimates if v is not None and np.isfinite(v) and v > 0]
    if not vals:
        raise ValueError("no valid scale estimates")
    spread = max(vals) / min(vals) if len(vals) > 1 else 1.0
    agree = spread <= disagreement_ratio
    fused = float(np.median(vals)) if agree else vals[0]
    return fused, {"n": len(vals), "values": [round(v, 4) for v in vals],
                   "spread": round(spread, 3), "agree": agree}


def bbox_height_warning(height_m: float, lo: float = 2.0) -> str | None:
    """Flag a model bbox height too thin to be a real floor-to-ceiling span —
    the telltale sign of passing a single thin-discipline dtdx (e.g. just the
    fire-pipe run) instead of the full multi-discipline model, which silently
    turns the ceiling-height scale anchor into nonsense (pipe-slab thickness,
    not room height) and produces paths far shorter than reality."""
    if height_m < lo:
        return (f"model bbox height {height_m:.2f}m looks too thin for a floor-to-ceiling "
                f"span — pass the full multi-discipline --dtdx (architecture included), not "
                f"a single thin MEP-run file, or the ceiling-height scale anchor is meaningless")
    return None


def speed_warning(path_m: float, duration_s: float | None, lo: float = 0.15, hi: float = 2.0) -> str | None:
    """Flag an implausible average walking speed (path length / video duration) —
    a cheap cross-check that catches gross scale errors in either direction."""
    if not duration_s or duration_s <= 0:
        return None
    speed = path_m / duration_s
    if speed < lo or speed > hi:
        return f"implausible walking speed {speed:.2f} m/s (path={path_m:.1f}m / {duration_s:.1f}s)"
    return None

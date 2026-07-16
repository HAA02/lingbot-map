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
    the primary/validated anchor: when they all agree, average them in
    (cross-validation, mild smoothing).

    On disagreement the tie-break depends on how many anchors we have:
    - With only two, there is no majority — keep the primary unchanged rather than
      let a lone unconfirmed secondary drag it (e.g. floor rarely visible looking up).
    - With three or more, trust a MAJORITY consensus over the primary: if a subset
      that mutually agrees forms a majority, use its median. This is what the
      independent corridor-width anchor buys — when the two robust anchors (wall +
      camera) agree at ~3x but the fragile ceiling primary undershoots at ~1x, the
      consensus should win instead of the primary silently keeping the wrong scale.
    - If no such majority exists (all mutually disagree), stay conservative and keep
      the primary rather than pick an arbitrary outlier.

    disagreement_ratio: max/min above which anchors are flagged as disagreeing."""
    vals = [float(v) for v in estimates if v is not None and np.isfinite(v) and v > 0]
    if not vals:
        raise ValueError("no valid scale estimates")
    n = len(vals)
    spread = max(vals) / min(vals) if n > 1 else 1.0
    agree = spread <= disagreement_ratio
    if agree:
        fused = float(np.median(vals))
    elif n <= 2:
        fused = vals[0]
    else:
        s = sorted(vals)
        lo = hi = 0                                  # widest mutually-agreeing window
        for a in range(n):
            b = a
            while b + 1 < n and s[b + 1] / s[a] <= disagreement_ratio:
                b += 1
            if (b - a) > (hi - lo):
                lo, hi = a, b
        window = s[lo:hi + 1]
        fused = float(np.median(window)) if len(window) > n / 2 else vals[0]
    return fused, {"n": n, "values": [round(v, 4) for v in vals],
                   "spread": round(spread, 3), "agree": agree}


def apply_axis_split_scale(poses_yup: dict, pts_yup: np.ndarray, s_h: float, s_v: float):
    """Apply an anisotropic diag(s_h, s_v, s_h) scale in the gravity-aligned Y-up
    frame to the camera poses and the scan cloud.

    Monocular recon compresses the two HORIZONTAL axes noticeably more than the
    vertical one, so a single isotropic scale cannot place the path metrically: the
    horizontal scale (from the corridor-width anchor) and the vertical scale (from
    the ceiling-height anchor) must be applied on their own axes.

    A non-uniform scale does NOT preserve directions, so after scaling the forward
    and up vectors they are re-normalised and re-orthogonalised. `up` is kept as the
    anchor — it stays near-vertical, so the horizon stays level — and `forward` is
    made perpendicular to it. Anchoring on up also preserves forward's horizontal
    heading (its X and Z both scale by s_h, so its XZ azimuth is unchanged), so the
    look direction still follows the horizontally-scaled path. If forward is
    (near-)parallel to up (looking straight up/down) only re-normalisation is done.

    poses_yup: dict {"c": (N,3), "f": (N,3), "u": (N,3)} of centres / forward / up
        unit vectors in the gravity-aligned frame.
    pts_yup: (M,3) scan cloud in the same frame.
    Returns (poses_scaled, pts_scaled); poses_scaled has the same {"c","f","u"} keys
    (same structure as the input) and pts_scaled is the scaled (M,3) cloud."""
    d = np.array([s_h, s_v, s_h], dtype=np.float64)
    c = np.asarray(poses_yup["c"], dtype=np.float64) * d
    f = np.asarray(poses_yup["f"], dtype=np.float64) * d
    u = np.asarray(poses_yup["u"], dtype=np.float64) * d
    u = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-12)
    f_perp = f - np.sum(f * u, axis=1, keepdims=True) * u    # Gram-Schmidt against up
    nrm = np.linalg.norm(f_perp, axis=1, keepdims=True)
    f_out = np.where(nrm < 1e-9, f, f_perp)                  # parallel -> keep scaled f
    f_out = f_out / (np.linalg.norm(f_out, axis=1, keepdims=True) + 1e-12)
    poses_scaled = {"c": c, "f": f_out, "u": u}
    pts_scaled = np.asarray(pts_yup, dtype=np.float64) * d
    return poses_scaled, pts_scaled


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


def speed_warning(path_m: float, duration_s: float | None, lo: float = 0.5, hi: float = 2.0) -> str | None:
    """Flag an implausible average walking speed (path length / video duration) —
    a cheap cross-check that catches gross scale errors in either direction.

    lo assumes near-continuous walking (a survey walk-through, not a static
    inspection): normal gait is ~0.9-1.4 m/s, so an average below ~0.5 m/s over
    the whole clip signals the path was reconstructed too short — exactly the ~3x
    monocular under-scale seen on real data (10.53 m / 35.3 s = 0.30 m/s warns,
    while a metrically correct 32.8 m / 35.3 s = 0.93 m/s does not)."""
    if not duration_s or duration_s <= 0:
        return None
    speed = path_m / duration_s
    if speed < lo or speed > hi:
        return f"implausible walking speed {speed:.2f} m/s (path={path_m:.1f}m / {duration_s:.1f}s)"
    return None

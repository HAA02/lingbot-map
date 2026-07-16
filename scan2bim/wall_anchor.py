"""Corridor-width metric scale anchor.

The ceiling-height and camera-height anchors in ``metric_scale`` are both fragile
on "looking up at the pipes" footage: the ceiling anchor silently over/under-shoots
when the scan misses the full floor->ceiling span, and the camera anchor leans on
an *assumed* eye-carry height. The horizontal gap between two facing vertical walls
(a corridor) depends on neither: it is a purely horizontal measurement, unaffected
by how much of the floor/ceiling was captured or by the camera-height guess. That
makes it an independent third anchor for cross-validating monocular scale.

Detection (numpy + scipy only, no RANSAC dependency):
  1. Gravity-aligned Y-up cloud -> drop the floor and ceiling slabs by keeping a
     vertical mid-band (walls span the whole height; slabs sit at the Y extremes).
  2. Project the surviving wall points onto the horizontal *normal* direction
     (perpendicular to the corridor's long axis; the axis comes from the camera
     trajectory when available, else from the point PCA).
  3. A vertical wall becomes a sharp density spike on that 1D axis. find_peaks
     recovers the walls; the spacing between a facing pair is the corridor width
     in recon units. width_model / width_recon = metric scale.

Returns ``(None, info)`` when no confident facing pair is found (open space, only
floor/ceiling, too few points) so the caller can fall back to the other anchors.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

# fraction of the total vertical extent trimmed off the top and bottom to shed the
# floor and ceiling slabs before looking for walls.
_BAND_MARGIN = 0.15
_MIN_POINTS = 50
_NBINS = 256
_PROM_FRAC = 0.15    # a wall spike must clear this fraction of the tallest spike
_BLOB_RATIO = 3.0    # walls must tower over the typical bin; a diffuse (open-space)
                     # projection has max ~ median and is rejected as "no walls"


def _horizontal_axes(xz: np.ndarray, cam_xz: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Return (corridor_axis, wall_normal) unit vectors in the XZ plane. The axis
    is the corridor's long direction (camera walk direction if given, else the
    dominant point spread); the normal is what we project onto to separate walls."""
    axis = None
    if cam_xz is not None and len(cam_xz) >= 2:
        c = cam_xz - cam_xz.mean(0)
        if float((c ** 2).sum()) > 1e-9:
            axis = np.linalg.eigh(c.T @ c)[1][:, -1]
    if axis is None:
        c = xz - xz.mean(0)
        axis = np.linalg.eigh(c.T @ c)[1][:, -1]
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    normal = np.array([-axis[1], axis[0]])
    return axis, normal


def _wall_peaks(t: np.ndarray):
    """Density spikes of the points projected onto the wall-normal axis.
    Returns (positions, prominences) sorted ascending, or None if degenerate.

    The histogram range comes from an IQR fence (drop only far outliers), not a
    percentile trim nor a MAD/median window: facing walls each carry a big share
    of the mass, so a percentile trim would slice into the outermost walls and a
    median/MAD window collapses on bimodal data (the median falls inside one wall).
    The IQR spans the bulk regardless of modality, so every wall survives; the pad
    leaves the extreme walls an empty margin so their density spikes stay prominent."""
    q1, q3 = np.percentile(t, [25.0, 75.0])
    iqr = float(q3 - q1)
    if iqr < 1e-9:
        iqr = float(t.std())
    if iqr < 1e-9:
        return None
    keep = (t >= q1 - 3.0 * iqr) & (t <= q3 + 3.0 * iqr)
    tk = t[keep]
    if len(tk) < _MIN_POINTS:
        return None
    lo, hi = float(tk.min()), float(tk.max())
    span = hi - lo
    if span < 1e-9:
        return None
    lo -= 0.03 * span
    hi += 0.03 * span
    edges = np.linspace(lo, hi, _NBINS + 1)
    hist, _ = np.histogram(tk, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ker = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    ker /= ker.sum()
    h = np.convolve(hist.astype(float), ker, mode="same")
    if h.max() <= 0 or h.max() < _BLOB_RATIO * (float(np.median(h)) + 1e-9):
        return None
    idx, props = find_peaks(h, prominence=_PROM_FRAC * h.max(),
                            distance=max(3, _NBINS // 40))
    if len(idx) == 0:
        return None
    pos = centers[idx]
    prom = np.asarray(props["prominences"], dtype=float)
    order = np.argsort(pos)
    return pos[order], prom[order]


def _select_pair(pos: np.ndarray, cam_t: float | None):
    """Pick the facing wall pair that STRADDLES the trajectory: the nearest wall on
    each side of the camera path (the corridor actually walked). A pair that does
    not straddle the path is not that corridor, so it is rejected outright (returns
    None) — never used as a lenient fallback. On real data a non-straddling 0.93 m
    gap was mistakenly accepted and produced a wrong 1.06x scale; requiring straddle
    removes that failure mode. Without a trajectory there is nothing to straddle."""
    n = len(pos)
    if n < 2 or cam_t is None:
        return None
    below = np.where(pos < cam_t)[0]
    above = np.where(pos > cam_t)[0]
    if len(below) and len(above):
        return int(below[-1]), int(above[0])
    return None


def _choose_scale(recon_w: float, gaps: list[float], widths: list[float]) -> tuple[float, float]:
    """Map the selected recon gap to a model corridor width. With one model width
    it is a direct ratio. With several, pick the width whose implied single global
    scale most consistently maps *all* detected recon gaps onto model widths."""
    if len(widths) == 1:
        return widths[0] / recon_w, widths[0]
    best = None
    for w in widths:
        s = w / recon_w
        cost = sum(min(abs(g * s - wj) / wj for wj in widths) for g in gaps)
        if best is None or cost < best[0] - 1e-12:
            best = (cost, s, w)
    return best[1], best[2]


def _confidence(prom: np.ndarray, i: int, j: int) -> float:
    q = float(min(prom[i], prom[j]) / (prom.max() + 1e-12))
    return float(np.clip(0.55 + 0.35 * q, 0.0, 1.0))


def estimate_wall_scale(pts_yup: np.ndarray, model_corridor_widths: list[float],
                        cam_xz: np.ndarray | None = None,
                        trajectory_radius: float | None = None) -> tuple[float | None, dict]:
    """Estimate metric scale from the gap between facing vertical walls in a
    gravity-aligned Y-up point cloud, versus a known model corridor width.

    pts_yup: (N, 3) gravity-aligned cloud, columns [X, Y, Z], Y up.
    model_corridor_widths: known real corridor width(s) in metres.
    cam_xz: (M, 2) camera trajectory in the [X, Z] plane. REQUIRED for a result —
        scale is only returned from a wall pair that straddles the path (the
        corridor actually walked); without a trajectory nothing can straddle.
    trajectory_radius: if set, first keep only points within this XZ distance of the
        trajectory (typically one ceiling-height). The full, often ceiling-facing
        scan rarely shows clean wall spikes; restricting to the corridor walked does.
        None (default) applies no prefilter, so prior callers are unaffected.

    Returns (scale, info) or (None, info) when no straddling facing pair is found.
    """
    info: dict = {"n_walls": 0}
    pts = np.asarray(pts_yup, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < _MIN_POINTS:
        info["fail"] = "insufficient points"
        return None, info
    widths = [float(w) for w in model_corridor_widths if w and float(w) > 0]
    if not widths:
        info["fail"] = "no model corridor widths"
        return None, info

    cam = None
    if cam_xz is not None:
        cam = np.asarray(cam_xz, dtype=np.float64)
        if cam.ndim != 2 or cam.shape[1] != 2 or len(cam) < 2:
            cam = None

    if trajectory_radius is not None and cam is not None and float(trajectory_radius) > 0:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(cam).query(pts[:, [0, 2]], k=1, workers=-1)
        pts = pts[d < float(trajectory_radius)]
        info["n_after_radius"] = int(len(pts))
        if len(pts) < _MIN_POINTS:
            info["fail"] = "no points within trajectory radius"
            return None, info

    y = pts[:, 1]
    ylo, yhi = float(y.min()), float(y.max())
    yr = yhi - ylo
    if yr < 1e-6:
        info["fail"] = "degenerate vertical extent"
        return None, info
    band = (y > ylo + _BAND_MARGIN * yr) & (y < yhi - _BAND_MARGIN * yr)
    wall_pts = pts[band]
    if len(wall_pts) < _MIN_POINTS:
        info["fail"] = "no mid-band wall points"
        return None, info
    xz = wall_pts[:, [0, 2]]

    axis, normal = _horizontal_axes(xz, cam)
    t = xz @ normal

    peaks = _wall_peaks(t)
    if peaks is None:
        info["fail"] = "no wall density peaks"
        return None, info
    pos, prom = peaks
    info["n_walls"] = int(len(pos))
    if len(pos) < 2:
        info["fail"] = "fewer than two vertical walls"
        return None, info

    cam_t = float(np.median(cam @ normal)) if cam is not None else None
    sel = _select_pair(pos, cam_t)
    if sel is None:
        info["fail"] = "no straddling pair"
        return None, info
    i, j = sel
    recon_w = float(abs(pos[j] - pos[i]))
    if recon_w <= 1e-6:
        info["fail"] = "degenerate wall gap"
        return None, info

    gaps = list(np.diff(pos)) + [recon_w]
    scale, model_w = _choose_scale(recon_w, gaps, widths)
    conf = _confidence(prom, i, j)

    info.update({
        "recon_width": round(recon_w, 4),
        "model_width": round(float(model_w), 4),
        "scale": round(float(scale), 4),
        "n_pairs": int(len(pos) * (len(pos) - 1) // 2),
        "straddles_trajectory": True,
        "axis": [round(float(axis[0]), 4), round(float(axis[1]), 4)],
        "normal": [round(float(normal[0]), 4), round(float(normal[1]), 4)],
        "wall_positions": [round(float(p), 4) for p in pos],
        "confidence": round(conf, 3),
    })
    return float(scale), info

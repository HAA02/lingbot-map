"""Anisotropic (heading-relative) forward scale for monocular recon->model placement.

metric_scale.apply_axis_split_scale already splits the ISOTROPIC-horizontal scale
(s_h, from the corridor-width wall/DXF anchor) from the vertical one (s_v). But on
"looking up at pipes" footage the monocular compression is not merely horizontal-vs-
vertical: it is HEADING-RELATIVE — the walk direction (forward) is compressed more
than the sideways (lateral) direction. A single s_h on both horizontal axes then
places the corridor WIDTH correctly (that is what the wall anchor measured) yet
leaves the walked LENGTH short (observed on upload_1781521406685: leg-A corner lands
~5 m before the DXF lounge opening, avg speed 0.31 m/s).

This module supplies the missing forward axis:
  * anisotropic_scale_tensor: a 2x2 XZ tensor A = R(theta)*diag(s_f, s_h)*R(theta)^T
    aligned to the leg-A heading theta — s_f along the walk, s_h across it. With
    s_f == s_h it is exactly s_h*I, so it degrades to the current isotropic behaviour.
  * apply_forward_scale: apply that XZ tensor (+ vertical s_v on Y) to gravity-aligned
    Y-up poses/points, re-orthogonalising forward/up with the SAME convention as
    metric_scale.apply_axis_split_scale (anchor up, Gram-Schmidt forward).
  * corridor_open_boundary: read the model point where the corridor's facing wall
    pair ends (opens to the lounge) straight off the DXF plan — the forward landmark
    L_end (mirrors dxf_plan.corridor_widths_near's local perpendicular measure).
  * estimate_forward_scale: s_f = (start->L_end model forward distance) / (leg-A recon
    arclen) — the forward analogue of s_h = model_width / recon_gap.

Nothing here decides placement (yaw/offset) or fabricates observed evidence; it only
supplies a scale tensor and a DXF landmark. tools/build_coplay.py::place_rigid applies
it ONLY behind the explicit --forward-scale flag (isotropic path byte-identical).
"""
from __future__ import annotations

import numpy as np


def anisotropic_scale_tensor(theta_legA: float, s_forward: float, s_lateral: float) -> np.ndarray:
    """2x2 XZ scale tensor aligned to the leg-A heading.

    A = R(theta)*diag(s_forward, s_lateral)*R(theta)^T, R(theta) the 2D rotation by
    theta_legA (RADIANS). Applied to an XZ vector it stretches the component ALONG
    theta by s_forward and the perpendicular component by s_lateral. Symmetric
    positive-definite; eigenvalues {s_forward, s_lateral} with the s_forward
    eigenvector along (cos theta, sin theta). s_forward == s_lateral -> s*I (the
    isotropic case, so the caller degrades exactly to apply_axis_split_scale)."""
    c, s = np.cos(theta_legA), np.sin(theta_legA)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return R @ np.diag([float(s_forward), float(s_lateral)]) @ R.T


def apply_forward_scale(poses_yup: dict, pts_yup: np.ndarray, A_xz: np.ndarray, s_v: float):
    """Apply the anisotropic XZ tensor A_xz (2x2, acting on [x, z]) plus a vertical
    s_v on Y to gravity-aligned Y-up poses and the scan cloud.

    Same contract and re-orthogonalisation as metric_scale.apply_axis_split_scale
    (which this generalises: apply_forward_scale(A=s_h*I, s_v) == apply_axis_split_
    scale(s_h, s_v)). A non-uniform scale does not preserve directions, so after
    scaling forward/up we re-normalise up (kept as the anchor, horizon stays level)
    and Gram-Schmidt forward against it; if forward is (near-)parallel to up only
    re-normalisation is done.

    poses_yup: {"c":(N,3), "f":(N,3), "u":(N,3)}. pts_yup: (M,3). A_xz: (2,2).
    Returns (poses_scaled, pts_scaled) with the same keys/shape as the input."""
    A = np.asarray(A_xz, dtype=np.float64)
    # 3x3 linear map: XZ block = A, Y = s_v (rows/cols ordered x, y, z).
    M = np.array([[A[0, 0], 0.0, A[0, 1]],
                  [0.0, float(s_v), 0.0],
                  [A[1, 0], 0.0, A[1, 1]]], dtype=np.float64)
    c = np.asarray(poses_yup["c"], dtype=np.float64) @ M.T
    f = np.asarray(poses_yup["f"], dtype=np.float64) @ M.T
    u = np.asarray(poses_yup["u"], dtype=np.float64) @ M.T
    u = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-12)
    f_perp = f - np.sum(f * u, axis=1, keepdims=True) * u    # Gram-Schmidt against up
    nrm = np.linalg.norm(f_perp, axis=1, keepdims=True)
    f_out = np.where(nrm < 1e-9, f, f_perp)                  # parallel -> keep scaled f
    f_out = f_out / (np.linalg.norm(f_out, axis=1, keepdims=True) + 1e-12)
    poses_scaled = {"c": c, "f": f_out, "u": u}
    pts_scaled = np.asarray(pts_yup, dtype=np.float64) @ M.T
    return poses_scaled, pts_scaled


def corridor_open_boundary(wall_segments: np.ndarray, axis_origin, axis_dir,
                           s_range=(-6.0, 40.0), step: float = 0.5,
                           half_width: float = 4.0, parallel_deg: float = 15.0,
                           min_len: float = 0.4) -> tuple:
    """Model point L_end where the corridor's facing wall pair ENDS (opens to the
    lounge), read off the DXF plan.

    Walks along the corridor axis (axis_origin + s*axis_dir) and at each s keeps the
    corridor-parallel walls (dir within parallel_deg of the axis) whose along-axis
    span brackets s, taking the nearest wall on each side (the same local
    perpendicular measure as dxf_plan.corridor_widths_near). L_end is the LAST s at
    which such a facing pair exists before it disappears (a wall drops out) — the
    boundary where the corridor opens up.

    wall_segments: (N,2,2) metres, DXF/model-raw plan frame (dxf_plan.load_wall_
    segments; plan_transform ~ identity on this export). axis_origin/axis_dir: (2,)
    model XZ. Returns (L_end_xz (2,), info) with s_end and the per-sample gaps; info
    has "fail" and L_end None if no facing pair is ever found."""
    seg = np.asarray(wall_segments, dtype=np.float64)
    o = np.asarray(axis_origin, dtype=np.float64)
    ax = np.asarray(axis_dir, dtype=np.float64)
    ax = ax / (np.linalg.norm(ax) + 1e-12)
    perp = np.array([-ax[1], ax[0]])
    info: dict = {"n_segments": int(len(seg))}
    if len(seg) == 0:
        info["fail"] = "no wall segments"
        return None, info
    d = seg[:, 1] - seg[:, 0]
    L = np.linalg.norm(d, axis=1)
    dn = d / (L[:, None] + 1e-12)
    keep = (np.abs(dn @ ax) > np.cos(np.deg2rad(parallel_deg))) & (L > min_len)
    ps = seg[keep]
    if len(ps) < 2:
        info["fail"] = "fewer than two corridor-parallel walls"
        return None, info
    a0 = (ps[:, 0] - o) @ ax
    a1 = (ps[:, 1] - o) @ ax
    alo, ahi = np.minimum(a0, a1), np.maximum(a0, a1)
    woff = (ps.mean(axis=1) - o) @ perp
    s_end = None
    gaps = []
    for s in np.arange(s_range[0], s_range[1] + 1e-9, step):
        beside = (alo <= s) & (ahi >= s)
        nb = woff[beside]
        below = nb[(nb < -0.1) & (nb > -half_width)]
        above = nb[(nb > 0.1) & (nb < half_width)]
        if below.size and above.size:
            gap = float(above.min() - below.max())
            gaps.append((round(float(s), 2), round(gap, 3)))
            s_end = float(s)               # advance the boundary while the pair holds
        elif s_end is not None:
            break                          # pair just dropped out -> corridor opened
    if s_end is None:
        info["fail"] = "no facing wall pair along the corridor axis"
        return None, info
    L_end = o + ax * s_end
    info.update({"s_end": round(s_end, 3), "L_end": [round(float(L_end[0]), 3), round(float(L_end[1]), 3)],
                 "n_gap_samples": len(gaps), "last_gaps": gaps[-4:]})
    return L_end, info


def estimate_forward_scale(legA_recon_arclen: float, start_xz, L_end_xz, run_dir,
                           s_h: float | None = None, band=(1.0, 3.0)) -> tuple:
    """Forward scale s_f = (start->L_end model forward distance) / (leg-A recon arclen).

    The forward analogue of the wall anchor's s_h = model_width / recon_gap: the model
    distance the walk should cover along the corridor divided by the recon leg-A's own
    (raw, XZ) arclength. forward distance is the |projection| of (L_end - start) onto
    the corridor direction run_dir, so a small lateral offset of start does not inflate
    it.

    s_h/band are OPTIONAL and only annotate info (never clamp s_f — the caller decides):
    band is expressed as multiples of s_h, i.e. physically s_h*band[0] < s_f <= s_h*band[1].
    Returns (s_f, info). Raises ValueError on a non-positive arclen."""
    arc = float(legA_recon_arclen)
    if arc <= 1e-9:
        raise ValueError("legA_recon_arclen must be positive")
    start = np.asarray(start_xz, dtype=np.float64)
    lend = np.asarray(L_end_xz, dtype=np.float64)
    run = np.asarray(run_dir, dtype=np.float64)
    run = run / (np.linalg.norm(run) + 1e-12)
    forward_dist = abs(float((lend - start) @ run))
    s_f = forward_dist / arc
    info = {"s_f": round(s_f, 4), "forward_dist": round(forward_dist, 4),
            "legA_recon_arclen": round(arc, 4),
            "start_xz": [round(float(start[0]), 3), round(float(start[1]), 3)],
            "L_end_xz": [round(float(lend[0]), 3), round(float(lend[1]), 3)]}
    if s_h is not None and s_h > 0:
        lo, hi = s_h * band[0], s_h * band[1]
        info["s_h"] = round(float(s_h), 4)
        info["band"] = [round(lo, 4), round(hi, 4)]
        info["in_band"] = bool(lo < s_f <= hi)
    return s_f, info

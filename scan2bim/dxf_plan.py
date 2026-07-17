"""Revit-exported DXF floor plan as the AUTHORITATIVE corridor-width source.

The SXX triangle-soup wall estimate (tools.build_coplay.model_corridor_widths) is
noisy and furniture-contaminated, and the recon straddle gate abstains when the
walker hugs one wall. An official DXF plan carries the real wall lines, so the
corridor clear width can be read straight off the drawing — the single number that
resolves the monocular horizontal scale (s_h = model_width / recon_gap).

Frame note (measured on Gasan_7F): the DXF and the dtdx model come from the same
Revit export, so the DXF (x, y) plan maps to the dtdx (x, z) plan by a near-identity
translation (estimate_plan_transform reports the residual; it refuses a forced fit).

ezdxf is imported lazily so importing this module never hard-requires it.
"""
from __future__ import annotations

import numpy as np

# $INSUNITS code -> metres per drawing unit (4 = mm, this export).
_UNIT_TO_M = {1: 0.0254, 2: 0.3048, 4: 0.001, 5: 0.01, 6: 1.0}
_WALL_LAYERS = ("A-WALL", "A-GLAZ")
_DOOR_LAYERS = ("A-DOOR",)


def _ezdxf():
    try:
        import ezdxf
        return ezdxf
    except ImportError as e:                                     # pragma: no cover - env guard
        raise ImportError("scan2bim.dxf_plan needs `ezdxf` (pip install ezdxf) to read DXF plans") from e


def _entity_segments(e):
    """Yield (x0,y0,x1,y1) polyline/line edges in DRAWING units for one entity."""
    t = e.dxftype()
    if t == "LINE":
        yield (e.dxf.start.x, e.dxf.start.y, e.dxf.end.x, e.dxf.end.y)
    elif t == "LWPOLYLINE":
        pts = [(p[0], p[1]) for p in e.get_points()]
        for i in range(len(pts) - 1):
            yield (*pts[i], *pts[i + 1])
        if e.closed and len(pts) > 2:
            yield (*pts[-1], *pts[0])
    elif t == "POLYLINE":
        # ezdxf caveat: vertex location is not slice-indexable -> read .x/.y explicitly
        vs = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
        for i in range(len(vs) - 1):
            yield (*vs[i], *vs[i + 1])
        if e.is_closed and len(vs) > 2:
            yield (*vs[-1], *vs[0])


def load_wall_segments(dxf_path, layers=_WALL_LAYERS) -> np.ndarray:
    """Wall/glazing line segments from a DXF plan, in METRES, shape (N, 2, 2)
    (each row = [[x0,y0],[x1,y1]]). A-WALL is direct polylines; A-GLAZ is block
    INSERTs, exploded via virtual_entities(). $INSUNITS sets the unit scale."""
    ezdxf = _ezdxf()
    doc = ezdxf.readfile(str(dxf_path))
    scale = _UNIT_TO_M.get(int(doc.header.get("$INSUNITS", 4) or 4), 0.001)
    msp = doc.modelspace()
    segs: list = []
    for e in msp:
        if not str(e.dxf.layer).startswith(tuple(layers)):
            continue
        if e.dxftype() == "INSERT":
            for v in e.virtual_entities():
                segs.extend(_entity_segments(v))
        else:
            segs.extend(_entity_segments(e))
    if not segs:
        return np.zeros((0, 2, 2), dtype=np.float64)
    return np.asarray(segs, dtype=np.float64).reshape(-1, 2, 2) * scale


def _best_shift(a: np.ndarray, b: np.ndarray, span: float = 12.0, step: float = 0.05) -> tuple[float, float]:
    """Translation of `a` onto `b` (1D) maximising histogram overlap. Returns
    (shift, overlap_fraction)."""
    lo = min(float(a.min()), float(b.min())) - span
    hi = max(float(a.max()), float(b.max())) + span
    edges = np.arange(lo, hi + step, step)
    hb, _ = np.histogram(b, bins=edges)
    best_s, best_score = 0.0, -1.0
    for s in np.arange(-span, span + step, step):
        ha, _ = np.histogram(a + s, bins=edges)
        score = float(np.minimum(ha, hb).sum())
        if score > best_score:
            best_score, best_s = score, float(s)
    frac = best_score / (min(len(a), len(b)) + 1e-9)
    return best_s, frac


def estimate_plan_transform(dxf_segments: np.ndarray, dtdx_wall_pts_plan: np.ndarray,
                            max_residual: float = 0.5) -> dict:
    """Estimate the 2D map DXF(x, y) -> dtdx(x, z) as a PURE TRANSLATION (both are
    the same Revit export). X and Y are shifted independently by 1D histogram
    cross-correlation, then the residual = median nearest-neighbour distance from the
    translated DXF wall midpoints to the dtdx wall points. residual > max_residual
    -> ok=False (refuse a forced fit rather than invent an alignment).

    dxf_segments: (N,2,2) metres (load_wall_segments output).
    dtdx_wall_pts_plan: (M,2) dtdx wall vertices projected to (x, z), metres.
    Returns {"translation": (tx,ty), "residual": float, "ok": bool, "method"}.
    """
    from scipy.spatial import cKDTree
    seg = np.asarray(dxf_segments, dtype=np.float64)
    dst = np.asarray(dtdx_wall_pts_plan, dtype=np.float64)
    if seg.size == 0 or dst.size == 0:
        return {"translation": (0.0, 0.0), "residual": float("inf"), "ok": False,
                "method": "empty input"}
    mid = seg.mean(axis=1)
    tx, fx = _best_shift(mid[:, 0], dst[:, 0])
    ty, fy = _best_shift(mid[:, 1], dst[:, 1])
    # subsample for the residual NN query (both sides can be large)
    m = mid[np.linspace(0, len(mid) - 1, min(4000, len(mid))).astype(int)] + np.array([tx, ty])
    d = dst[np.linspace(0, len(dst) - 1, min(60000, len(dst))).astype(int)]
    resid = float(np.median(cKDTree(d).query(m, k=1, workers=-1)[0]))
    return {"translation": (round(tx, 3), round(ty, 3)), "residual": round(resid, 3),
            "ok": bool(resid <= max_residual), "overlap": (round(fx, 3), round(fy, 3)),
            "method": "pure_translation_hist_xcorr"}


def _resample_polyline(xz: np.ndarray, step: float = 0.3) -> np.ndarray:
    xz = np.asarray(xz, dtype=np.float64)
    if len(xz) < 2:
        return xz
    seglen = np.linalg.norm(np.diff(xz, axis=0), axis=1)
    total = float(seglen.sum())
    if total < 1e-6:
        return xz
    d = np.concatenate([[0.0], np.cumsum(seglen)])
    u = np.arange(0.0, total, step)
    return np.column_stack([np.interp(u, d, xz[:, 0]), np.interp(u, d, xz[:, 1])])


def corridor_widths_near(segments: np.ndarray, traj_xz: np.ndarray, radius: float = 2.5,
                         width_range: tuple = (0.6, 4.0), parallel_deg: float = 15.0,
                         samples: int = 40) -> tuple[list, dict]:
    """Local clear widths (metres) of the corridor a trajectory locus runs through.

    A LOCAL perpendicular measure, not a global face projection: on a dense FAB
    plan the corridor-parallel walls of many rooms pile up on one normal axis and
    fabricate phantom faces between the real corridor walls. Instead, at each point
    sampled along the trajectory we keep only walls whose ALONG-corridor span
    actually brackets that point (a wall truly beside the walker, not one from a room
    further down the run) and take the nearest such wall on each side — the corridor
    the walker is in right there. The distinct local widths over the walk are the
    candidate corridor clear widths (the DXF replacement for the furniture-noisy
    model_corridor_widths triangle gaps).

    Returns (sorted_distinct_widths, info); info carries the median/representative
    width and the local-width sample count. Empty list + a "fail" reason when no
    facing pair is found in width_range."""
    seg = np.asarray(segments, dtype=np.float64)
    traj = _resample_polyline(np.asarray(traj_xz, dtype=np.float64))
    info: dict = {"n_segments": int(len(seg))}
    if len(seg) == 0 or len(traj) < 2:
        info["fail"] = "empty segments or trajectory"
        return [], info
    origin = traj.mean(0)
    c = traj - origin
    axis = np.linalg.eigh(c.T @ c)[1][:, -1]
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    normal = np.array([-axis[1], axis[0]])

    d = seg[:, 1] - seg[:, 0]
    dn = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    parallel = np.abs(dn @ axis) >= np.cos(np.deg2rad(parallel_deg))     # walls running along the corridor
    ps = seg[parallel]
    if len(ps) < 2:
        info["fail"] = "fewer than two corridor-parallel wall segments"
        return [], info
    a0 = (ps[:, 0] - origin) @ axis; a1 = (ps[:, 1] - origin) @ axis
    alo, ahi = np.minimum(a0, a1), np.maximum(a0, a1)                    # each wall's along-corridor span
    woff = (ps.mean(axis=1) - origin) @ normal                          # each wall's normal offset

    tj_a = (traj - origin) @ axis
    tj_n = (traj - origin) @ normal
    s_samples = np.linspace(float(tj_a.min()), float(tj_a.max()), samples)
    n_at = np.interp(s_samples, np.sort(tj_a), tj_n[np.argsort(tj_a)])   # walker's normal offset along the run
    lo, hi = width_range
    local = []
    for sp, npp in zip(s_samples, n_at):
        beside = (alo <= sp) & (ahi >= sp)                              # wall truly spans this along-position
        nb = woff[beside]
        below = nb[nb < npp - 0.02]; above = nb[nb > npp + 0.02]
        if below.size and above.size and (float(above.min() - below.max()) <= 2 * radius):
            w = float(above.min() - below.max())
            if lo <= w <= hi:
                local.append(w)
    info["n_local_samples"] = int(len(local))
    if not local:
        info["fail"] = "no facing wall pair spanning the trajectory in width_range"
        return [], info
    local = np.array(local)
    info["median_width"] = round(float(np.median(local)), 3)
    out = sorted({round(float(w), 3) for w in local})
    return out, info


def door_positions_near(dxf_path, traj_xz: np.ndarray, radius: float = 3.0) -> tuple[np.ndarray, dict]:
    """A-DOOR block INSERT positions (metres) within `radius` of the trajectory.
    Stored for later forward-axis calibration; not used by the scale yet."""
    ezdxf = _ezdxf()
    from scipy.spatial import cKDTree
    doc = ezdxf.readfile(str(dxf_path))
    scale = _UNIT_TO_M.get(int(doc.header.get("$INSUNITS", 4) or 4), 0.001)
    pts = []
    for e in doc.modelspace().query("INSERT"):
        if str(e.dxf.layer).startswith(_DOOR_LAYERS):
            pts.append((e.dxf.insert.x * scale, e.dxf.insert.y * scale))
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    traj = np.asarray(traj_xz, dtype=np.float64)
    if len(pts) == 0 or len(traj) < 1:
        return np.zeros((0, 2)), {"n_doors_total": int(len(pts)), "n_near": 0}
    near = pts[cKDTree(traj).query(pts, k=1, workers=-1)[0] < radius]
    return near, {"n_doors_total": int(len(pts)), "n_near": int(len(near))}

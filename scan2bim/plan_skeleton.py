"""DXF floor-plan SKELETON (corridor centrelines, corners, doors) + the PLAN-MATCH
CANDIDATE SCHEMA that `scan2bim/coarse_match.py` and `tools/build_coplay.py
--plan-match` share as their API contract.

WHY a skeleton
--------------
`dxf_plan.corridor_widths_near` / `forward_scale.corridor_open_boundary` answer local
questions ("how wide is the corridor beside this trajectory?", "where does the facing
pair end?") and therefore need a trajectory that is ALREADY roughly placed. Coarse
matching has to work the other way round: derive the placement from the plan itself.
That needs the plan reduced to a small, ordered set of matchable landmarks:

    corridor legs (centreline + clear width + arclength)
    corners       (where two legs meet, with the traversal turn angle)
    doors         (A-DOOR inserts, pinned to a leg at an arclength s)

which is exactly the recon side's observable vocabulary (walked arclength, turn
events, door-passing events). `plan_events()` emits that vocabulary with stable ids so
a matcher can talk about "which plan landmark did recon event k hit" and, just as
importantly, about WHICH ONES IT FAILED TO EXPLAIN (outlier list).

METHOD (honest about what it is)
--------------------------------
1. Dominant wall orientation theta0 (length-weighted, folded to [0,90)) -> rotate the
   plan axis-aligned. Only the two Manhattan families are used; diagonal walls are
   ignored (recorded in info, never silently "fixed").
2. Per family, wall segments are clustered into wall LINES by offset and their spans
   merged (gaps up to `wall_join` are bridged, so a door opening does not cut a wall).
3. A pair of lines whose offset gap is inside `width_range` and whose spans overlap by
   at least `min_leg` is a corridor RIBBON. A perpendicular wall that really crosses
   the ribbon (covers >= `cross_frac` of the gap) SPLITS it — that is what rejects room
   interiors, which are short boxes closed by their own end walls.
4. Nested ribbons (the two faces of a thick wall produce several) are merged with the
   MINIMUM gap kept, i.e. the CLEAR width — the same quantity dxf_plan measures.
5. Legs are extended to their mutual intersections -> corner/tee nodes; unattached
   endpoints become "end" nodes; doors are attached to the nearest leg.

No step invents a landmark: an empty/failing input returns empty lists plus
`info["fail"]`, mirroring `dxf_plan.estimate_plan_transform`'s refusal philosophy.

FRAMES / UNITS
--------------
Everything is METRES in the DXF plan frame (x, y), which on the Gasan_7F export maps
to the dtdx model plan (x, z) by a near-identity translation
(`dxf_plan.estimate_plan_transform` measures it and refuses a forced fit). Angles are
DEGREES unless a name ends in `_rad`. All returned values are JSON-serialisable
(plain floats/ints/lists, rounded to mm).
"""
from __future__ import annotations

import numpy as np

from scan2bim import dxf_plan                     # read-only reuse (walls + A-DOOR inserts)

# --------------------------------------------------------------------------------------
# SECTION 1 — PLAN-MATCH CANDIDATE SCHEMA (the dev-wire contract; do not weaken)
# --------------------------------------------------------------------------------------

SKELETON_SCHEMA = "coplay.plan_skeleton/1.0"
PLAN_MATCH_SCHEMA = "coplay.plan_match/1.0"

#: Transform carried by every candidate. COMPOSITION ORDER (must match
#: tools/build_coplay.py::place_rigid, which flips X first, then scales, then yaws,
#: then offsets — build_coplay.py:993 / :1069 / :1028 / :1043):
#:
#:     p_plan = R(yaw_deg) @ A(theta_deg; s_f, s_h) @ diag(chi, 1) @ q_recon + translation
#:
#: with A(theta; s_f, s_h) = R(theta) diag(s_f, s_h) R(theta)^T, i.e. exactly
#: `forward_scale.anisotropic_scale_tensor(theta_rad, s_f, s_h)`. q_recon is the
#: gravity-aligned recon plan vector (x, z). `apply_candidate_transform()` below IS the
#: normative implementation — consumers must use it rather than re-deriving the order.
TRANSFORM_FIELDS = {
    "yaw_deg": "recon->plan rotation in the plan XY(=model XZ) plane, DEGREES, applied "
               "AFTER chi and the scale tensor (same convention as build_coplay info['yaw'])",
    "chi": "handedness flip, EXACTLY -1 or +1: recon x is multiplied by chi FIRST "
           "(mirrored recon; build_coplay.py:993)",
    "s_h": "lateral (corridor-width) scale, metres per recon unit; > 0",
    "s_f": "forward (walk-direction) scale, metres per recon unit; > 0. s_f == s_h is "
           "the isotropic case and then theta_deg is irrelevant",
    "theta_deg": "heading of the s_f axis IN THE chi-FLIPPED RECON frame, DEGREES "
                 "(the leg-A heading used by forward_scale)",
    "translation": "[tx, ty] metres, added LAST in the plan frame",
}

#: One ranked matching hypothesis.
CANDIDATE_FIELDS = {
    "rank": "int, 0 = best; assigned by finalize_match (dense, score-descending)",
    "transform": "dict with exactly TRANSFORM_FIELDS",
    "score": "float in [0, 1], higher is better; comparable ONLY within one result",
    "inlier_ratio": "float in [0, 1] = n_inliers / n_events",
    "n_inliers": "int",
    "n_events": "int, recon events offered to the matcher",
    "residual": "float metres, robust residual over the INLIERS ONLY (outliers are "
                "excluded, never absorbed); the reducer is the matcher's — see "
                "coarse_match.RESID_AGG",
    "matches": "list of [recon_event_index:int, plan_event_id:str] inlier pairs",
    "outliers": "list of OUTLIER_FIELDS dicts — the events this candidate does NOT "
                "explain. MUST be emitted even when the candidate wins (partial "
                "matching is expected: the walk covers only part of the floor)",
    "method": "str, matcher variant that produced it (e.g. 'corner2+trimmed')",
}

#: An unexplained event. `plan_event_id` is None when nothing plausible was near.
OUTLIER_FIELDS = {
    "kind": "one of 'leg' | 'corner' | 'door' (an ASSOCIABLE recon event the candidate "
            "could not explain) or 'span' (a ~one-corridor-width piece of the walk at "
            "which the DRAWING contradicts the walked path — see coarse_match's span "
            "change scan. 'span' outliers are a CHANGE REPORT, not an association "
            "failure: they are never counted in n_events/inlier_ratio, so they cannot "
            "move a verdict)",
    "event_index": "int, index into the recon event list",
    "plan_event_id": "str id from plan_events(), or None",
    "residual": "float metres (or degrees for a corner turn), how badly it missed",
    "reason": "short str, e.g. 'no_plan_door_within_tol'",
}

#: Result envelope returned by coarse_match and consumed by build_coplay --plan-match.
RESULT_FIELDS = {
    "schema": f"str, always {PLAN_MATCH_SCHEMA!r}",
    "status": "one of STATUSES",
    "ok": "bool, MUST equal (status == 'ok') — mirrors estimate_plan_transform['ok']",
    "best": "the winning candidate dict, or None. MUST be None unless status == 'ok'",
    "candidates": "list of candidates, rank-ordered. Kept even on hold/reject so a "
                  "human or a second source can adjudicate the tie",
    "hold_reason": "str from HOLD_REASONS/REJECT_REASONS when status != 'ok', else None",
    "margin": "float score gap best-vs-runner-up, or None when there is one candidate",
    "gates": "dict of the thresholds actually applied (margin_min, min_inlier_ratio, "
             "max_residual) — a report must state the bar it cleared",
    "info": "free-form diagnostics dict (never load-bearing for the verdict)",
}

STATUSES = ("ok", "hold", "reject")

#: HOLD = the evidence does not single out an answer -> NEVER auto-confirm. The design
#: invariant of this team: ambiguity is reported, not resolved by picking the top score.
HOLD_REASONS = (
    "no_candidate",          # nothing was generated at all
    "ambiguous_margin",      # top two scores within margin_min -> tie
    "insufficient_events",   # too few recon events to constrain the transform
    "no_plan_skeleton",      # the plan yielded no usable corridor legs
    "conflicting_sources",   # two independent estimates disagree beyond tolerance
)

#: REJECT = a candidate existed but failed a hard gate -> refuse the fit (do not
#: fabricate an alignment), same stance as estimate_plan_transform(residual > max).
REJECT_REASONS = (
    "residual_exceeded",
    "inlier_ratio_below_min",
    "scale_out_of_band",
)

DEFAULT_GATES = {"margin_min": 0.08, "min_inlier_ratio": 0.60, "max_residual": 0.60}


def make_transform(yaw_deg: float, chi: int, s_h: float, s_f: float,
                   theta_deg: float, translation) -> dict:
    """Schema-valid TRANSFORM_FIELDS dict (values rounded for JSON stability)."""
    t = np.asarray(translation, dtype=np.float64).reshape(2)
    return {"yaw_deg": round(float(yaw_deg), 3), "chi": int(np.sign(chi) or 1),
            "s_h": round(float(s_h), 5), "s_f": round(float(s_f), 5),
            "theta_deg": round(float(theta_deg), 3),
            "translation": [round(float(t[0]), 4), round(float(t[1]), 4)]}


def make_candidate(transform: dict, score: float, n_inliers: int, n_events: int,
                   residual: float, matches=None, outliers=None,
                   method: str = "unspecified") -> dict:
    """Schema-valid candidate. inlier_ratio is DERIVED (never passed in) so it can
    never disagree with n_inliers/n_events."""
    n_ev = int(n_events)
    n_in = int(n_inliers)
    return {"rank": -1, "transform": dict(transform), "score": round(float(score), 5),
            "inlier_ratio": round(n_in / n_ev, 5) if n_ev > 0 else 0.0,
            "n_inliers": n_in, "n_events": n_ev, "residual": round(float(residual), 5),
            "matches": [[int(a), str(b)] for a, b in (matches or [])],
            "outliers": [dict(o) for o in (outliers or [])], "method": str(method)}


def _result(status: str, best, cands: list, reason, margin, gates: dict, info: dict) -> dict:
    return {"schema": PLAN_MATCH_SCHEMA, "status": status, "ok": status == "ok",
            "best": best, "candidates": cands, "hold_reason": reason,
            "margin": None if margin is None else round(float(margin), 5),
            "gates": dict(gates), "info": dict(info or {})}


def make_hold_result(reason: str, candidates=None, info=None, gates=None,
                     margin=None) -> dict:
    """A HOLD envelope: status='hold', ok=False, best=None — by construction there is
    no way to hand back a transform while holding."""
    if reason not in HOLD_REASONS:
        raise ValueError(f"unknown hold reason {reason!r}; extend HOLD_REASONS deliberately")
    cands = _ranked(candidates or [])
    return _result("hold", None, cands, reason, margin, {**DEFAULT_GATES, **(gates or {})},
                   info or {})


def make_reject_result(reason: str, candidates=None, info=None, gates=None) -> dict:
    """A REJECT envelope (a candidate existed but failed a hard gate)."""
    if reason not in REJECT_REASONS:
        raise ValueError(f"unknown reject reason {reason!r}; extend REJECT_REASONS deliberately")
    cands = _ranked(candidates or [])
    return _result("reject", None, cands, reason, None, {**DEFAULT_GATES, **(gates or {})},
                   info or {})


def _ranked(candidates) -> list:
    out = sorted((dict(c) for c in candidates), key=lambda c: -float(c["score"]))
    for i, c in enumerate(out):
        c["rank"] = i
    return out


def finalize_match(candidates, gates=None, info=None) -> dict:
    """Turn scored candidates into a schema-valid result, ENFORCING this team's
    invariants so no matcher can quietly relax them:

      * no candidates                      -> HOLD 'no_candidate'
      * best.inlier_ratio < min_inlier_ratio -> REJECT 'inlier_ratio_below_min'
      * best.residual   > max_residual      -> REJECT 'residual_exceeded'
      * (best.score - runner_up.score) < margin_min -> HOLD 'ambiguous_margin'
      * otherwise                           -> ok, best = rank 0

    Gate order is fixed (hard gates before the tie test) so the reported reason is
    deterministic; every failure found is also listed in info['gate_failures'].
    A single candidate has margin = None and is unambiguous by construction.
    """
    g = {**DEFAULT_GATES, **(gates or {})}
    cands = _ranked(candidates or [])
    inf = dict(info or {})
    if not cands:
        return make_hold_result("no_candidate", [], inf, g)
    best = cands[0]
    margin = None if len(cands) < 2 else float(best["score"]) - float(cands[1]["score"])
    fails = []
    if float(best["inlier_ratio"]) < g["min_inlier_ratio"]:
        fails.append("inlier_ratio_below_min")
    if float(best["residual"]) > g["max_residual"]:
        fails.append("residual_exceeded")
    if margin is not None and margin < g["margin_min"]:
        fails.append("ambiguous_margin")
    inf["gate_failures"] = fails
    inf["n_candidates"] = len(cands)
    for r in REJECT_REASONS:
        if r in fails:
            out = make_reject_result(r, cands, inf, g)
            out["margin"] = None if margin is None else round(float(margin), 5)
            return out
    if "ambiguous_margin" in fails:
        return make_hold_result("ambiguous_margin", cands, inf, g, margin=margin)
    return _result("ok", best, cands, None, margin, g, inf)


def validate_match_result(obj) -> tuple:
    """(ok, errors): check a result against the schema and the HOLD invariant. Both
    coarse_match's tests and build_coplay's wiring should call this so the contract is
    machine-checked instead of prose-checked."""
    e: list = []
    if not isinstance(obj, dict):
        return False, ["result is not a dict"]
    if obj.get("schema") != PLAN_MATCH_SCHEMA:
        e.append(f"schema must be {PLAN_MATCH_SCHEMA!r}, got {obj.get('schema')!r}")
    for k in RESULT_FIELDS:
        if k not in obj:
            e.append(f"missing result field {k!r}")
    st = obj.get("status")
    if st not in STATUSES:
        e.append(f"status {st!r} not in {STATUSES}")
    if obj.get("ok") != (st == "ok"):
        e.append("ok must equal (status == 'ok')")
    if st == "ok":
        if obj.get("best") is None:
            e.append("status 'ok' requires best")
        if obj.get("hold_reason") is not None:
            e.append("status 'ok' must not carry a hold_reason")
    else:
        if obj.get("best") is not None:
            e.append("non-ok result must have best=None (HOLD/REJECT never returns a transform)")
        r = obj.get("hold_reason")
        if r not in HOLD_REASONS + REJECT_REASONS:
            e.append(f"hold_reason {r!r} not in HOLD_REASONS+REJECT_REASONS")
    cands = obj.get("candidates")
    if not isinstance(cands, list):
        e.append("candidates must be a list")
        return not e, e
    scores = []
    for i, c in enumerate(cands):
        if not isinstance(c, dict):
            e.append(f"candidate {i} is not a dict")
            continue
        for k in CANDIDATE_FIELDS:
            if k not in c:
                e.append(f"candidate {i} missing {k!r}")
        if c.get("rank") != i:
            e.append(f"candidate {i} rank must be {i}, got {c.get('rank')!r}")
        tf = c.get("transform")
        if not isinstance(tf, dict):
            e.append(f"candidate {i} transform must be a dict")
        else:
            for k in TRANSFORM_FIELDS:
                if k not in tf:
                    e.append(f"candidate {i} transform missing {k!r}")
            if tf.get("chi") not in (-1, 1):
                e.append(f"candidate {i} chi must be -1 or +1, got {tf.get('chi')!r}")
            for k in ("s_h", "s_f"):
                v = tf.get(k)
                if not (isinstance(v, (int, float)) and np.isfinite(v) and v > 0):
                    e.append(f"candidate {i} transform {k} must be a positive finite number")
            t = tf.get("translation")
            if not (isinstance(t, (list, tuple)) and len(t) == 2 and all(np.isfinite(t))):
                e.append(f"candidate {i} translation must be 2 finite numbers")
        ir = c.get("inlier_ratio")
        if not (isinstance(ir, (int, float)) and 0.0 <= float(ir) <= 1.0):
            e.append(f"candidate {i} inlier_ratio must be in [0,1]")
        elif c.get("n_events"):
            exp = float(c["n_inliers"]) / float(c["n_events"])
            if abs(float(ir) - exp) > 1e-4:
                e.append(f"candidate {i} inlier_ratio {ir} != n_inliers/n_events {exp:.4f}")
        if not isinstance(c.get("outliers"), list):
            e.append(f"candidate {i} outliers must be a list")
        else:
            for o in c["outliers"]:
                miss = [k for k in OUTLIER_FIELDS if k not in o]
                if miss:
                    e.append(f"candidate {i} outlier missing {miss}")
        scores.append(float(c.get("score", float("nan"))))
    if len(scores) > 1 and any(b - a > 1e-9 for a, b in zip(scores, scores[1:])):
        e.append("candidates must be sorted by descending score")
    if st == "ok" and cands and obj.get("best") is not cands[0] and obj.get("best") != cands[0]:
        e.append("best must be candidates[0]")
    return not e, e


def apply_candidate_transform(transform: dict, pts_recon_xz) -> np.ndarray:
    """NORMATIVE implementation of the candidate transform (see TRANSFORM_FIELDS):

        p = R(yaw) @ A(theta; s_f, s_h) @ diag(chi, 1) @ q + t

    pts_recon_xz: (M,2) gravity-aligned recon (x, z). Returns (M,2) plan metres.
    Reuses forward_scale.anisotropic_scale_tensor so the anisotropy convention cannot
    drift from the placement code."""
    from scan2bim.forward_scale import anisotropic_scale_tensor
    q = np.asarray(pts_recon_xz, dtype=np.float64).reshape(-1, 2).copy()
    q[:, 0] *= float(transform["chi"])
    A = anisotropic_scale_tensor(np.deg2rad(float(transform["theta_deg"])),
                                float(transform["s_f"]), float(transform["s_h"]))
    R = _rot(float(transform["yaw_deg"]))
    t = np.asarray(transform["translation"], dtype=np.float64).reshape(2)
    return (q @ A.T) @ R.T + t


# --------------------------------------------------------------------------------------
# SECTION 2 — plan skeleton extraction
# --------------------------------------------------------------------------------------

#: Corridor clear-width band (metres). The Gasan_7F corridor measures ~1.82 m; rooms
#: and lobbies are wider and are meant to fall outside this band.
DEFAULT_WIDTH_RANGE = (1.2, 2.6)


def _rot(a_deg: float) -> np.ndarray:
    r = np.deg2rad(float(a_deg))
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _dirs(seg: np.ndarray):
    d = seg[:, 1] - seg[:, 0]
    return d, np.linalg.norm(d, axis=1)


def principal_angle_deg(segments, tol_deg: float = 3.0) -> float:
    """Length-weighted dominant wall orientation folded to [0, 90) degrees: a 1-degree
    histogram peak refined by the (period-90) weighted mean of its neighbourhood."""
    seg = np.asarray(segments, dtype=np.float64).reshape(-1, 2, 2)
    if len(seg) == 0:
        return 0.0
    d, L = _dirs(seg)
    keep = L > 1e-9
    if not keep.any():
        return 0.0
    ang = np.degrees(np.arctan2(d[keep, 1], d[keep, 0])) % 90.0
    w = L[keep]
    hist, edges = np.histogram(ang, bins=90, range=(0.0, 90.0), weights=w)
    k = int(np.argmax(hist))
    peak = 0.5 * float(edges[k] + edges[k + 1])
    dd = (ang - peak + 45.0) % 90.0 - 45.0
    m = np.abs(dd) <= tol_deg
    if not m.any():
        return float(peak % 90.0)
    return float((peak + float(np.average(dd[m], weights=w[m]))) % 90.0)


def _wall_lines(seg: np.ndarray, axis: int, parallel_deg: float, offset_tol: float,
                wall_join: float, min_seg: float) -> list:
    """Wall LINES of one Manhattan family in the rotated frame.
    axis 0: walls running along +x (u = x, offset v = y). axis 1: along +y (u = y, v = x).
    Returns [{"v", "spans": [(lo,hi)...], "n", "length"}] sorted by v."""
    d, L = _dirs(seg)
    ok = L > min_seg
    dn = d / (L[:, None] + 1e-12)
    sin_tol = np.sin(np.deg2rad(parallel_deg))
    if axis == 0:
        par = ok & (np.abs(dn[:, 1]) <= sin_tol)
        u0, u1, v = seg[:, 0, 0], seg[:, 1, 0], seg[:, :, 1].mean(axis=1)
    else:
        par = ok & (np.abs(dn[:, 0]) <= sin_tol)
        u0, u1, v = seg[:, 0, 1], seg[:, 1, 1], seg[:, :, 0].mean(axis=1)
    idx = np.nonzero(par)[0]
    if idx.size == 0:
        return []
    order = idx[np.argsort(v[idx])]
    groups, cur = [], [int(order[0])]
    for i in order[1:]:
        if v[i] - v[cur[0]] <= offset_tol:
            cur.append(int(i))
        else:
            groups.append(cur)
            cur = [int(i)]
    groups.append(cur)
    out = []
    for g in groups:
        w = L[g]
        spans = sorted((float(min(u0[i], u1[i])), float(max(u0[i], u1[i]))) for i in g)
        merged: list = []
        for lo, hi in spans:
            if merged and lo <= merged[-1][1] + wall_join:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        out.append({"v": float(np.average(v[g], weights=w)), "n": len(g),
                    "length": float(w.sum()), "spans": [(a, b) for a, b in merged]})
    return out


def _cross_bars(seg: np.ndarray, axis: int, parallel_deg: float, min_seg: float):
    """Segments PERPENDICULAR to a ribbon of `axis`, as (u_at, v_lo, v_hi)."""
    d, L = _dirs(seg)
    ok = L > min_seg
    dn = d / (L[:, None] + 1e-12)
    sin_tol = np.sin(np.deg2rad(parallel_deg))
    if axis == 0:                                   # ribbon along x -> bars along y
        per = ok & (np.abs(dn[:, 0]) <= sin_tol)
        u_at = seg[:, :, 0].mean(axis=1)
        a, b = seg[:, 0, 1], seg[:, 1, 1]
    else:
        per = ok & (np.abs(dn[:, 1]) <= sin_tol)
        u_at = seg[:, :, 1].mean(axis=1)
        a, b = seg[:, 0, 0], seg[:, 1, 0]
    return u_at[per], np.minimum(a, b)[per], np.maximum(a, b)[per]


def _overlaps(A, B) -> list:
    out = []
    for a0, a1 in A:
        for b0, b1 in B:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi > lo:
                out.append((lo, hi))
    return sorted(out)


def _ribbons(lines: list, bars, width_range, min_leg: float, cross_frac: float,
             eps: float = 0.05) -> list:
    """Facing wall-line pairs -> corridor ribbons, split by walls that cross them."""
    u_at, v_lo, v_hi = bars
    out = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            gap = lines[j]["v"] - lines[i]["v"]
            if not (width_range[0] <= gap <= width_range[1]):
                continue
            lo_v, hi_v = lines[i]["v"], lines[j]["v"]
            v_mid = 0.5 * (lo_v + hi_v)
            for a0, a1 in _overlaps(lines[i]["spans"], lines[j]["spans"]):
                if a1 - a0 < min_leg:
                    continue
                cover = np.minimum(v_hi, hi_v) - np.maximum(v_lo, lo_v)
                cut = (u_at > a0 + eps) & (u_at < a1 - eps) & (cover >= cross_frac * gap)
                edges = [a0] + sorted(float(x) for x in u_at[cut]) + [a1]
                for p0, p1 in zip(edges, edges[1:]):
                    if p1 - p0 >= min_leg:
                        out.append({"u0": float(p0), "u1": float(p1), "v": float(v_mid),
                                    "width": float(gap)})
    return out


def _merge_ribbons(ribs: list, dedupe_tol: float, join: float) -> list:
    """Merge nested/abutting ribbons of one family. The merged width is the MINIMUM
    gap (the CLEAR width — thick walls otherwise inflate it), the midline the
    length-weighted mean."""
    if not ribs:
        return []
    ribs = sorted(ribs, key=lambda r: (r["v"], r["u0"]))
    groups: list = []
    for r in ribs:
        for g in groups:
            if abs(r["v"] - g[0]["v"]) <= dedupe_tol and any(
                    (r["u0"] <= q["u1"] + join and q["u0"] <= r["u1"] + join) for q in g):
                g.append(r)
                break
        else:
            groups.append([r])
    out = []
    for g in groups:
        w = np.array([r["u1"] - r["u0"] for r in g], dtype=np.float64)
        widths = np.array([r["width"] for r in g], dtype=np.float64)
        out.append({"u0": min(r["u0"] for r in g), "u1": max(r["u1"] for r in g),
                    "v": float(np.average([r["v"] for r in g], weights=w)),
                    "width": float(widths.min()), "width_max": float(widths.max()),
                    "n_pairs": len(g)})
    return out


def _leg_dict(rib: dict, axis: int, M: np.ndarray) -> dict:
    """Rotated-frame ribbon -> plan-frame leg (a at u0, b at u1). `M` is the SAME
    forward matrix used to rotate the plan in (M = _rot(-theta0)); right-multiplying by
    it undoes that rotation (X = X_rot @ M because M is orthonormal)."""
    if axis == 0:
        a_r, b_r = np.array([rib["u0"], rib["v"]]), np.array([rib["u1"], rib["v"]])
    else:
        a_r, b_r = np.array([rib["v"], rib["u0"]]), np.array([rib["v"], rib["u1"]])
    a, b = a_r @ M, b_r @ M
    d = b - a
    L = float(np.linalg.norm(d))
    return {"axis": int(axis), "u0": rib["u0"], "u1": rib["u1"], "v": rib["v"],
            "a": a, "b": b, "dir": d / (L + 1e-12), "length": L,
            "width": rib["width"], "width_max": rib.get("width_max", rib["width"]),
            "n_pairs": int(rib.get("n_pairs", 1)),
            "a_wall": a.copy(), "b_wall": b.copy(), "wall_length": L}


def _pt(p) -> list:
    return [round(float(p[0]), 3), round(float(p[1]), 3)]


def corridor_skeleton(segments, doors=None, width_range=DEFAULT_WIDTH_RANGE,
                      min_leg: float = 2.5, parallel_deg: float = 12.0,
                      offset_tol: float = 0.10, wall_join: float = 1.0,
                      min_seg: float = 0.15, dedupe_tol: float = 0.35,
                      cross_frac: float = 0.6, corner_tol: float | None = None,
                      node_tol: float = 0.4, door_radius: float = 1.8,
                      max_legs: int = 24) -> dict:
    """Corridor centreline / corner / door graph from (N,2,2) wall segments (metres,
    `dxf_plan.load_wall_segments` output).

    Returns a JSON-serialisable dict:
      legs      [{"id","a","b","dir","length","width","width_max","a_wall","b_wall",
                  "wall_length","nodes":[a_node,b_node],"doors":[door ids]}]
                 a/b are the centreline ends AFTER extension to corner nodes;
                 a_wall/b_wall are the raw facing-pair ends (b_wall is the corridor
                 OPEN BOUNDARY, i.e. the same landmark as
                 forward_scale.corridor_open_boundary's L_end).
      nodes     [{"id","kind":"corner"|"tee"|"end"|"door","xy","legs",...}]
      doors     [{"id","xy","leg","s","offset"}]  s = arclength from that leg's a end
      walls     [[[x0,y0],[x1,y1]], ...] — every input segment longer than `min_seg`,
                 verbatim (metres, plan frame). The skeleton is a REDUCTION (facing wall
                 pairs -> centrelines) and necessarily drops what it cannot pair: a new
                 partition across the corridor, a wall that moved out of its offset
                 cluster. `coarse_match`'s span change scan needs those raw walls to ask
                 "does the drawing put a wall where the walk actually went, and is the
                 drawing's corridor as wide there as the rest of the walk found it" —
                 questions the leg/corner/door graph alone cannot answer. Carrying them
                 here (rather than re-reading the DXF) keeps the matcher's input ONE
                 object. Never used for centreline geometry.
      info      {"theta0_deg","n_segments","n_walls","n_wall_lines","n_legs",
                 "median_width",...} and "fail" when nothing usable was found (no
                 fabricated leg, ever).

    `doors` may be an (M,2) array of plan-frame door positions; `plan_skeleton()` fills
    it from the DXF's A-DOOR inserts instead.
    """
    seg0 = np.asarray(segments, dtype=np.float64).reshape(-1, 2, 2)
    info: dict = {"n_segments": int(len(seg0)), "width_range": list(width_range),
                  "min_leg": float(min_leg)}
    empty = {"schema": SKELETON_SCHEMA, "legs": [], "nodes": [], "doors": [],
             "centerline": [], "walls": [], "info": info}
    if len(seg0) == 0:
        info["fail"] = "no wall segments"
        return empty

    theta0 = principal_angle_deg(seg0)
    info["theta0_deg"] = round(theta0, 3)
    M = _rot(-theta0)                                    # rotate into the axis-aligned frame
    seg = seg0.reshape(-1, 2) @ M.T
    seg = seg.reshape(-1, 2, 2)
    d, L = _dirs(seg)
    dn = d / (L[:, None] + 1e-12)
    sin_tol = np.sin(np.deg2rad(parallel_deg))
    aligned = (np.abs(dn[:, 0]) <= sin_tol) | (np.abs(dn[:, 1]) <= sin_tol)
    info["n_diagonal_ignored"] = int((~aligned & (L > min_seg)).sum())

    legs: list = []
    n_lines = 0
    for axis in (0, 1):
        lines = _wall_lines(seg, axis, parallel_deg, offset_tol, wall_join, min_seg)
        n_lines += len(lines)
        bars = _cross_bars(seg, axis, parallel_deg, min_seg)
        ribs = _merge_ribbons(_ribbons(lines, bars, width_range, min_leg, cross_frac),
                              dedupe_tol, wall_join)
        legs.extend(_leg_dict(r, axis, M) for r in ribs)
    info["n_wall_lines"] = n_lines
    if not legs:
        info["fail"] = ("no facing wall pair with a clear width in "
                        f"{tuple(width_range)} m spanning >= {min_leg} m")
        return empty
    legs.sort(key=lambda g: (g["axis"], round(g["v"], 3), g["u0"]))
    if len(legs) > max_legs:                             # keep the longest, deterministically
        info["n_legs_dropped"] = len(legs) - max_legs
        legs = sorted(sorted(legs, key=lambda g: -g["length"])[:max_legs],
                      key=lambda g: (g["axis"], round(g["v"], 3), g["u0"]))
    for i, g in enumerate(legs):
        g["id"] = i

    # --- corners: intersections of perpendicular legs, measured on the RAW extents ---
    hits: list = []
    for gi in legs:
        for gj in legs:
            if gj["id"] <= gi["id"] or gi["axis"] == gj["axis"]:
                continue
            tol = corner_tol if corner_tol is not None else max(
                1.2, 0.75 * (gi["width"] + gj["width"]))
            # perpendicular families: each leg's u-coordinate at the crossing is the
            # OTHER leg's offset v (holds for both (axis0,axis1) orderings)
            a_i, a_j = gj["v"], gi["v"]
            if not (gi["u0"] - tol <= a_i <= gi["u1"] + tol):
                continue
            if not (gj["u0"] - tol <= a_j <= gj["u1"] + tol):
                continue
            p_r = np.array([gj["v"], gi["v"]]) if gi["axis"] == 0 else np.array([gi["v"], gj["v"]])
            hits.append({"legs": (gi["id"], gj["id"]), "u_on": (a_i, a_j),
                         "xy": p_r @ M, "tol": tol})

    nodes: list = []

    def _node(xy, kind, leg_ids, **extra) -> int:
        for nd in nodes:
            if np.linalg.norm(np.asarray(nd["_xy"]) - xy) <= node_tol and nd["kind"] == kind:
                for lid in leg_ids:
                    if lid not in nd["legs"]:
                        nd["legs"].append(int(lid))
                nd.update(extra)
                return nd["id"]
        nd = {"id": len(nodes), "kind": kind, "xy": _pt(xy), "_xy": np.asarray(xy, float),
              "legs": [int(x) for x in leg_ids], **extra}
        nodes.append(nd)
        return nd["id"]

    for h in hits:
        gi, gj = legs[h["legs"][0]], legs[h["legs"][1]]
        depth = []
        for g, u in ((gi, h["u_on"][0]), (gj, h["u_on"][1])):
            depth.append(min(abs(u - g["u0"]), abs(u - g["u1"])) if g["u0"] <= u <= g["u1"] else 0.0)
        thr = max(gi["width"], gj["width"])
        kind = "corner" if max(depth) <= thr else "tee"
        # traversal turn: incoming along gi towards the node, outgoing along gj away
        in_dir = gi["dir"] if abs(h["u_on"][0] - gi["u0"]) > abs(h["u_on"][0] - gi["u1"]) else -gi["dir"]
        out_dir = gj["dir"] if abs(h["u_on"][1] - gj["u0"]) < abs(h["u_on"][1] - gj["u1"]) else -gj["dir"]
        cr = float(in_dir[0] * out_dir[1] - in_dir[1] * out_dir[0])
        dt = float(in_dir @ out_dir)
        turn = float(np.degrees(np.arctan2(cr, dt)))
        nid = _node(h["xy"], kind, h["legs"], turn_deg=round(abs(turn), 2),
                    turn_signed_deg=round(turn, 2), turn_order=[int(gi["id"]), int(gj["id"])])
        for g, u in ((gi, h["u_on"][0]), (gj, h["u_on"][1])):     # extend to the node
            if u < g["u0"]:
                g["u0"] = float(u)
            elif u > g["u1"]:
                g["u1"] = float(u)
            g["_node_at"] = g.get("_node_at", [])
            g["_node_at"].append((float(u), nid))

    for g in legs:                                           # rebuild ends after extension
        upd = _leg_dict({"u0": g["u0"], "u1": g["u1"], "v": g["v"], "width": g["width"],
                         "width_max": g["width_max"], "n_pairs": g["n_pairs"]}, g["axis"], M)
        g["a"], g["b"], g["dir"], g["length"] = upd["a"], upd["b"], upd["dir"], upd["length"]
        na = nb = None
        mids = []
        tol_end = max(node_tol, g["width"])
        for u, nid in g.get("_node_at", []):
            if abs(u - g["u0"]) <= tol_end:
                na = nid
            elif abs(u - g["u1"]) <= tol_end:
                nb = nid
            else:
                mids.append(int(nid))                        # a tee crossing mid-leg
        g["mid_nodes"] = mids
        if na is None:
            na = _node(g["a"], "end", [g["id"]])
        if nb is None:
            nb = _node(g["b"], "end", [g["id"]])
        g["nodes"] = [int(na), int(nb)]
        g["doors"] = []

    skel = {"schema": SKELETON_SCHEMA,
            "legs": [{"id": g["id"], "a": _pt(g["a"]), "b": _pt(g["b"]),
                      "dir": [round(float(g["dir"][0]), 5), round(float(g["dir"][1]), 5)],
                      "length": round(g["length"], 3), "width": round(g["width"], 3),
                      "width_max": round(g["width_max"], 3), "n_pairs": g["n_pairs"],
                      "a_wall": _pt(g["a_wall"]), "b_wall": _pt(g["b_wall"]),
                      "wall_length": round(g["wall_length"], 3),
                      "nodes": g["nodes"], "mid_nodes": g["mid_nodes"],
                      "doors": []} for g in legs],
            "nodes": [{k: v for k, v in nd.items() if not k.startswith("_")} for nd in nodes],
            "doors": [], "centerline": [[_pt(g["a"]), _pt(g["b"])] for g in legs],
            "walls": [[_pt(p), _pt(q)] for p, q in seg0
                      if float(np.linalg.norm(q - p)) > min_seg],
            "info": info}
    info["n_walls"] = len(skel["walls"])
    info["n_legs"] = len(skel["legs"])
    info["median_width"] = round(float(np.median([g["width"] for g in skel["legs"]])), 3)
    info["total_leg_length"] = round(float(sum(g["length"] for g in skel["legs"])), 3)
    info["n_corners"] = sum(1 for n in skel["nodes"] if n["kind"] == "corner")
    info["n_tees"] = sum(1 for n in skel["nodes"] if n["kind"] == "tee")
    info["n_ends"] = sum(1 for n in skel["nodes"] if n["kind"] == "end")
    if doors is not None:
        attach_doors(skel, doors, radius=door_radius)
    return skel


def attach_doors(skel: dict, doors, radius: float = 1.8, end_slack: float = 0.5) -> dict:
    """Pin door positions (M,2 plan metres) onto the nearest corridor leg, storing the
    arclength s from that leg's `a` end (the door-passing event's plan coordinate) and
    the signed lateral offset. Doors farther than `radius` from every centreline are
    NOT attached (counted in info) — a door deep inside a room is not a corridor event.
    Mutates and returns `skel`."""
    pts = np.asarray(doors, dtype=np.float64).reshape(-1, 2)
    skel["doors"] = []
    for g in skel["legs"]:
        g["doors"] = []
    n_un = 0
    for k, p in enumerate(pts):
        best = None
        for g in skel["legs"]:
            a = np.asarray(g["a"], dtype=np.float64)
            u = np.asarray(g["dir"], dtype=np.float64)
            s = float((p - a) @ u)
            if not (-end_slack <= s <= g["length"] + end_slack):
                continue
            off = float((p - a) @ np.array([-u[1], u[0]]))
            if abs(off) <= radius and (best is None or abs(off) < abs(best[2])):
                best = (g, s, off)
        if best is None:
            n_un += 1
            continue
        g, s, off = best
        did = len(skel["doors"])
        skel["doors"].append({"id": did, "xy": _pt(p), "leg": int(g["id"]),
                              "s": round(float(np.clip(s, 0.0, g["length"])), 3),
                              "offset": round(off, 3)})
        g["doors"].append(did)
        skel["nodes"].append({"id": len(skel["nodes"]), "kind": "door", "xy": _pt(p),
                              "legs": [int(g["id"])], "door": did,
                              "s": skel["doors"][-1]["s"]})
    skel["info"]["n_doors_attached"] = len(skel["doors"])
    skel["info"]["n_doors_unattached"] = int(n_un)
    return skel


def centerline_points(skel: dict, step: float = 0.25, leg_ids=None) -> np.ndarray:
    """Corridor centrelines resampled to (M,2) plan points — the "trajectory" that
    dxf_plan.door_positions_near / corridor_widths_near expect, now derived from the
    plan instead of from an already-placed recon walk.

    `leg_ids` restricts the output to those legs. MEASURED CAVEAT: the width/boundary
    helpers (`dxf_plan.corridor_widths_near`, `forward_scale.corridor_open_boundary`)
    fit ONE global axis, so an L-shaped multi-leg centreline gives them a diagonal axis
    with no parallel walls ("fewer than two corridor-parallel wall segments"). Pass a
    SINGLE leg id when feeding those two; the full set is fine for the door search,
    which is axis-free (nearest-neighbour only)."""
    out = []
    for g in skel.get("legs", []):
        if leg_ids is not None and g["id"] not in leg_ids:
            continue
        a = np.asarray(g["a"], dtype=np.float64)
        b = np.asarray(g["b"], dtype=np.float64)
        n = max(2, int(np.ceil(float(np.linalg.norm(b - a)) / max(step, 1e-3))) + 1)
        out.append(a + (b - a) * np.linspace(0.0, 1.0, n)[:, None])
    return np.vstack(out) if out else np.zeros((0, 2))


def plan_events(skel: dict) -> list:
    """The plan-side event vocabulary a coarse matcher scores against, with the stable
    ids used by candidate['matches'] / candidate['outliers']['plan_event_id']:

        {"id": "leg:0",    "kind": "leg",    "length", "width", "a", "b"}
        {"id": "corner:3", "kind": "corner", "xy", "turn_deg", "legs"}
        {"id": "door:2",   "kind": "door",   "xy", "xy_pass", "leg", "s"}

    Order: legs (by id), then corners/tees (by node id), then doors (by id).

    `xy_pass` IS THE ONE A MATCHER MUST COMPARE AGAINST, and the distinction is not
    cosmetic. `xy` is where the door IS — on a wall FACE, half the clear width off the
    centreline (`attach_doors` records that as `offset`). A walk observes a door as a
    PASSING event, i.e. on the centreline. Scoring the walk's passing point against the
    door's own `xy` therefore builds a fixed |offset| ~ W/2 error into every correct
    match — 0.91 m on a 1.82 m corridor, which is the FULL D2 offset budget — and a
    matcher that minimises it slides the whole placement sideways by that much to make
    the doors "fit" (measured: it traded an inlier on the untouched corridor for one on
    the moved wall and still reported median residual 0.0015 m). `xy_pass` projects the
    door back onto its leg's centreline (a + s*dir), which is the event the walk actually
    produced. Doors with no host leg keep `xy_pass = xy`."""
    ev = []
    legs_by_id = {}
    for g in skel.get("legs", []):
        ev.append({"id": f"leg:{g['id']}", "kind": "leg", "length": g["length"],
                   "width": g["width"], "a": g["a"], "b": g["b"]})
        legs_by_id[int(g["id"])] = g
    for nd in skel.get("nodes", []):
        if nd["kind"] in ("corner", "tee"):
            ev.append({"id": f"corner:{nd['id']}", "kind": "corner", "xy": nd["xy"],
                       "turn_deg": nd.get("turn_deg"), "legs": nd["legs"],
                       "node_kind": nd["kind"]})
    for d in skel.get("doors", []):
        g = legs_by_id.get(int(d["leg"])) if d.get("leg") is not None else None
        if g is None:
            xy_pass = list(d["xy"])
        else:
            p = (np.asarray(g["a"], dtype=np.float64)
                 + float(d["s"]) * np.asarray(g["dir"], dtype=np.float64))
            xy_pass = _pt(p)
        ev.append({"id": f"door:{d['id']}", "kind": "door", "xy": d["xy"],
                   "xy_pass": xy_pass, "leg": d["leg"], "s": d["s"]})
    return ev


def plan_skeleton(dxf_path, layers=None, door_radius: float = 1.8, **kwargs) -> dict:
    """Full skeleton straight off a DXF: walls via `dxf_plan.load_wall_segments`, doors
    via `dxf_plan.door_positions_near` searched along the extracted centreline (not a
    recon trajectory). dxf_plan is imported unmodified; ezdxf stays a lazy import."""
    segs = (dxf_plan.load_wall_segments(dxf_path) if layers is None
            else dxf_plan.load_wall_segments(dxf_path, layers=layers))
    skel = corridor_skeleton(segs, door_radius=door_radius, **kwargs)
    skel["info"]["dxf_path"] = str(dxf_path)
    if not skel["legs"]:
        return skel
    traj = centerline_points(skel)
    near, dinfo = dxf_plan.door_positions_near(dxf_path, traj, radius=door_radius)
    skel["info"]["doors_dxf"] = dinfo
    attach_doors(skel, near, radius=door_radius)
    return skel

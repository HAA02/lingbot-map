"""COARSE MATCHER: recon trajectory <-> DXF plan skeleton, OUTLIER-TOLERANT and
PARTIAL. Produces ranked `coplay.plan_match/1.0` candidates (the schema and the
HOLD/REJECT invariants live in `scan2bim.plan_skeleton` and are consumed here, never
re-derived).

WHAT IS BEING MATCHED (the event vocabulary, both sides)
-------------------------------------------------------
`plan_skeleton.plan_events()` reduces the drawing to legs / corners / doors. This
module reduces the WALK to the same three things (`recon_events()`):

    leg    a straight stretch of the walk: chord + WALKED ARCLENGTH (the observable
           that fixes scale), heading
    corner an interior vertex of the simplified walk: position + turn angle/sign
    door   a door-PASSING event (arclength along the walk where a door was seen),
           interpolated onto the trajectory

A candidate transform is then seeded from ONE corner correspondence (recon corner k
<-> plan corner node, with the two traversal assignments and both chirality signs) —
plus, for each such seed, the variants whose SCALES come from a SECOND corner
correspondence instead of from the plan legs' remaining length (`_scales_from_pair`,
added because a cut leg makes the length-based guess wrong in a way no local search can
undo) — refined by trimmed re-association, and scored. Nothing is fitted globally: the walk is
expected to cover only PART of the floor and the field is expected to DISAGREE with
the drawing, so every event the winner cannot explain is emitted in `outliers` instead
of being absorbed into a least-squares residual.

WHY A CORNER IS THE MINIMAL SEED (and why 2 legs are AMBIGUOUS)
--------------------------------------------------------------
The placement model (`plan_skeleton.TRANSFORM_FIELDS`) has 6 unknowns: chi, yaw,
theta, s_f, s_h, t(2). theta is not free - it is the heading of walk leg 0 in the
chi-flipped frame - so a single corner supplies exactly enough: the two leg arclengths
give (s_f, s_h) (the anisotropic pair the fwdscale cycle measured, s_f 3.53 vs s_h
1.97), the incoming leg direction gives yaw, the corner point gives t, and the turn
SIGN decides chi.

MEASURED CONSEQUENCE, stated because it is load-bearing: with s_f and s_h both free, an
L-shaped walk (2 legs, 1 corner) matches an L-shaped corridor in TWO exact ways -
forwards, and backwards with chi flipped and s_f rescaled by the leg-length ratio. Both
have residual 0, so the tie is real, not a scoring artefact, and this matcher reports
HOLD 'ambiguous_margin' rather than picking one (design invariant (2)).

Breaking it needs evidence, not a tie-break rule, and MEASUREMENT (not intuition) says
which evidence works - tests/test_coarse_match.py pins each line below:

  * a DOOR event DOES break it. The backwards fit puts the door where the drawing has
    none, loses that event, and the margin opens to 0.22 >> margin_min 0.08.
  * MORE LEGS ALONE DOES NOT. A 3-leg U and a 4-leg staircase still tie (margin 0.0 /
    1e-5): partial traversal means a walked leg only has to FIT INSIDE its plan leg, so
    leg lengths bound the scale from one side and never pin it. Corner NODES do pin it
    (that is what `_scales_from_pair` uses) but a 2-leg L has only ONE corner, so no pair
    exists and the tie is untouched — the second-corner seed sharpens accuracy where a
    leg END moved, it does not manufacture evidence where there is none.
  * `s_h_prior=` DOES NOT break the 2-leg L either, contrary to what this docstring
    claimed before it was measured: only s_f is rescaled by the backwards fit (s_h 1.97
    -> 2.01 on the 1.82 m L fixture), so no usable s_h tolerance separates the two. What
    it DOES buy is pruning the MIRRORED candidates on a richer plan (re-measured on the
    4-leg staircase after the P2 scoring fix: 8 -> 6 candidates at the default +-25 %,
    where the margin is ALREADY 0.263 and does not move because the surviving mirror's
    s_h 1.49 is just inside that band; at +-20 % it is 8 -> 3 and the margin opens to
    0.472). A prior that agrees with no candidate is a REJECT 'scale_out_of_band' -
    never a fit pulled onto it.

TRIMMING / ROBUSTNESS
---------------------
* association is greedy one-to-one by residual, per kind, inside per-kind tolerances
  derived from the corridor clear width - a plan landmark can be claimed by only one
  walk event (repeated geometry otherwise double-counts);
* the transform is refined by a MEDIAN correction over INLIERS only (trimmed
  translation) plus a coordinate-descent search over multiplicative scale factors -
  partial traversal makes the seed's leg-length ratio biased LOW, and the factor 1.0 is
  always tried first so an already-exact seed is never moved (ties keep the seed);
* leg association is asymmetric: walking only part of a plan leg costs nothing (partial
  match), walking PAST its end DISQUALIFIES the pairing (the walk would go through a
  wall) — but the over-run never enters the cost that is minimised, because leg ends move
  whenever a wall is edited and a fit must not be steered by them (see OVER_WEIGHT);
* `residual` is the MEAN LATERAL error over the INLIERS ONLY — bounded by the
  association tolerances by construction, so a field change still cannot blow it up, but
  no inlier is free (see RESID_AGG for why the median it replaced was exploitable) — and
  the score credits it only in proportion to how much of the walk the candidate actually
  explains (see Q_BASE/W_RESID — a self-chosen subset always fits better, so an
  unnormalised residual bonus rewards explaining LESS);
* candidates that differ only numerically are deduped, otherwise one answer found from
  several corners would look like a tie and force a spurious HOLD.

REFUSAL PHILOSOPHY (inherited from dxf_plan.estimate_plan_transform)
--------------------------------------------------------------------
Every exit is one of: ok / hold / reject. HOLD when the evidence does not single out an
answer (no corner in the walk, no corner in the plan, tie inside `margin_min`); REJECT
when a candidate exists but fails a hard gate (residual, inlier ratio, scale out of the
physical band / disagreeing with `s_h_prior`). `finalize_match` applies those gates, so
this module cannot relax them by construction, and no non-ok result ever carries a
transform.

UNITS / FRAMES
--------------
Plan side: metres in the DXF plan frame. Recon side: gravity-aligned recon plan (x, z)
in RECON units (monocular, up to scale) - the candidate's s_f/s_h are metres per recon
unit. Angles are degrees.
"""
from __future__ import annotations

import numpy as np

from scan2bim.forward_scale import anisotropic_scale_tensor
from scan2bim.plan_skeleton import (DEFAULT_GATES, apply_candidate_transform,
                                    finalize_match, make_candidate, make_hold_result,
                                    make_reject_result, make_transform, plan_events)

# --------------------------------------------------------------------------------------
# tolerances / weights (all overridable; P2 tunes these against the qa perturbation suite)
# --------------------------------------------------------------------------------------

#: Floors for the association tolerances (metres / degrees).
TOL_FLOOR = {"corner_pos": 0.90, "leg_lat": 0.45, "door_pos": 1.20,
             "turn_deg": 25.0, "leg_ang_deg": 25.0}

#: ...and their corridor-clear-width multiples (the actual tolerance is the larger).
#: leg_lat 0.30*W = 0.55 m on a 1.82 m corridor: tight enough that a 0.6 m wall move
#: (the qa perturbation band is 0.3-1.0 m) leaves the leg OUTSIDE tolerance, loose
#: enough to survive walking off-centre.
TOL_PER_WIDTH = {"corner_pos": 0.60, "leg_lat": 0.30, "door_pos": 0.80}

#: An over-run past a plan leg's END costs OVER_WEIGHT x what the same distance costs
#: sideways. A DISCOUNT and not a free slack on purpose: a free slack has a cliff, and
#: with a 0.9 m cliff a placement that starts the walk 1 m inside a dead-end wall became
#: a confident 'ok' with the moved corridor unreported (measured). See `_pt_line` for why
#: the two directions differ at all.
#:
#: WHERE IT APPLIES (P2, measured — see `_associate`): over-run is an ASSOCIATION VETO
#: (a leg that runs too far past a plan leg's end is not that leg) but it is NOT part of
#: the residual VALUE that scoring and refinement minimise. Charging it as residual made
#: the matcher PAY to keep the walk inside a leg whose END had moved, and the cheapest
#: way to stop paying is to shrink the scale and slide the placement: every 'ok'
#: confirmation outside the D2 band in the cycle-12 sweep (T>=15 %) was that slide —
#: e.g. a truncated leg (0.91 m of wall removed at its start) pulled s_h 1.97 -> 1.80 and
#: the translation 1.05 m off, while the residual it reported fell to 3e-5. Ends are the
#: unreliable landmark (`_pt_line`); a fit must not be steered by them.
OVER_WEIGHT = 0.5

#: Event weights in the score: a corner pins position AND rotation, a leg only its own
#: line, a door only a point.
EVENT_WEIGHTS = {"corner": 1.5, "leg": 1.0, "door": 1.0}

#: How the per-inlier residuals are reduced to the one number `residual` that the score
#: and the `max_residual` gate see. It is the MEAN, and it used to be the median.
#:
#: The median was chosen so "a few field changes cannot inflate it" — but under this
#: matcher's own design a field change is supposed to leave the inlier set entirely (it
#: becomes an OUTLIER), so the median was not protecting against field changes, it was
#: giving HALF the inliers a free pass up to the association tolerance. Those tolerances
#: (corner_pos 1.09 m, door_pos 1.46 m on a 1.82 m corridor) are WIDER THAN THE ACCURACY
#: THIS PIPELINE CLAIMS (D2 offset 0.5 x W = 0.91 m), so a placement can slide by more
#: than the whole D2 budget, keep every landmark inside tolerance, and still report a
#: median residual of ~0. MEASURED on the moved-wall fixture: a fit slid 0.88 m — far
#: enough to blame the UNTOUCHED corridor and miss the moved one — reported median
#: 0.0125 m and outscored the correct fit; on the mean it reports 0.43 m and loses, and
#: the correct fit lands 0.14 m out with a clean change report.
#:
#: This does NOT reintroduce the least-squares behaviour the module rejects: outliers are
#: still excluded from the number entirely, and every value entering the mean is capped
#: by its kind's tolerance, so one changed landmark can move it by at most tol/N.
RESID_AGG = "mean"

#: score = f * (Q_BASE + W_RESID*resid_term*f + W_COV*coverage) with f the weighted
#: inlier fraction; Q_BASE + W_RESID + W_COV == 1 so score stays in [0, 1].
#:
#: MEASURED reason for the MULTIPLICATIVE form (an additive score was tried first and
#: was too flat): with `score = a*inliers + b*resid + c*coverage`, a candidate that
#: explains only 6 of 9 events still collects the FULL residual term, because whatever
#: little it did match, it matched exactly. A wrong placement then landed 0.048 below
#: the right one on a perturbed plan — inside margin_min — turning a recoverable match
#: into a spurious HOLD. Multiplying by the inlier fraction makes "how much of the walk
#: is explained" the dominant axis and residual/coverage only refine it.
#:
#: WHY resid_term IS MULTIPLIED BY f A SECOND TIME (P2): the residual is a median over
#: the candidate's OWN, SELF-CHOSEN inlier set, so it is not comparable across
#: candidates that explain different amounts of the walk — a smaller subset can always
#: be fitted better (selection bias, the same effect the paragraph above found and only
#: half-corrected). Left unnormalised, W_RESID's 0.25 swing outranked a two-event
#: difference in explanatory power, and MEASURED that is what lifted mirrored/backwards
#: placements into the top two: at the lowest perturbation point of the cycle-12 sweep a
#: MIRROR sat in the top two of 10 of the 12 'ambiguous_margin' HOLDs, always with a
#: near-zero residual on 2-3 fewer events than the true fit. Crediting the residual bonus
#: only for the fraction actually explained ("no credit for fitting data you excluded")
#: is the fix; it is a change of BASIS, not of any threshold.
Q_BASE, W_RESID, W_COV = 0.60, 0.25, 0.15

#: Two candidates that agree to within THE ACCURACY THIS PIPELINE CLAIMS (team DoD D2:
#: yaw <= 5 deg, offset <= 0.5 x corridor width, per-axis scale <= 10 %) are the SAME
#: answer and are deduped. Without this the margin test fires on numerical variants of
#: one placement (measured: s_f 3.35 vs 3.53 with a 0.6 m offset) instead of on genuinely
#: different placements — which would weaken, not strengthen, the HOLD invariant.
DEDUPE_YAW_DEG, DEDUPE_SCALE_REL, DEDUPE_T_FLOOR = 5.0, 0.10, 0.40

#: Multiplicative scale trials, ordered by |log f| so 1.0 (the seed) wins every tie.
SCALE_FACTORS = (1.0, 0.95, 1.05, 0.90, 1.10, 0.80, 1.25)

#: SECOND-CORNER SEEDS (`_scales_from_pair`). A one-corner seed has to GUESS the scale
#: from how much plan leg is AVAILABLE past the node, which is an upper bound the walk
#: only partly uses — and, worse, an upper bound that MOVES when the drawing is edited.
#: A second corner correspondence replaces the guess with a MEASUREMENT: the vector
#: between two plan corner NODES is observable and does not care where the legs were cut.
#:
#: `PAIR_MIN_PROJ` is the conditioning guard. The corner-to-corner vector is decomposed on
#: the walk's own (forward, lateral) axes; an axis that carries less than this fraction of
#: the vector's length is NOT measured by that pair, and the seed keeps its one-corner
#: value for that axis instead of dividing by a near-zero projection (two corners joined by
#: a single straight leg pin the scale ALONG the leg only — the honest answer is one scale,
#: not two). Both axes can never be under the guard at once while it is < 1/sqrt(2).
#: `PAIR_YAW_ITERS`: yaw is a function of the scales (see `_build`), so the linear solve is
#: iterated to a fixed point; it is exact in one step whenever the incoming walk leg lies on
#: a tensor axis (the usual case) and this many steps otherwise.
PAIR_MIN_PROJ, PAIR_YAW_ITERS = 0.20, 3

#: Physically plausible s_f/s_h anisotropy (the measured case is 3.53/1.97 = 1.79).
DEFAULT_ANISO_BAND = (0.2, 5.0)

#: Metres per recon unit; deliberately wide (a monocular recon has no scale prior).
DEFAULT_SCALE_BAND = (0.05, 50.0)


# --------------------------------------------------------------------------------------
# small geometry helpers
# --------------------------------------------------------------------------------------

def _rot(a_deg: float) -> np.ndarray:
    r = np.deg2rad(float(a_deg))
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(2)
    return v / (float(np.linalg.norm(v)) + 1e-12)


def _ang(v) -> float:
    v = np.asarray(v, dtype=np.float64).reshape(2)
    return float(np.degrees(np.arctan2(v[1], v[0])))


def _wrap180(a: float) -> float:
    return float((float(a) + 180.0) % 360.0 - 180.0)


def _pt_line(P, A, U, L):
    """(K,2) points vs the (L,) plan legs given by origin A, unit direction U and length
    L: (lateral distance to the INFINITE line, over-run past either end, projection on
    the line), each (K,L[,2]).

    LATERAL and LONGITUDINAL error are kept apart on purpose, because the two plan
    landmarks have very different certainty: a leg's CENTRELINE is pinned by two facing
    walls (the 1.82 m clear width), while its ENDS are wherever the facing pair happened
    to stop (an opening, a tee, the extension to a corner node) and move as soon as one
    wall is edited. Mixing them into one point-to-SEGMENT distance made a truncated plan
    leg look like a 0.91 m mismatch and pushed the matcher into a 5 % scale drift to
    'fix' it (measured)."""
    P = np.asarray(P, dtype=np.float64).reshape(-1, 1, 2)
    A = np.asarray(A, dtype=np.float64).reshape(1, -1, 2)
    U = np.asarray(U, dtype=np.float64).reshape(1, -1, 2)
    Ln = np.asarray(L, dtype=np.float64).reshape(1, -1)
    d = P - A
    t = np.sum(d * U, axis=2)
    lat = np.abs(d[:, :, 0] * U[:, :, 1] - d[:, :, 1] * U[:, :, 0])
    over = np.maximum(np.maximum(-t, t - Ln), 0.0)
    return lat, over, A + t[:, :, None] * U


# --------------------------------------------------------------------------------------
# SECTION 1 — recon-side events (the walk in the plan's vocabulary)
# --------------------------------------------------------------------------------------

def _dp_keep(pts: np.ndarray, tol: float) -> list:
    """Douglas-Peucker vertex indices (deterministic, iterative)."""
    n = len(pts)
    if n <= 2:
        return list(range(n))
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        ab = pts[j] - pts[i]
        d = _pt_line(pts[i + 1:j], pts[i][None, :], _unit(ab)[None, :],
                     [float(np.linalg.norm(ab))])[0][:, 0]
        k = int(np.argmax(d))
        if float(d[k]) > tol:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m))
            stack.append((m, j))
    return np.nonzero(keep)[0].tolist()


def _turn_at(pts: np.ndarray, idx: list, k: int) -> float:
    """Signed turn (degrees) of the simplified walk at vertex position k (interior)."""
    a = _unit(pts[idx[k]] - pts[idx[k - 1]])
    b = _unit(pts[idx[k + 1]] - pts[idx[k]])
    return float(np.degrees(np.arctan2(a[0] * b[1] - a[1] * b[0], float(a @ b))))


def _prune_vertices(pts: np.ndarray, s: np.ndarray, idx: list, min_turn_deg: float,
                    min_leg_len: float) -> list:
    """Drop vertices that are not real corners: shallow turns first (noise / gentle
    drift), then stubs shorter than `min_leg_len` (a corner is only kept if BOTH its
    legs are long enough to be walked corridor)."""
    idx = list(idx)
    while len(idx) > 2:
        turns = [abs(_turn_at(pts, idx, k)) for k in range(1, len(idx) - 1)]
        k = int(np.argmin(turns))
        if turns[k] < min_turn_deg:
            idx.pop(k + 1)
            continue
        lens = [float(s[idx[i + 1]] - s[idx[i]]) for i in range(len(idx) - 1)]
        m = int(np.argmin(lens))
        if lens[m] < min_leg_len:
            if m == 0:
                idx.pop(0)
            elif m == len(lens) - 1:
                idx.pop(-1)
            else:                       # keep the sharper of the two bounding corners
                idx.pop(m if abs(_turn_at(pts, idx, m)) <= abs(_turn_at(pts, idx, m + 1))
                        else m + 1)
            continue
        break
    return idx


def door_arclengths(traj_xz, times, door_times) -> list:
    """Door-PASSING times -> arclengths along the walk (recon units).

    `times` are the per-pose timestamps of `traj_xz` (same length, non-decreasing),
    `door_times` the instants a door was passed. A door observed outside the pose time
    span is clamped to the ends, which the caller can spot by comparing to the total
    arclength. This is the only conversion the matcher needs: everything downstream
    speaks arclength."""
    traj = np.asarray(traj_xz, dtype=np.float64).reshape(-1, 2)
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    if len(t) != len(traj):
        raise ValueError(f"times {len(t)} != trajectory {len(traj)}")
    if len(traj) < 2:
        return []
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(traj, axis=0), axis=1))]
    return [float(np.interp(float(td), t, s)) for td in np.asarray(door_times, dtype=np.float64).reshape(-1)]


def recon_events(traj_xz, door_s=None, simplify_tol=None, simplify_frac: float = 0.02,
                 min_turn_deg: float = 35.0, min_leg_len=None,
                 min_leg_frac: float = 0.06) -> list:
    """The WALK as legs / corners / doors, mirroring `plan_skeleton.plan_events()`.

    traj_xz  (N,2) gravity-aligned recon plan points (recon units), time-ordered.
    door_s   arclengths (recon units) at which a door was passed (see
             `door_arclengths` to get them from timestamps).

    Tolerances are RELATIVE (`simplify_frac`, `min_leg_frac` of the total walked
    arclength) because recon units are arbitrary. Returns
        [{"index", "kind": "leg",    "a","b","arclen","chord","heading_deg","s0","s1"},
         {"index", "kind": "corner", "xy","turn_deg","turn_signed_deg","s","legs"},
         {"index", "kind": "door",   "xy","s","leg"}]
    with "index" the position in the returned list (that is what candidate['matches']
    and outliers refer to). An empty/degenerate walk gives []."""
    traj = np.asarray(traj_xz, dtype=np.float64).reshape(-1, 2)
    if len(traj) >= 2:                                   # drop repeated poses
        keep = np.r_[True, np.linalg.norm(np.diff(traj, axis=0), axis=1) > 1e-9]
        traj = traj[keep]
    if len(traj) < 2:
        return []
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(traj, axis=0), axis=1))]
    total = float(s[-1])
    if total <= 1e-9:
        return []
    tol = float(simplify_tol) if simplify_tol is not None else max(simplify_frac * total, 1e-9)
    min_len = float(min_leg_len) if min_leg_len is not None else max(min_leg_frac * total, 1e-9)
    idx = _prune_vertices(traj, s, _dp_keep(traj, tol), min_turn_deg, min_len)

    ev: list = []
    legs: list = []
    for i in range(len(idx) - 1):
        a, b = traj[idx[i]], traj[idx[i + 1]]
        d = b - a
        legs.append({"index": len(ev), "kind": "leg", "a": a, "b": b,
                     "arclen": float(s[idx[i + 1]] - s[idx[i]]),
                     "chord": float(np.linalg.norm(d)), "heading_deg": _ang(d),
                     "s0": float(s[idx[i]]), "s1": float(s[idx[i + 1]]),
                     "dir": _unit(d), "leg": i})
        ev.append(legs[-1])
    for k in range(1, len(idx) - 1):
        t = _turn_at(traj, idx, k)
        ev.append({"index": len(ev), "kind": "corner", "xy": traj[idx[k]],
                   "turn_deg": abs(t), "turn_signed_deg": t, "s": float(s[idx[k]]),
                   "legs": [k - 1, k]})
    for sd in (np.asarray(door_s, dtype=np.float64).reshape(-1) if door_s is not None
               else np.zeros(0)):
        sc = float(np.clip(sd, 0.0, total))
        xy = np.array([float(np.interp(sc, s, traj[:, 0])), float(np.interp(sc, s, traj[:, 1]))])
        leg = next((g["leg"] for g in legs if g["s0"] - 1e-9 <= sc <= g["s1"] + 1e-9), None)
        ev.append({"index": len(ev), "kind": "door", "xy": xy, "s": sc, "leg": leg})
    return ev


def recon_summary(rev: list) -> dict:
    """JSON-safe counts/extents of a recon event list (goes into result['info'])."""
    legs = [e for e in rev if e["kind"] == "leg"]
    return {"n_events": len(rev), "n_legs": len(legs),
            "n_corners": sum(1 for e in rev if e["kind"] == "corner"),
            "n_doors": sum(1 for e in rev if e["kind"] == "door"),
            "leg_arclens": [round(float(g["arclen"]), 4) for g in legs],
            "total_arclen": round(float(sum(g["arclen"] for g in legs)), 4),
            "turns_deg": [round(float(e["turn_signed_deg"]), 1)
                          for e in rev if e["kind"] == "corner"]}


# --------------------------------------------------------------------------------------
# SECTION 2 — prepared arrays for association
# --------------------------------------------------------------------------------------

def _prep_recon(rev: list) -> dict:
    c = [e for e in rev if e["kind"] == "corner"]
    d = [e for e in rev if e["kind"] == "door"]
    g = [e for e in rev if e["kind"] == "leg"]
    blocks = [np.asarray([e["xy"] for e in c], dtype=np.float64).reshape(-1, 2),
              np.asarray([e["xy"] for e in d], dtype=np.float64).reshape(-1, 2),
              np.asarray([e["a"] for e in g], dtype=np.float64).reshape(-1, 2),
              np.asarray([e["b"] for e in g], dtype=np.float64).reshape(-1, 2)]
    return {"corner": c, "door": d, "leg": g, "pts": np.vstack(blocks) if blocks else np.zeros((0, 2)),
            "n": (len(c), len(d), len(g)),
            "corner_turn": np.asarray([e["turn_deg"] for e in c], dtype=np.float64),
            "leg_arclen": np.asarray([e["arclen"] for e in g], dtype=np.float64),
            "n_events": len(rev)}


def _prep_plan(pev: list) -> dict:
    c = [e for e in pev if e["kind"] == "corner"]
    d = [e for e in pev if e["kind"] == "door"]
    g = [e for e in pev if e["kind"] == "leg"]
    ga = np.asarray([e["a"] for e in g], dtype=np.float64).reshape(-1, 2)
    gb = np.asarray([e["b"] for e in g], dtype=np.float64).reshape(-1, 2)
    gd = gb - ga
    return {"corner": c, "door": d, "leg": g,
            "corner_xy": np.asarray([e["xy"] for e in c], dtype=np.float64).reshape(-1, 2),
            "corner_turn": np.asarray([np.nan if e.get("turn_deg") is None else e["turn_deg"]
                                       for e in c], dtype=np.float64),
            # the door's CENTRELINE passing point, never the door leaf's own position —
            # see plan_skeleton.plan_events, where the W/2 bias this avoids is measured.
            "door_xy": np.asarray([e.get("xy_pass", e["xy"]) for e in d],
                                  dtype=np.float64).reshape(-1, 2),
            "leg_a": ga, "leg_b": gb,
            "leg_dir": gd / (np.linalg.norm(gd, axis=1, keepdims=True) + 1e-12),
            "leg_len": np.asarray([e["length"] for e in g], dtype=np.float64),
            "width_med": float(np.median([e["width"] for e in g])) if g else 0.0}


def _tolerances(width_med: float, tol=None) -> dict:
    out = dict(TOL_FLOOR)
    for k, f in TOL_PER_WIDTH.items():
        out[k] = max(TOL_FLOOR[k], f * float(width_med))
    out.update({k: float(v) for k, v in (tol or {}).items()})
    return out


# --------------------------------------------------------------------------------------
# SECTION 3 — association / scoring of ONE transform (the trimmed step)
# --------------------------------------------------------------------------------------

def _associate(tf: dict, R: dict, P: dict, tol: dict, gates: dict, weights: dict) -> dict:
    """Greedy one-to-one association of recon events to plan events under `tf`.

    Returns {"score","matches","outliers","n_inliers","residual","corr" (median-
    translation correction vectors), "inlier_kinds"}. Unmatched events are reported
    with the nearest same-kind plan landmark (if one was even close) and a reason —
    that list IS the field-change report the team is after."""
    nc, nd, ng = R["n"]
    pts = apply_candidate_transform(tf, R["pts"]) if len(R["pts"]) else R["pts"]
    c_t, d_t = pts[:nc], pts[nc:nc + nd]
    a_t, b_t = pts[nc + nd:nc + nd + ng], pts[nc + nd + ng:]
    m_t = 0.5 * (a_t + b_t)

    pairs = []          # (residual, recon_index, plan_id, kind, correction vectors)
    near = {}           # recon_index -> (best residual, plan_id, reason if rejected)

    # ---- corners -------------------------------------------------------------------
    if nc and len(P["corner_xy"]):
        D = np.linalg.norm(c_t[:, None, :] - P["corner_xy"][None, :, :], axis=2)
        dturn = np.abs(R["corner_turn"][:, None] - P["corner_turn"][None, :])
        dturn = np.where(np.isnan(dturn), 0.0, dturn)          # plan turn unknown -> no veto
        for i, e in enumerate(R["corner"]):
            j = int(np.argmin(D[i]))
            ok = (D[i] <= tol["corner_pos"]) & (dturn[i] <= tol["turn_deg"])
            if ok.any():
                for jj in np.nonzero(ok)[0]:
                    pairs.append((float(D[i, jj]), e["index"], P["corner"][int(jj)]["id"],
                                  "corner", [P["corner_xy"][int(jj)] - c_t[i]]))
            reason = ("turn_mismatch" if D[i, j] <= tol["corner_pos"]
                      else "no_plan_corner_within_tol")
            resid = (float(dturn[i, j]) if reason == "turn_mismatch" else float(D[i, j]))
            near[e["index"]] = (resid, P["corner"][j]["id"] if D[i, j] <= 3.0 * tol["corner_pos"]
                                else None, reason)
    else:
        for e in R["corner"]:
            near[e["index"]] = (float("inf"), None, "no_plan_corner")

    # ---- doors ---------------------------------------------------------------------
    if nd and len(P["door_xy"]):
        D = np.linalg.norm(d_t[:, None, :] - P["door_xy"][None, :, :], axis=2)
        for i, e in enumerate(R["door"]):
            j = int(np.argmin(D[i]))
            for jj in np.nonzero(D[i] <= tol["door_pos"])[0]:
                pairs.append((float(D[i, jj]), e["index"], P["door"][int(jj)]["id"], "door",
                              [P["door_xy"][int(jj)] - d_t[i]]))
            near[e["index"]] = (float(D[i, j]),
                                P["door"][j]["id"] if D[i, j] <= 3.0 * tol["door_pos"] else None,
                                "no_plan_door_within_tol")
    else:
        for e in R["door"]:
            near[e["index"]] = (float("inf"), None, "no_plan_door")

    # ---- legs ----------------------------------------------------------------------
    if ng and len(P["leg_a"]):
        latA, ovA, pA = _pt_line(a_t, P["leg_a"], P["leg_dir"], P["leg_len"])
        latB, ovB, pB = _pt_line(b_t, P["leg_a"], P["leg_dir"], P["leg_len"])
        latM = _pt_line(m_t, P["leg_a"], P["leg_dir"], P["leg_len"])[0]
        over = OVER_WEIGHT * np.maximum(ovA, ovB)
        lat = np.maximum(np.maximum(latA, latB), latM)
        # `gate` decides WHETHER the walk leg can be that plan leg (over-run included:
        # walking past the end means walking through a wall); `lat` is what the match
        # COSTS once accepted. Keeping the end out of the cost is the whole point — see
        # OVER_WEIGHT. The outlier report still quotes `gate`, so a leg rejected for
        # over-running says so with the over-run in its number.
        gate_leg = np.maximum(lat, over)
        dir_t = (b_t - a_t)
        dir_t = dir_t / (np.linalg.norm(dir_t, axis=1, keepdims=True) + 1e-12)
        cosang = np.abs(dir_t @ P["leg_dir"].T)
        aligned = cosang >= np.cos(np.deg2rad(tol["leg_ang_deg"]))
        for i, e in enumerate(R["leg"]):
            j = int(np.argmin(gate_leg[i]))
            ok = aligned[i] & (gate_leg[i] <= tol["leg_lat"])
            for jj in np.nonzero(ok)[0]:
                pairs.append((float(lat[i, jj]), e["index"], P["leg"][int(jj)]["id"], "leg",
                              [pA[i, int(jj)] - a_t[i], pB[i, int(jj)] - b_t[i]]))
            if aligned[i].any():
                ja = int(np.asarray(np.nonzero(aligned[i])[0])[np.argmin(gate_leg[i][aligned[i]])])
                reason = ("no_plan_leg_within_tol" if gate_leg[i, ja] > tol["leg_lat"]
                          else "plan_leg_taken")
                near[e["index"]] = (float(gate_leg[i, ja]),
                                    P["leg"][ja]["id"] if gate_leg[i, ja] <= 3.0 * tol["leg_lat"] else None,
                                    reason)
            else:
                near[e["index"]] = (float(gate_leg[i, j]), None, "leg_heading_mismatch")
    else:
        for e in R["leg"]:
            near[e["index"]] = (float("inf"), None, "no_plan_leg")

    # ---- greedy one-to-one ---------------------------------------------------------
    pairs.sort(key=lambda p: (p[0], p[1], p[2]))
    used_r, used_p = set(), set()
    matches, corr, resids, kinds = [], [], [], []
    for res, ri, pid, kind, vecs in pairs:
        if ri in used_r or pid in used_p:
            continue
        used_r.add(ri)
        used_p.add(pid)
        matches.append((ri, pid))
        corr.extend(vecs)
        resids.append(res)
        kinds.append(kind)

    kind_of = {e["index"]: e["kind"] for e in (R["corner"] + R["door"] + R["leg"])}
    outliers = []
    for ri in sorted(kind_of):
        if ri in used_r:
            continue
        res, pid, reason = near.get(ri, (float("inf"), None, "unmatched"))
        outliers.append({"kind": kind_of[ri], "event_index": int(ri), "plan_event_id": pid,
                         "residual": round(float(res if np.isfinite(res) else -1.0), 4),
                         "reason": reason})

    w_all = sum(weights.get(k, 1.0) for k in kind_of.values()) or 1.0
    w_in = sum(weights.get(k, 1.0) for k in kinds)
    # MEAN, not median, over the inliers — see RESID_AGG.
    resid_agg = float(np.mean(resids)) if resids else float("inf")
    # coverage: walked arclength explained by inlier legs / total walked arclength
    leg_index = {e["index"]: i for i, e in enumerate(R["leg"])}
    tot_arc = float(R["leg_arclen"].sum())
    arc_in = float(sum(R["leg_arclen"][leg_index[ri]] for ri, _ in matches if ri in leg_index))
    coverage = arc_in / tot_arc if tot_arc > 1e-9 else 0.0
    frac = w_in / w_all
    resid_term = 0.0 if not resids else max(0.0, 1.0 - resid_agg / max(gates["max_residual"], 1e-9))
    score = frac * (Q_BASE + W_RESID * resid_term * frac + W_COV * coverage)
    return {"score": float(score), "matches": matches, "outliers": outliers,
            "n_inliers": len(matches), "residual": (resid_agg if resids else float("inf")),
            "corr": corr, "kinds": kinds, "coverage": float(coverage)}


# --------------------------------------------------------------------------------------
# SECTION 4 — hypotheses (one corner correspondence each) and refinement
# --------------------------------------------------------------------------------------

def _plan_approaches(P: dict, min_avail: float) -> list:
    """Every way the walk could traverse a plan corner: (in-leg, out-leg) ordered pair
    x which side of each leg is walked. Directions are DETERMINED by geometry (the
    incoming direction must point at the node, the outgoing away from it); the
    available lengths are the distances from the node to the legs' far ends, i.e. the
    most the walk could have covered."""
    out = []
    by_id = {e["id"]: i for i, e in enumerate(P["leg"])}
    for ci, ev in enumerate(P["corner"]):
        xy = P["corner_xy"][ci]
        ids = [f"leg:{i}" for i in ev.get("legs", [])]
        ids = [i for i in ids if i in by_id]
        for a in ids:
            for b in ids:
                if a == b:
                    continue
                ia, ib = by_id[a], by_id[b]
                for din, lin in _sides(P, ia, xy, min_avail):
                    for dout, lout in _sides(P, ib, xy, min_avail):
                        turn = _wrap180(_ang(dout) - _ang(din))
                        out.append({"xy": xy, "in_dir": din, "out_dir": dout,
                                    "in_avail": lin, "out_avail": lout,
                                    "in_leg": a, "out_leg": b, "turn_deg": abs(turn),
                                    "turn_signed_deg": turn, "corner_id": ev["id"]})
    return out


def _sides(P: dict, i: int, xy: np.ndarray, min_avail: float) -> list:
    """(direction of travel, available length) for arriving at / leaving from `xy` along
    plan leg i, from either side; sides shorter than min_avail are dropped."""
    a, b = P["leg_a"][i], P["leg_b"][i]
    u = P["leg_dir"][i]
    s = float((xy - a) @ u)
    opts = []
    if s >= min_avail:                                  # walk a -> xy (direction +u)
        opts.append((u.copy(), s))
    if P["leg_len"][i] - s >= min_avail:                # walk b -> xy (direction -u)
        opts.append((-u, float(P["leg_len"][i] - s)))
    return opts


def _yaw_for(theta_deg: float, s_f: float, s_h: float, in_dir_flip: np.ndarray,
             plan_in_dir: np.ndarray) -> float:
    """The rotation that carries the SCALED incoming walk direction onto the plan
    incoming direction. Computed through the actual tensor, so it stays exact even when
    the walk's corner is not exactly 90 deg — which is also why it DEPENDS ON THE SCALES
    (an anisotropic tensor turns a direction that is not one of its eigenvectors)."""
    A = anisotropic_scale_tensor(np.deg2rad(theta_deg), s_f, s_h)
    return _wrap180(_ang(plan_in_dir) - _ang(A @ in_dir_flip))


def _build(chi: int, theta_deg: float, s_f: float, s_h: float, in_dir_flip: np.ndarray,
           plan_in_dir: np.ndarray, q_anchor: np.ndarray, p_anchor: np.ndarray,
           dt: np.ndarray) -> dict:
    """Assemble a schema transform from the hypothesis parameters.

    yaw comes from `_yaw_for`, and the translation anchors the walk's corner onto the
    plan's corner, plus the trimmed correction `dt`."""
    yaw = _yaw_for(theta_deg, s_f, s_h, in_dir_flip, plan_in_dir)
    tf0 = make_transform(yaw, chi, s_h, s_f, theta_deg, [0.0, 0.0])
    t = np.asarray(p_anchor, dtype=np.float64) - apply_candidate_transform(tf0, [q_anchor])[0]
    return make_transform(yaw, chi, s_h, s_f, theta_deg, t + np.asarray(dt, dtype=np.float64))


def _scales_from_pair(theta_deg: float, in_dir_flip: np.ndarray, plan_in_dir: np.ndarray,
                      dq: np.ndarray, dp: np.ndarray, s_f0: float, s_h0: float,
                      min_proj: float = PAIR_MIN_PROJ, iters: int = PAIR_YAW_ITERS):
    """(s_f, s_h) that carry the recon corner-to-corner vector `dq` (ALREADY chi-flipped)
    onto the plan corner-to-corner vector `dp`, under the same yaw rule `_build` uses.
    Returns None if the pair measures nothing usable.

    WHY THIS IS SOLVABLE AT ALL. With chi and theta fixed the placement is
    p = s_f (q'.e_f0) e_f + s_h (q'.e_h0) e_h + t, with e_f0/e_h0 the tensor's own axes
    (rot(theta)) and e_f/e_h the same pair rotated by yaw. Differencing two
    correspondences kills t, leaving two DECOUPLED scalar equations — one per axis — so a
    second corner fixes both scales exactly, and the anchor in `_build` then fixes the
    translation. That is the whole point: a one-corner seed can only bound the scale by
    the plan leg's remaining LENGTH (`_plan_approaches`' in_avail/out_avail), which is an
    over-estimate under partial traversal and simply wrong once the leg's end has been
    edited, and no local search over scale alone can repair it because moving the scale
    also moves the placement (see `_refine`'s MEASURED LIMIT).

    `s_f0`/`s_h0` are the one-corner seed's values and are KEPT for whichever axis this
    pair does not observe (see PAIR_MIN_PROJ). yaw is iterated to a fixed point because
    it is itself a function of the scales."""
    dq = np.asarray(dq, dtype=np.float64).reshape(2)
    dp = np.asarray(dp, dtype=np.float64).reshape(2)
    nq = float(np.linalg.norm(dq))
    if nq <= 1e-9 or float(np.linalg.norm(dp)) <= 1e-9:
        return None
    V = _rot(theta_deg)
    e_f0, e_h0 = V[:, 0], V[:, 1]
    a, b = float(dq @ e_f0), float(dq @ e_h0)
    obs_f, obs_h = abs(a) >= min_proj * nq, abs(b) >= min_proj * nq
    if not (obs_f or obs_h):                       # unreachable while min_proj < 1/sqrt(2)
        return None
    s_f, s_h = float(s_f0), float(s_h0)
    for _ in range(max(1, int(iters))):
        Ry = _rot(_yaw_for(theta_deg, s_f, s_h, in_dir_flip, plan_in_dir))
        nf = (float(dp @ (Ry @ e_f0)) / a) if obs_f else s_f
        nh = (float(dp @ (Ry @ e_h0)) / b) if obs_h else s_h
        if not (nf > 1e-9 and nh > 1e-9):
            return None
        done = abs(nf - s_f) <= 1e-9 * nf and abs(nh - s_h) <= 1e-9 * nh
        s_f, s_h = nf, nh
        if done:
            break
    return s_f, s_h


def _hypotheses(R: dict, P: dict, tol: dict, aniso_band, pair_seeds: bool = True) -> list:
    """Seed states: (recon corner) x (plan corner traversal) x chi, prefiltered only by
    turn magnitude and by an impossible scale TENSOR (aniso_band). `scale_band` is
    deliberately NOT applied here: a candidate that exists but sits outside the physical
    scale band must be REPORTED and rejected (REJECT 'scale_out_of_band'), not silently
    never generated.

    `theta` is ALWAYS the heading of walk leg 0 in the chi-flipped frame (so s_f is the
    leg-A forward scale build_coplay's place_rigid means), with the tensor's two
    eigenvalues assigned by which walk axis leg 0 lies on.

    `pair_seeds` adds, for every one-corner seed, the variants whose SCALES come from a
    SECOND corner correspondence (`_scales_from_pair`) instead of from the plan legs'
    remaining length. The anchor, chi and theta are the one-corner seed's — only the
    scales change — so this widens the seed set along the exact axis the one-corner seed
    is blind on, and never replaces it (the plain seed is always emitted too, and comes
    first so it wins ties)."""
    if not R["leg"]:
        return []
    d0 = R["leg"][0]["dir"]
    seeds = []
    for rc in R["corner"]:
        g_in, g_out = R["leg"][rc["legs"][0]], R["leg"][rc["legs"][1]]
        if g_in["arclen"] <= 1e-9 or g_out["arclen"] <= 1e-9:
            continue
        for pc in P["approaches"]:
            if abs(rc["turn_deg"] - pc["turn_deg"]) > tol["turn_deg"]:
                continue
            s_in = pc["in_avail"] / g_in["arclen"]
            s_out = pc["out_avail"] / g_out["arclen"]
            for chi in (1, -1):
                F = np.array([float(chi), 1.0])
                u_f, w_f, d0_f = _unit(g_in["dir"] * F), _unit(g_out["dir"] * F), _unit(d0 * F)
                theta = _ang(d0_f)
                if abs(float(u_f @ d0_f)) >= abs(float(w_f @ d0_f)):
                    s_f, s_h = s_in, s_out            # walk leg 0 lies along the in-leg
                else:
                    s_f, s_h = s_out, s_in
                if not (s_f > 1e-9 and s_h > 1e-9):
                    continue
                base = {"chi": chi, "theta": theta, "s_f": s_f, "s_h": s_h,
                        "u_f": u_f, "plan_in_dir": pc["in_dir"],
                        "q": np.asarray(rc["xy"], dtype=np.float64),
                        "p": np.asarray(pc["xy"], dtype=np.float64),
                        "dt": np.zeros(2),
                        "method": f"corner+trimmed(r{rc['index']}->{pc['corner_id']}"
                                  f",{pc['in_leg']}->{pc['out_leg']},chi{chi:+d})"}
                if aniso_band[0] <= s_f / s_h <= aniso_band[1]:
                    seeds.append(base)
                if pair_seeds:
                    seeds.extend(_pair_seeds(base, rc, pc, R, P, tol, aniso_band, F))
    return seeds


def _pair_seeds(base: dict, rc: dict, pc: dict, R: dict, P: dict, tol: dict, aniso_band,
                F: np.ndarray) -> list:
    """The second-corner variants of one seed (see `_hypotheses` / `_scales_from_pair`).

    The second correspondence is subject to the SAME turn-magnitude prefilter as the
    anchor one, must use a different recon corner AND a different plan corner, and the
    resulting tensor must stay inside `aniso_band` — a pair implies a scale, it does not
    get to imply an impossible one. Variants that reproduce the anchor seed's own scales
    are dropped as duplicates rather than scored twice."""
    out = []
    for rc2 in R["corner"]:
        if rc2["index"] == rc["index"]:
            continue
        dq = (np.asarray(rc2["xy"], dtype=np.float64) - base["q"]) * F
        for j, ev2 in enumerate(P["corner"]):
            if ev2["id"] == pc["corner_id"]:
                continue
            t2 = P["corner_turn"][j]
            if np.isfinite(t2) and abs(rc2["turn_deg"] - float(t2)) > tol["turn_deg"]:
                continue
            got = _scales_from_pair(base["theta"], base["u_f"], base["plan_in_dir"], dq,
                                    P["corner_xy"][j] - base["p"], base["s_f"], base["s_h"])
            if got is None:
                continue
            s_f, s_h = got
            if not (aniso_band[0] <= s_f / s_h <= aniso_band[1]):
                continue
            if (abs(s_f - base["s_f"]) <= 1e-6 * base["s_f"]
                    and abs(s_h - base["s_h"]) <= 1e-6 * base["s_h"]):
                continue
            out.append({**base, "s_f": s_f, "s_h": s_h,
                        "method": base["method"][:-1] + f",r{rc2['index']}->{ev2['id']})"})
    return out


def _refine(seed: dict, R: dict, P: dict, tol: dict, gates: dict, weights: dict,
            iters: int, factors) -> tuple:
    """Coordinate-descent trimmed refinement: scale factors then a median translation
    correction over the current inliers. Only STRICT improvements are taken and factor
    1.0 comes first, so an exact seed is never perturbed.

    MEASURED LIMIT, recorded because it is why `_pair_seeds` exists: this descent cannot
    reach a better answer that is separated from the seed by a JOINT move in (scale,
    translation). The last confident-wrong answer in the cycle-15 sweep was exactly that
    — the true placement scored 0.622 against the winner's 0.597 on that perturbed plan
    and was never generated, because the plan leg it needs is truncated, so at the true
    scale the leg failed the over-run veto before the translation could catch up. Letting
    the trimmed translation follow each scale trial IN THE SAME STEP was tried and
    measured: it does NOT reach it either (winner unchanged at 0.597). The fix was a
    better SEED, not a better local search — a second corner correspondence
    (`_scales_from_pair`) fixes scale and position at once, and on that same case it
    generates a 0.639 candidate 0.37 m from the truth. Do not re-attempt this by widening
    SCALE_FACTORS or `iters`."""
    st = {"s_f": seed["s_f"], "s_h": seed["s_h"], "dt": np.asarray(seed["dt"], dtype=np.float64)}

    def ev(state):
        tf = _build(seed["chi"], seed["theta"], state["s_f"], state["s_h"], seed["u_f"],
                    seed["plan_in_dir"], seed["q"], seed["p"], state["dt"])
        return tf, _associate(tf, R, P, tol, gates, weights)

    tf, ass = ev(st)
    for _ in range(max(0, int(iters))):
        improved = False
        for key in ("s_f", "s_h"):
            for f in factors:
                if abs(f - 1.0) < 1e-12:
                    continue
                trial = {**st, key: st[key] * float(f)}
                tf2, ass2 = ev(trial)
                if ass2["score"] > ass["score"] + 1e-9:
                    st, tf, ass, improved = trial, tf2, ass2, True
        if ass["corr"]:
            dt = st["dt"] + np.median(np.asarray(ass["corr"], dtype=np.float64).reshape(-1, 2), axis=0)
            tf2, ass2 = ev({**st, "dt": dt})
            if ass2["score"] > ass["score"] + 1e-9:
                st, tf, ass, improved = {**st, "dt": dt}, tf2, ass2, True
        if not improved:
            break
    return tf, ass


def _same_transform(a: dict, b: dict, yaw_tol: float = DEDUPE_YAW_DEG,
                    t_tol: float = DEDUPE_T_FLOOR, s_rel: float = DEDUPE_SCALE_REL) -> bool:
    if int(a["chi"]) != int(b["chi"]):
        return False
    if abs(_wrap180(a["yaw_deg"] - b["yaw_deg"])) > yaw_tol:
        return False
    if float(np.linalg.norm(np.asarray(a["translation"]) - np.asarray(b["translation"]))) > t_tol:
        return False
    for k in ("s_f", "s_h"):
        if abs(a[k] - b[k]) > s_rel * max(abs(a[k]), abs(b[k])):
            return False
    return True


# --------------------------------------------------------------------------------------
# SECTION 5 — the entry point
# --------------------------------------------------------------------------------------

def coarse_match(skel: dict, traj_xz, door_s=None, gates=None, tol=None, weights=None,
                 scale_band=DEFAULT_SCALE_BAND, aniso_band=DEFAULT_ANISO_BAND,
                 s_h_prior=None, s_h_rel_tol: float = 0.25, max_candidates: int = 8,
                 refine_iters: int = 2, min_inliers: int = 2, min_avail: float = 1.0,
                 scale_factors=SCALE_FACTORS, recon_kwargs=None,
                 pair_seeds: bool = True) -> dict:
    """Match a recon walk against a plan skeleton; return a `coplay.plan_match/1.0`
    result (see `plan_skeleton.RESULT_FIELDS`).

    skel        `plan_skeleton.corridor_skeleton()` / `plan_skeleton()` output.
    traj_xz     (N,2) gravity-aligned recon plan points, time-ordered, RECON units.
    door_s      arclengths (recon units) of door-passing events (optional).
    gates       overrides `plan_skeleton.DEFAULT_GATES` (margin_min, min_inlier_ratio,
                max_residual) — reported in result['gates'].
    tol         overrides the association tolerances (see TOL_FLOOR/TOL_PER_WIDTH).
    scale_band / aniso_band  physical bands on (s_f, s_h) metres-per-recon-unit and on
                s_f/s_h. Candidates outside are dropped; if that leaves NOTHING while
                candidates existed -> REJECT 'scale_out_of_band' (no forced fit).
    s_h_prior   an INDEPENDENT lateral scale (e.g. the corridor-width wall anchor).
                Candidates whose s_h differs by more than `s_h_rel_tol` are dropped the
                same way. This is the evidence-based way out of the 2-leg L ambiguity —
                it adds a measurement, it does not relax the margin.
    pair_seeds  also seed the scales from a SECOND corner correspondence
                (`_scales_from_pair`) instead of only from the plan legs' remaining
                length. Exposed so the two seed sets can be MEASURED against each other;
                the default is on.

    Exits: HOLD 'no_plan_skeleton' (no plan legs), HOLD 'insufficient_events' (the walk
    has no corner, or the plan has none — a straight walk cannot fix s_h/position along
    the corridor), HOLD 'no_candidate', REJECT via the gates, else ok."""
    g = {**DEFAULT_GATES, **(gates or {})}
    w = {**EVENT_WEIGHTS, **(weights or {})}
    info: dict = {"matcher": "coarse_match/1.0"}
    pev = plan_events(skel or {})
    P = _prep_plan(pev)
    info["plan"] = {"n_legs": len(P["leg"]), "n_corners": len(P["corner"]),
                    "n_doors": len(P["door"]), "width_med": round(P["width_med"], 3)}
    info.update({k: v for k, v in (skel or {}).get("info", {}).items()
                 if k in ("theta0_deg", "median_width", "fail")})
    if not P["leg"]:
        return make_hold_result("no_plan_skeleton", [], info, g)

    rev = recon_events(traj_xz, door_s=door_s, **(recon_kwargs or {}))
    R = _prep_recon(rev)
    info["recon"] = recon_summary(rev)
    t = _tolerances(P["width_med"], tol)
    info["tol"] = {k: round(float(v), 4) for k, v in t.items()}
    if not R["corner"] or not P["corner"]:
        info["fail_seed"] = ("walk has no turn" if not R["corner"] else "plan has no corner node")
        return make_hold_result("insufficient_events", [], info, g)

    P["approaches"] = _plan_approaches(P, min_avail)
    seeds = _hypotheses(R, P, t, aniso_band, pair_seeds=bool(pair_seeds))
    info["n_approaches"] = len(P["approaches"])
    info["n_hypotheses"] = len(seeds)
    info["n_pair_seeds"] = sum(1 for s in seeds if s["method"].count("->") > 2)
    if not seeds:
        return make_hold_result("no_candidate", [], info, g)

    scored = []
    for sd in seeds:
        tf, ass = _refine(sd, R, P, t, g, w, refine_iters, scale_factors)
        if ass["n_inliers"] < int(min_inliers) or not np.isfinite(ass["residual"]):
            continue
        scored.append((ass["score"], tf, ass, sd["method"]))
    scored.sort(key=lambda x: (-x[0], x[3]))
    info["n_scored"] = len(scored)
    if not scored:
        info["fail_seed"] = f"no hypothesis reached {min_inliers} inliers"
        return make_hold_result("no_candidate", [], info, g)

    kept = []
    t_tol = max(DEDUPE_T_FLOOR, 0.5 * P["width_med"])
    for sc, tf, ass, method in scored:
        if any(_same_transform(tf, k[1], t_tol=t_tol) for k in kept):
            continue
        kept.append((sc, tf, ass, method))
        if len(kept) >= int(max_candidates):
            break
    info["n_deduped"] = len(kept)

    def _cand(sc, tf, ass, method):
        return make_candidate(tf, score=sc, n_inliers=ass["n_inliers"], n_events=R["n_events"],
                              residual=ass["residual"], matches=ass["matches"],
                              outliers=ass["outliers"], method=method)

    in_band, out_band = [], []
    for sc, tf, ass, method in kept:
        s_f, s_h = float(tf["s_f"]), float(tf["s_h"])
        bad = (not (scale_band[0] <= s_f <= scale_band[1])
               or not (scale_band[0] <= s_h <= scale_band[1])
               or not (aniso_band[0] <= s_f / max(s_h, 1e-9) <= aniso_band[1]))
        if s_h_prior is not None and abs(s_h - float(s_h_prior)) > s_h_rel_tol * float(s_h_prior):
            bad = True
        (out_band if bad else in_band).append(_cand(sc, tf, ass, method))
    info["n_out_of_band"] = len(out_band)
    if s_h_prior is not None:
        info["s_h_prior"] = [round(float(s_h_prior), 5), round(float(s_h_rel_tol), 4)]
    info["scale_band"] = [float(scale_band[0]), float(scale_band[1])]
    info["aniso_band"] = [float(aniso_band[0]), float(aniso_band[1])]
    if not in_band:
        return make_reject_result("scale_out_of_band", out_band, info, g)
    return finalize_match(in_band, gates=g, info=info)

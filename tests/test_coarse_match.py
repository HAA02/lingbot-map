"""Tests for scan2bim.coarse_match: the OUTLIER-TOLERANT partial matcher that places a
recon walk on a `plan_skeleton` and returns ranked `coplay.plan_match/1.0` candidates.

GROUND TRUTH BY CONSTRUCTION. Every case here builds a synthetic corridor from an
axis-aligned centreline (`_corridor_walls`, the same 1.82 m clear width as the Gasan_7F
corridor and the same fixture style as tests/test_plan_skeleton.py), walks that
centreline, and then maps the walk into "recon units" with the INVERSE of a chosen
schema transform (`_to_recon`). So the answer the matcher must find is known exactly,
and `test_inverse_round_trips` proves the harness itself before anything is asked of
the matcher.

The four things pinned here are the team's contract, not the implementation's habits:

  1. no perturbation      -> status 'ok' and the transform recovered EXACTLY
                             (measured 7e-15 m pointwise), partial traversal included;
  2. perturbed drawing    -> the transform still recovered inside DoD D2 (yaw <= 5 deg,
                             offset <= 0.5 x corridor width, per-axis scale <= 10 %) AND
                             the changed geometry reported in `outliers`, with
                             inlier_ratio still above the gate;
  3. ambiguous geometry   -> status 'hold', hold_reason 'ambiguous_margin', best None,
                             and the tied candidates STILL EXPOSED for adjudication;
  4. nothing that fits    -> 'reject'/'hold' with best None — no fabricated transform.

Every scenario additionally runs `plan_skeleton.validate_match_result`, so the schema
and the "a non-ok result never carries a transform" invariant are machine-checked on
real matcher output rather than on hand-built dicts (test_plan_skeleton.py does those).

NOTE: synthetic plans only — no real SXX/Gasan DXF exists in this repo.
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scan2bim import coarse_match as cm
from scan2bim import plan_skeleton as ps
from scan2bim.forward_scale import anisotropic_scale_tensor
from test_plan_skeleton import CENTER_A, CENTER_B, W, _l_corridor_segments

# --------------------------------------------------------------------------------------
# synthetic plan + walk harness
# --------------------------------------------------------------------------------------

#: The transform the matcher has to find. s_f/s_h are the fwdscale cycle's measured
#: anisotropic pair (3.53 forward, 1.97 lateral metres per recon unit) and chi=-1 is the
#: handedness flip the .dtdx exports need, so this is the real operating point.
TRUE = {"yaw_deg": -37.0, "chi": -1, "s_f": 3.53, "s_h": 1.97, "translation": [4.0, -2.5]}

#: 4-leg staircase corridor (v, h, v, h), 3 corners. Legs 11.09 / 17.18 / 13.00 / 14.91 m.
STAIR = [(0.91, 0.0), (0.91, 11.09), (18.09, 11.09), (18.09, 24.09), (33.0, 24.09)]
#: A-DOOR positions in the plan (on a wall face) and the matching walk-side passing
#: points on the centreline: one door per leg plus a second on the long leg.
STAIR_PLAN_DOORS = [(1.82, 3.0), (5.0, 12.0), (13.0, 10.18), (17.18, 15.0), (25.0, 25.0)]
STAIR_WALK_DOORS = [(0.91, 3.0), (5.0, 11.09), (13.0, 11.09), (18.09, 15.0), (25.0, 24.09)]

#: DoD D2, the accuracy this pipeline claims — used as the acceptance band on a
#: PERTURBED plan (the clean case is asserted far tighter, see TestExactRecovery).
D2_YAW_DEG, D2_SCALE_REL, D2_OFFSET = 5.0, 0.10, 0.5 * W


def _corridor_walls(spine, width: float = W, caps=(True, True)) -> np.ndarray:
    """Wall segments of a corridor of CLEAR width `width` about an axis-aligned
    centreline, as (N,2,2) metres: the two wall polylines offset by +-width/2 with
    mitred 90 deg joins (that miter is what makes the clear width exact at a corner),
    plus optional dead-end caps. Generalises tests/test_plan_skeleton._l_corridor_segments
    to any number of legs."""
    c = np.asarray(spine, dtype=np.float64).reshape(-1, 2)
    w = 0.5 * float(width)
    u = [(b - a) / np.linalg.norm(b - a) for a, b in zip(c, c[1:])]
    n = [np.array([-d[1], d[0]]) for d in u]
    left, right = [], []
    for i in range(len(c)):
        off = n[0] if i == 0 else (n[-1] if i == len(c) - 1 else n[i - 1] + n[i])
        left.append(c[i] + w * off)
        right.append(c[i] - w * off)
    segs = [(p, q) for poly in (left, right) for p, q in zip(poly, poly[1:])]
    if caps[0]:
        segs.append((left[0], right[0]))
    if caps[1]:
        segs.append((left[-1], right[-1]))
    return np.asarray(segs, dtype=np.float64).reshape(-1, 2, 2)


def _densify(poly, step: float = 0.15) -> np.ndarray:
    """Polyline -> evenly sampled points (a walk, not a polyline, is what the matcher
    is handed)."""
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    out = [poly[0][None, :]]
    for a, b in zip(poly, poly[1:]):
        m = max(2, int(np.ceil(float(np.linalg.norm(b - a)) / step)) + 1)
        out.append((a + (b - a) * np.linspace(0.0, 1.0, m)[:, None])[1:])
    return np.vstack(out)


def _true_transform(leg0_dir, **over) -> dict:
    """A schema transform with theta_deg forced to its CONTRACTUAL meaning: the heading
    of walk leg 0 in the chi-flipped recon frame. Since the walk is generated by
    inverting this transform, that pins theta = heading(plan leg 0) - yaw (the chi flip
    cancels), which is exactly what `coarse_match._hypotheses` reconstructs."""
    p = {**TRUE, **over}
    d = np.asarray(leg0_dir, dtype=np.float64).reshape(2)
    theta = float(np.degrees(np.arctan2(d[1], d[0]))) - float(p["yaw_deg"])
    return ps.make_transform(p["yaw_deg"], p["chi"], p["s_h"], p["s_f"], theta,
                             p["translation"])


def _to_recon(tf: dict, plan_pts) -> np.ndarray:
    """Inverse of `plan_skeleton.apply_candidate_transform` (plan metres -> recon units).
    Pinned against the forward map in `test_inverse_round_trips`."""
    p = np.asarray(plan_pts, dtype=np.float64).reshape(-1, 2)
    A = anisotropic_scale_tensor(np.deg2rad(float(tf["theta_deg"])),
                                 float(tf["s_f"]), float(tf["s_h"]))
    R = ps._rot(float(tf["yaw_deg"]))
    q = ((p - np.asarray(tf["translation"], dtype=np.float64)) @ R) @ np.linalg.inv(A)
    q[:, 0] *= float(tf["chi"])
    return q


def _make_walk(spine, doors=(), step: float = 0.15, **over):
    """(true transform, recon trajectory, plan-frame walk, door arclengths in recon
    units). Door arclengths go through the public `door_arclengths` with the pose index
    as the timestamp, so that conversion is exercised on every scenario."""
    spine = np.asarray(spine, dtype=np.float64).reshape(-1, 2)
    tf = _true_transform(spine[1] - spine[0], **over)
    plan_walk = _densify(spine, step)
    traj = _to_recon(tf, plan_walk)
    t = np.arange(len(traj), dtype=np.float64)
    hits = [float(np.argmin(np.linalg.norm(plan_walk - np.asarray(d, dtype=np.float64),
                                           axis=1))) for d in doors]
    return tf, traj, plan_walk, cm.door_arclengths(traj, t, hits)


def _stair_skeleton(spine=STAIR, plan_doors=STAIR_PLAN_DOORS, wall_shift=None) -> dict:
    """The staircase plan skeleton, optionally with a wall pair MOVED: `wall_shift` is
    (segment indices, dx, dy) applied to `_corridor_walls(spine)`."""
    segs = _corridor_walls(spine)
    if wall_shift is not None:
        idx, dx, dy = wall_shift
        segs = segs.copy()
        segs[list(idx)] += np.array([dx, dy], dtype=np.float64)
    return ps.corridor_skeleton(segs, doors=list(plan_doors))


class _MatchAssertions(unittest.TestCase):
    """Shared contract assertions. Every scenario runs `assert_contract`."""

    def assert_contract(self, res: dict, skel: dict, traj, door_s=None):
        """Schema + HOLD invariant + referential integrity of matches/outliers."""
        ok, errs = ps.validate_match_result(res)
        self.assertTrue(ok, errs)
        self.assertEqual(res["schema"], ps.PLAN_MATCH_SCHEMA)
        self.assertIn(res["status"], ps.STATUSES)
        self.assertEqual(res["ok"], res["status"] == "ok")
        if res["status"] != "ok":                      # no transform ever leaks on a refusal
            self.assertIsNone(res["best"])
            self.assertIn(res["hold_reason"], ps.HOLD_REASONS + ps.REJECT_REASONS)
        json.dumps(res)                                # the wire format must survive JSON
        n_ev = len(cm.recon_events(traj, door_s=door_s))
        plan_ids = {e["id"] for e in ps.plan_events(skel)}
        for c in res["candidates"]:
            self.assertEqual(c["n_events"], n_ev, c["method"])
            seen = [int(a) for a, _ in c["matches"]] + [int(o["event_index"])
                                                        for o in c["outliers"]]
            self.assertEqual(sorted(seen), list(range(n_ev)),
                             f"matches+outliers must partition the recon events: {c}")
            self.assertEqual(len(c["matches"]), c["n_inliers"])
            for _, pid in c["matches"]:
                self.assertIn(pid, plan_ids)
            for o in c["outliers"]:
                self.assertIn(o["kind"], ("leg", "corner", "door"))
                self.assertTrue(o["plan_event_id"] is None or o["plan_event_id"] in plan_ids, o)
                self.assertIsInstance(o["reason"], str)
                self.assertTrue(o["reason"])

    def assert_transform_close(self, got: dict, want: dict, yaw_tol=D2_YAW_DEG,
                               scale_rel=D2_SCALE_REL, offset=D2_OFFSET):
        self.assertEqual(int(got["chi"]), int(want["chi"]), f"handedness: {got}")
        self.assertLessEqual(abs(cm._wrap180(got["yaw_deg"] - want["yaw_deg"])), yaw_tol, got)
        for k in ("s_f", "s_h"):
            self.assertLessEqual(abs(got[k] - want[k]) / want[k], scale_rel, f"{k}: {got}")
        d = float(np.linalg.norm(np.asarray(got["translation"])
                                 - np.asarray(want["translation"])))
        self.assertLessEqual(d, offset, f"translation: {got}")


# --------------------------------------------------------------------------------------
# 0 — the harness itself (so a later failure cannot be blamed on the fixture)
# --------------------------------------------------------------------------------------

class TestHarness(_MatchAssertions):

    def test_inverse_round_trips(self):
        tf = _true_transform((0.0, 1.0))
        p = _densify(STAIR)
        np.testing.assert_allclose(ps.apply_candidate_transform(tf, _to_recon(tf, p)),
                                   p, atol=1e-9)

    def test_synthetic_plan_has_the_intended_skeleton(self):
        skel = _stair_skeleton()
        self.assertEqual(len(skel["legs"]), 4, skel["info"])
        self.assertEqual(skel["info"]["n_corners"], 3, skel["nodes"])
        self.assertEqual(len(skel["doors"]), len(STAIR_PLAN_DOORS), skel["doors"])
        for g in skel["legs"]:
            self.assertAlmostEqual(g["width"], W, delta=0.02, msg=g)
        self.assertEqual(sorted(round(g["length"], 2) for g in skel["legs"]),
                         [11.09, 13.0, 14.91, 17.18])

    def test_L_fixture_reused_from_test_plan_skeleton(self):
        skel = ps.corridor_skeleton(_l_corridor_segments())
        self.assertEqual(len(skel["legs"]), 2, skel["info"])
        self.assertEqual(skel["info"]["n_corners"], 1)


# --------------------------------------------------------------------------------------
# 1 — the recon-side event vocabulary
# --------------------------------------------------------------------------------------

class TestReconEvents(_MatchAssertions):

    def test_walk_reduces_to_legs_corners_doors_with_true_arclengths(self):
        tf, traj, _, ds = _make_walk(STAIR, STAIR_WALK_DOORS)
        rev = cm.recon_events(traj, door_s=ds)
        s = cm.recon_summary(rev)
        self.assertEqual((s["n_legs"], s["n_corners"], s["n_doors"]), (4, 3, 5))
        self.assertEqual(s["n_events"], len(rev))
        # plan leg lengths / the scale that maps that leg's axis back to recon units
        for arc, (L, sc) in zip(s["leg_arclens"], [(11.09, TRUE["s_f"]), (17.18, TRUE["s_h"]),
                                                   (13.0, TRUE["s_f"]), (14.91, TRUE["s_h"])]):
            self.assertAlmostEqual(arc, L / sc, delta=0.01)
        self.assertEqual([round(t) for t in s["turns_deg"]], [90, -90, 90])
        self.assertEqual([e["index"] for e in rev], list(range(len(rev))))

    def test_straight_walk_has_no_corner(self):
        _, traj, _, _ = _make_walk([(0.91, 0.0), (0.91, 11.09)])
        rev = cm.recon_events(traj)
        self.assertEqual(cm.recon_summary(rev)["n_corners"], 0)
        self.assertEqual(cm.recon_summary(rev)["n_legs"], 1)

    def test_degenerate_walks_yield_no_events(self):
        for traj in (np.zeros((0, 2)), [[1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]):
            self.assertEqual(cm.recon_events(traj), [])

    def test_door_arclengths_interpolate_and_clamp(self):
        """Timestamps -> walked arclength. Times are deliberately NOT proportional to
        arclength (the walker pauses), so a wrong axis would show up: poses at t =
        0/2/10 sit at s = 0/1/5."""
        traj = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 4.0]])
        got = cm.door_arclengths(traj, [0.0, 2.0, 10.0], [0.0, 1.0, 6.0, -9.0, 99.0])
        np.testing.assert_allclose(got, [0.0, 0.5, 3.0, 0.0, 5.0], atol=1e-9)

    def test_door_arclengths_refuse_mismatched_times(self):
        with self.assertRaises(ValueError):
            cm.door_arclengths([[0.0, 0.0], [1.0, 0.0]], [0.0], [0.0])


# --------------------------------------------------------------------------------------
# 2 — SCENARIO 1: no perturbation -> ok, transform recovered exactly
# --------------------------------------------------------------------------------------

class TestExactRecovery(_MatchAssertions):
    """Known plan + known transform in -> the same transform back out. The bar is
    EXACT (not D2): with a walk generated from the drawing there is nothing to trade
    off, so any drift here is the matcher inventing one."""

    def setUp(self):
        self.skel = _stair_skeleton()
        self.tf, self.traj, self.plan_walk, self.ds = _make_walk(STAIR, STAIR_WALK_DOORS)
        self.res = cm.coarse_match(self.skel, self.traj, door_s=self.ds)

    def test_status_ok_and_contract_holds(self):
        self.assert_contract(self.res, self.skel, self.traj, self.ds)
        self.assertEqual(self.res["status"], "ok",
                         (self.res["hold_reason"], self.res["margin"], self.res["info"]))
        self.assertIsNone(self.res["hold_reason"])
        self.assertEqual(self.res["best"], self.res["candidates"][0])

    def test_transform_is_recovered_exactly(self):
        self.assert_transform_close(self.res["best"]["transform"], self.tf,
                                    yaw_tol=0.05, scale_rel=2e-3, offset=0.01)

    def test_placement_reproduces_the_walk_on_the_plan(self):
        got = ps.apply_candidate_transform(self.res["best"]["transform"], self.traj)
        self.assertLess(float(np.abs(got - self.plan_walk).max()), 1e-6)

    def test_everything_is_explained(self):
        best = self.res["best"]
        self.assertEqual(best["outliers"], [])
        self.assertEqual(best["inlier_ratio"], 1.0)
        self.assertLessEqual(best["residual"], self.res["gates"]["max_residual"])
        self.assertEqual(best["n_inliers"], best["n_events"])

    def test_the_answer_is_not_a_tie(self):
        self.assertGreaterEqual(self.res["margin"], self.res["gates"]["margin_min"])

    def test_partial_traversal_still_recovers_exactly(self):
        """The walk starts 2.5 m inside the corridor and stops 5 m short of the end:
        partial coverage must cost nothing (point-to-SEGMENT leg residual)."""
        partial = [(0.91, 2.5), (0.91, 11.09), (18.09, 11.09), (18.09, 24.09), (28.0, 24.09)]
        tf, traj, walk, ds = _make_walk(partial, STAIR_WALK_DOORS[:4])
        res = cm.coarse_match(self.skel, traj, door_s=ds)
        self.assert_contract(res, self.skel, traj, ds)
        self.assertEqual(res["status"], "ok", (res["hold_reason"], res["info"]))
        self.assert_transform_close(res["best"]["transform"], tf,
                                    yaw_tol=0.05, scale_rel=2e-3, offset=0.01)
        self.assertEqual(res["best"]["outliers"], [])

    def test_result_is_deterministic(self):
        again = cm.coarse_match(self.skel, self.traj, door_s=self.ds)
        self.assertEqual(json.dumps(again, sort_keys=True),
                         json.dumps(self.res, sort_keys=True))

    def test_gates_are_reported_as_applied(self):
        res = cm.coarse_match(self.skel, self.traj, door_s=self.ds,
                              gates={"min_inlier_ratio": 0.42})
        self.assertEqual(res["gates"]["min_inlier_ratio"], 0.42)
        self.assertEqual(res["gates"]["max_residual"], ps.DEFAULT_GATES["max_residual"])


# --------------------------------------------------------------------------------------
# 2b — P2: what the placement is allowed to be steered BY
# --------------------------------------------------------------------------------------

class TestWhatSteersThePlacement(_MatchAssertions):
    """Three P2 fixes, each pinned by the measurement that motivated it. All three are
    about the same failure: the matcher was being pulled onto landmarks that cannot
    carry the accuracy this pipeline claims, and then confirming the result."""

    def setUp(self):
        self.skel = _stair_skeleton()
        self.tf, self.traj, self.plan_walk, self.ds = _make_walk(STAIR, STAIR_WALK_DOORS)
        self.res = cm.coarse_match(self.skel, self.traj, door_s=self.ds)
        self.step = float(np.linalg.norm(np.diff(self.plan_walk, axis=0), axis=1).max())

    def test_a_door_is_matched_on_the_centreline_not_on_its_leaf(self):
        """A walk observes a door as a PASSING event, on the centreline; the drawing puts
        the leaf on a wall FACE. Comparing those two builds a fixed W/2 = 0.91 m error
        into every CORRECT match — the whole D2 offset budget — so the plan side must
        offer the passing point (plan_events['xy_pass']) and the matcher must use it.
        Both halves are checked here: each matched door is within half a walk sample of
        its passing point AND still a full half-width from its leaf."""
        self.assertEqual(self.res["status"], "ok")
        by_id = {e["id"]: e for e in ps.plan_events(self.skel)}
        rev = cm.recon_events(self.traj, door_s=self.ds)
        doors = [(ri, pid) for ri, pid in self.res["best"]["matches"] if pid.startswith("door")]
        self.assertGreaterEqual(len(doors), 4, self.res["best"]["matches"])
        for ri, pid in doors:
            e = next(x for x in rev if x["index"] == ri)
            p = ps.apply_candidate_transform(self.res["best"]["transform"], [e["xy"]])[0]
            pe = by_id[pid]
            to_pass = float(np.linalg.norm(p - np.asarray(pe["xy_pass"], dtype=np.float64)))
            to_leaf = float(np.linalg.norm(p - np.asarray(pe["xy"], dtype=np.float64)))
            self.assertLess(to_pass, 0.5 * self.step, f"{pid}: {to_pass:.4f} m off its passing point")
            self.assertAlmostEqual(to_leaf, 0.5 * W, delta=0.05,
                                   msg=f"{pid}: the leaf really is half a corridor away")

    def test_the_reported_residual_charges_every_inlier(self):
        """`residual` is the MEAN over inliers (coarse_match.RESID_AGG). The median it
        replaced let HALF the inliers sit anywhere inside the association tolerance for
        free — and those tolerances are wider than the D2 offset, which is the slack a
        slid placement lives in. On the clean plan every inlier really is exact, so the
        two agree here; what this pins is that the number is not a median in disguise."""
        best = self.res["best"]
        self.assertEqual(best["inlier_ratio"], 1.0)
        self.assertLess(best["residual"], 0.5 * self.step)
        self.assertGreater(best["residual"], 0.0)          # ...and it is not rounded away

    def test_a_truncated_plan_leg_no_longer_drags_the_placement(self):
        """One wall segment of leg A deleted, so the plan keeps 3 legs / 2 corners while
        the walk still has 4 legs — the shape of every confident-wrong answer in the
        cycle-12 sweep. The old scoring confirmed a placement 0.746 m off (an exactly
        fitting SLID subset: residual 0.0 on 9 of 12 events); it must now land on the
        truth."""
        segs = _corridor_walls(STAIR)
        skel = ps.corridor_skeleton(segs[[i for i in range(len(segs)) if i != 0]],
                                    doors=list(STAIR_PLAN_DOORS))
        self.assertEqual(len(skel["legs"]), 3)
        res = cm.coarse_match(skel, self.traj, door_s=self.ds)
        self.assert_contract(res, skel, self.traj, self.ds)
        self.assertEqual(res["status"], "ok", (res["hold_reason"], res["margin"]))
        off = float(np.linalg.norm(np.asarray(res["best"]["transform"]["translation"])
                                   - np.asarray(self.tf["translation"])))
        self.assertLess(off, 0.05, f"placement slid {off:.3f} m onto the truncated leg")
        self.assert_transform_close(res["best"]["transform"], self.tf)

    def test_without_door_evidence_the_same_walk_only_holds(self):
        """THE operating condition of the real upload (upload_1781521406685 has no
        physical doors at all). Stripped of doors, the 4-leg staircase is exactly as
        ambiguous as the 2-leg L — corner and leg geometry alone cannot tell the walk
        from its mirror — and the matcher must HOLD rather than pick. Measured across the
        qa perturbation sweep this condition confirms NOTHING and, just as importantly,
        gets NOTHING wrong: 0 confident answers, 0 outside D2."""
        skel = ps.corridor_skeleton(_corridor_walls(STAIR))
        res = cm.coarse_match(skel, self.traj)
        self.assert_contract(res, skel, self.traj)
        self.assertEqual((res["status"], res["hold_reason"]), ("hold", "ambiguous_margin"))
        self.assertIsNone(res["best"])
        self.assertLess(res["margin"], res["gates"]["margin_min"])
        top = res["candidates"][:2]
        self.assertNotEqual(int(top[0]["transform"]["chi"]), int(top[1]["transform"]["chi"]))

    def test_an_independent_lateral_prior_is_what_replaces_the_doors(self):
        """...and the way out is evidence, not a relaxed margin: the corridor-width wall
        anchor is an INDEPENDENT lateral scale, and at +-20 % it resolves the door-free
        staircase to the true placement (margin 0.36). Stated because it is the only
        route the real, door-free upload has."""
        skel = ps.corridor_skeleton(_corridor_walls(STAIR))
        res = cm.coarse_match(skel, self.traj, s_h_prior=TRUE["s_h"], s_h_rel_tol=0.20)
        self.assert_contract(res, skel, self.traj)
        self.assertEqual(res["status"], "ok", (res["hold_reason"], res["margin"]))
        self.assertGreaterEqual(res["margin"], res["gates"]["margin_min"])
        self.assert_transform_close(res["best"]["transform"], self.tf)


# --------------------------------------------------------------------------------------
# 3 — SCENARIO 2: perturbed drawing -> recovered anyway, changes reported as outliers
# --------------------------------------------------------------------------------------

class TestOutlierTolerance(_MatchAssertions):
    """The drawing disagrees with the field: leg C's wall pair MOVED 0.8 m (inside the
    qa perturbation band 0.3-1.0 m) and the door on leg D REMOVED. The walk is still the
    one generated from the TRUE plan."""

    #: indices into `_corridor_walls(STAIR)` of leg C's two walls.
    MOVED_WALLS = ([2, 6], 0.8, 0.0)
    #: plan doors after the edit: the (25.0, 25.0) door on leg D is gone and the leg C
    #: door travels with its wall.
    KEPT_DOORS = [(1.82, 3.0), (5.0, 12.0), (13.0, 10.18), (17.98, 15.0)]
    #: everything at x >= this in the plan frame is downstream of the moved wall.
    CHANGED_FROM_X = 15.0

    def setUp(self):
        self.skel = _stair_skeleton(plan_doors=self.KEPT_DOORS, wall_shift=self.MOVED_WALLS)
        self.tf, self.traj, self.plan_walk, self.ds = _make_walk(STAIR, STAIR_WALK_DOORS)
        self.res = cm.coarse_match(self.skel, self.traj, door_s=self.ds)
        self.rev = cm.recon_events(self.traj, door_s=self.ds)

    def _event_plan_xy(self, i: int) -> np.ndarray:
        """Where recon event `i` really is, in plan metres (via the TRUE transform)."""
        e = self.rev[i]
        q = 0.5 * (np.asarray(e["a"]) + np.asarray(e["b"])) if e["kind"] == "leg" else e["xy"]
        return ps.apply_candidate_transform(self.tf, [q])[0]

    def test_perturbed_plan_is_actually_perturbed(self):
        clean = _stair_skeleton()
        self.assertEqual(len(self.skel["legs"]), len(clean["legs"]))
        moved = [g for g in self.skel["legs"]
                 if abs(g["dir"][1]) > 0.9 and g["a"][0] > 10.0]
        self.assertEqual(len(moved), 1, self.skel["legs"])
        self.assertAlmostEqual(moved[0]["a"][0], 18.09 + 0.8, delta=0.02)
        self.assertEqual(len(self.skel["doors"]), 4, self.skel["doors"])

    def test_status_ok_and_contract_holds(self):
        self.assert_contract(self.res, self.skel, self.traj, self.ds)
        self.assertEqual(self.res["status"], "ok",
                         (self.res["hold_reason"], self.res["margin"], self.res["info"]))

    def test_transform_survives_the_perturbation_within_D2(self):
        self.assert_transform_close(self.res["best"]["transform"], self.tf)

    def test_untouched_half_of_the_walk_is_still_placed_within_D2(self):
        got = ps.apply_candidate_transform(self.res["best"]["transform"], self.traj)
        keep = self.plan_walk[:, 0] <= self.CHANGED_FROM_X
        self.assertGreater(int(keep.sum()), 50)
        err = float(np.linalg.norm(got[keep] - self.plan_walk[keep], axis=1).max())
        self.assertLessEqual(err, D2_OFFSET, f"max {err:.3f} m on the unchanged corridor")

    def test_the_changed_geometry_is_reported_as_outliers(self):
        out = self.res["best"]["outliers"]
        self.assertTrue(out, "a moved wall and a deleted door must not vanish silently")
        kinds = {o["kind"] for o in out}
        self.assertIn("leg", kinds, out)           # the moved corridor
        self.assertIn("door", kinds, out)          # the deleted door

    def test_no_untouched_event_is_blamed(self):
        """Precision: every reported outlier really is downstream of the moved wall —
        an outlier list that also flags the intact half would be useless as a change
        report."""
        for o in self.res["best"]["outliers"]:
            xy = self._event_plan_xy(int(o["event_index"]))
            self.assertGreaterEqual(float(xy[0]), self.CHANGED_FROM_X,
                                    f"outlier on the UNCHANGED corridor: {o} at {xy}")

    def test_inlier_ratio_stays_above_the_gate(self):
        best = self.res["best"]
        self.assertGreaterEqual(best["inlier_ratio"], self.res["gates"]["min_inlier_ratio"])
        self.assertLess(best["inlier_ratio"], 1.0)      # ...but it did notice something
        self.assertLessEqual(best["residual"], self.res["gates"]["max_residual"])

    def test_outlier_rows_carry_a_usable_report(self):
        for o in self.res["best"]["outliers"]:
            self.assertEqual(sorted(o), sorted(ps.OUTLIER_FIELDS))
            self.assertTrue(np.isfinite(o["residual"]))
        legs = [o for o in self.res["best"]["outliers"] if o["kind"] == "leg"]
        self.assertTrue(all(o["reason"].startswith(("no_plan_leg", "plan_leg_taken",
                                                    "leg_heading_mismatch")) for o in legs), legs)

    def test_a_tighter_gate_turns_the_same_evidence_into_a_refusal(self):
        """The gates are the bar and coarse_match cannot relax them: demand an inlier
        ratio this partial match cannot reach and the SAME evidence must REJECT."""
        r = cm.coarse_match(self.skel, self.traj, door_s=self.ds,
                            gates={"min_inlier_ratio": 0.95})
        self.assert_contract(r, self.skel, self.traj, self.ds)
        self.assertEqual((r["status"], r["hold_reason"]),
                         ("reject", "inlier_ratio_below_min"))
        self.assertIsNone(r["best"])
        self.assertTrue(r["candidates"])                # ...but the evidence is still shown


# --------------------------------------------------------------------------------------
# 4 — SCENARIO 3: ambiguity -> HOLD, never an auto-confirmed guess
# --------------------------------------------------------------------------------------

class TestAmbiguityHolds(_MatchAssertions):

    def _assert_hold(self, res, skel, traj, ds=None, reason="ambiguous_margin"):
        self.assert_contract(res, skel, traj, ds)
        self.assertEqual((res["status"], res["hold_reason"]), ("hold", reason),
                         res["info"])
        self.assertFalse(res["ok"])
        self.assertIsNone(res["best"])

    def test_repeated_parallel_corridors_hold_and_expose_both_places(self):
        """Two IDENTICAL L corridors 40 m apart: the walk fits both exactly, so there is
        no evidence that says which one — HOLD, with both placements listed."""
        segs = _l_corridor_segments()
        skel = ps.corridor_skeleton(np.vstack([segs, segs + np.array([40.0, 0.0])]),
                                    doors=[(W, 4.0), (W + 40.0, 4.0)])
        self.assertEqual(len(skel["legs"]), 4, skel["info"])
        walk = [(CENTER_A, 0.0), (CENTER_A, CENTER_B), (10.0, CENTER_B)]
        tf, traj, _, ds = _make_walk(walk, [(CENTER_A, 4.0)])
        res = cm.coarse_match(skel, traj, door_s=ds)
        self._assert_hold(res, skel, traj, ds)
        self.assertGreaterEqual(len(res["candidates"]), 2)
        self.assertLess(res["margin"], res["gates"]["margin_min"])
        top = [c["transform"] for c in res["candidates"][:2]]
        self.assertEqual(top[0]["chi"], top[1]["chi"])
        self.assertAlmostEqual(top[0]["yaw_deg"], top[1]["yaw_deg"], delta=0.5)
        dx = abs(top[0]["translation"][0] - top[1]["translation"][0])
        self.assertAlmostEqual(dx, 40.0, delta=0.5)     # exactly the repeat pitch
        # the true placement is among them, it is just not singled out
        self.assertTrue(any(abs(c["transform"]["translation"][0] - tf["translation"][0]) < 0.1
                            and abs(c["transform"]["translation"][1] - tf["translation"][1]) < 0.1
                            for c in res["candidates"]), res["candidates"])

    def test_two_leg_L_is_ambiguous_by_construction(self):
        """The module's own load-bearing claim: 2 legs + 1 corner + free (s_f, s_h) fit
        an L corridor forwards AND backwards-mirrored, both at residual 0."""
        skel = ps.corridor_skeleton(_l_corridor_segments())
        walk = [(CENTER_A, 0.0), (CENTER_A, CENTER_B), (10.0, CENTER_B)]
        tf, traj, _, _ = _make_walk(walk)
        res = cm.coarse_match(skel, traj)
        self._assert_hold(res, skel, traj)
        self.assertEqual(res["margin"], 0.0)
        a, b = res["candidates"][0]["transform"], res["candidates"][1]["transform"]
        self.assertNotEqual(a["chi"], b["chi"])         # the twin is the MIRRORED fit
        self.assertLess(abs(a["s_h"] - b["s_h"]) / a["s_h"], 0.05)   # ...s_h agrees,
        self.assertGreater(abs(a["s_f"] - b["s_f"]) / a["s_f"], 0.10)  # ...s_f does not

    def test_a_door_event_is_what_breaks_the_two_leg_tie(self):
        """Evidence, not a tie-break rule: add ONE door and the same walk resolves."""
        skel = ps.corridor_skeleton(_l_corridor_segments(), doors=[(W, 4.0)])
        self.assertEqual(len(skel["doors"]), 1)
        walk = [(CENTER_A, 0.0), (CENTER_A, CENTER_B), (10.0, CENTER_B)]
        tf, traj, _, ds = _make_walk(walk, [(CENTER_A, 4.0)])
        res = cm.coarse_match(skel, traj, door_s=ds)
        self.assert_contract(res, skel, traj, ds)
        self.assertEqual(res["status"], "ok", (res["hold_reason"], res["info"]))
        self.assert_transform_close(res["best"]["transform"], tf,
                                    yaw_tol=0.05, scale_rel=2e-3, offset=0.01)
        self.assertGreaterEqual(res["margin"], res["gates"]["margin_min"])

    def test_s_h_prior_does_not_break_the_two_leg_tie(self):
        """MEASURED, and the module docstring says so: the backwards fit keeps s_h
        (1.97 -> 2.01 here), so no usable s_h tolerance separates the two. A prior must
        not be allowed to look like evidence it is not."""
        skel = ps.corridor_skeleton(_l_corridor_segments())
        walk = [(CENTER_A, 0.0), (CENTER_A, CENTER_B), (10.0, CENTER_B)]
        _, traj, _, _ = _make_walk(walk)
        for rel in (0.25, 0.10):
            res = cm.coarse_match(skel, traj, s_h_prior=TRUE["s_h"], s_h_rel_tol=rel)
            self._assert_hold(res, skel, traj)

    def test_a_walk_without_a_turn_cannot_be_placed(self):
        skel = _stair_skeleton()
        _, traj, _, _ = _make_walk([(0.91, 0.0), (0.91, 11.09)])
        res = cm.coarse_match(skel, traj)
        self._assert_hold(res, skel, traj, reason="insufficient_events")
        self.assertEqual(res["candidates"], [])

    def test_no_plan_skeleton_holds(self):
        skel = ps.corridor_skeleton(np.zeros((0, 2, 2)))
        _, traj, _, _ = _make_walk(STAIR)
        res = cm.coarse_match(skel, traj)
        self._assert_hold(res, skel, traj, reason="no_plan_skeleton")

    def test_impossible_anisotropy_generates_nothing(self):
        """aniso_band is a PREFILTER on the seed tensor: squeeze it to isotropic and the
        anisotropic truth cannot even be hypothesised -> HOLD, not a squashed fit."""
        skel = _stair_skeleton()
        _, traj, _, ds = _make_walk(STAIR, STAIR_WALK_DOORS)
        res = cm.coarse_match(skel, traj, door_s=ds, aniso_band=(0.99, 1.01))
        self._assert_hold(res, skel, traj, ds, reason="no_candidate")
        self.assertEqual(res["info"]["n_hypotheses"], 0)


# --------------------------------------------------------------------------------------
# 5 — SCENARIO 4: nothing that fits -> REJECT, and no fabricated transform
# --------------------------------------------------------------------------------------

class TestRefusals(_MatchAssertions):

    def _assert_reject(self, res, skel, traj, ds, reason):
        self.assert_contract(res, skel, traj, ds)
        self.assertEqual((res["status"], res["hold_reason"]), ("reject", reason), res["info"])
        self.assertFalse(res["ok"])
        self.assertIsNone(res["best"])
        self.assertTrue(res["candidates"], "a REJECT must still show what it refused")

    def test_walk_that_does_not_belong_to_the_plan_is_rejected(self):
        """A 6-leg zigzag walk offered to a 2-leg L corridor: at best 2 legs and 1
        corner can be explained, so the inlier gate refuses instead of stretching the
        scales until something lines up."""
        skel = ps.corridor_skeleton(_l_corridor_segments())
        zig = [(0.0, 0.0), (0.0, 9.0), (7.0, 9.0), (7.0, 0.0), (14.0, 0.0), (14.0, 9.0),
               (21.0, 9.0)]
        tf, traj, _, _ = _make_walk(zig)
        self.assertEqual(cm.recon_summary(cm.recon_events(traj))["n_legs"], 6)
        res = cm.coarse_match(skel, traj)
        self._assert_reject(res, skel, traj, None, "inlier_ratio_below_min")
        self.assertLess(res["candidates"][0]["inlier_ratio"],
                        res["gates"]["min_inlier_ratio"])
        self.assertTrue(res["candidates"][0]["outliers"])

    def test_scale_outside_the_physical_band_is_rejected_not_clipped(self):
        skel = _stair_skeleton()
        tf, traj, _, ds = _make_walk(STAIR, STAIR_WALK_DOORS)
        res = cm.coarse_match(skel, traj, door_s=ds, scale_band=(0.05, 1.0))
        self._assert_reject(res, skel, traj, ds, "scale_out_of_band")
        self.assertEqual(res["info"]["n_out_of_band"], len(res["candidates"]))
        for c in res["candidates"]:                     # reported unclipped, as measured
            self.assertGreater(max(c["transform"]["s_f"], c["transform"]["s_h"]), 1.0)

    def test_a_disagreeing_independent_prior_is_a_refusal(self):
        """`s_h_prior` is a measurement, so a candidate that contradicts it is refused —
        it is never quietly pulled onto the prior."""
        skel = _stair_skeleton()
        tf, traj, _, ds = _make_walk(STAIR, STAIR_WALK_DOORS)
        res = cm.coarse_match(skel, traj, door_s=ds, s_h_prior=0.5)
        self._assert_reject(res, skel, traj, ds, "scale_out_of_band")
        self.assertEqual(res["info"]["s_h_prior"][0], 0.5)
        for c in res["candidates"]:
            self.assertGreater(abs(c["transform"]["s_h"] - 0.5), 0.25 * 0.5)

    def test_a_correct_prior_prunes_mirrored_candidates(self):
        """The measured, honest benefit of s_h_prior, RE-MEASURED after the P2 scoring
        fix: it prunes candidates that disagree with it — every one of them mirrored —
        and it never makes the answer more ambiguous.

        What it no longer does on this fixture is WIDEN the margin, and that is a result,
        not a slackened assertion: the mirrors used to reach the runner-up slot, so
        deleting them opened the gap (the old measurement, 0.22 -> 0.37). Scoring the
        residual on the fraction it actually explains already keeps them out of that slot
        (plain margin 0.263, top-2 gap now set by a NON-mirrored fit), so at the default
        +-25 % the prior removes only lower-ranked mirrors and the margin is unchanged to
        the last digit. The benefit is still there and still measurable — it just needs a
        tolerance tight enough to reach the surviving mirror (s_h 1.491, inside +-25 % of
        1.97 but outside +-20 %), which `test_a_tighter_prior_reaches_the_last_mirror`
        pins."""
        skel = _stair_skeleton()
        tf, traj, _, ds = _make_walk(STAIR, STAIR_WALK_DOORS)
        plain = cm.coarse_match(skel, traj, door_s=ds)
        primed = cm.coarse_match(skel, traj, door_s=ds, s_h_prior=TRUE["s_h"])
        self.assert_contract(primed, skel, traj, ds)
        self.assertEqual(primed["status"], "ok")
        self.assertLess(len(primed["candidates"]), len(plain["candidates"]))
        # ...and what it removed is exactly what disagreed with the PRIOR (that is the
        # mechanism — s_h, not handedness), mirrors among them.
        kept = {c["method"] for c in primed["candidates"]}
        dropped = [c for c in plain["candidates"] if c["method"] not in kept]
        self.assertTrue(dropped)
        for c in dropped:
            self.assertGreater(abs(c["transform"]["s_h"] - TRUE["s_h"]), 0.25 * TRUE["s_h"], c)
        self.assertTrue(any(int(c["transform"]["chi"]) != int(tf["chi"]) for c in dropped),
                        dropped)
        self.assertGreaterEqual(primed["margin"], plain["margin"])   # never MORE ambiguous
        self.assert_transform_close(primed["best"]["transform"], tf,
                                    yaw_tol=0.05, scale_rel=2e-3, offset=0.01)

    def test_a_tighter_prior_reaches_the_last_mirror(self):
        """The prior's margin benefit, still real, now stated at the tolerance where it
        actually applies: +-20 % excludes the surviving mirror's s_h (1.491) and the
        top-2 gap opens 0.263 -> 0.472. A prior is evidence with a reach, and this pins
        the reach instead of claiming it holds at every tolerance."""
        skel = _stair_skeleton()
        tf, traj, _, ds = _make_walk(STAIR, STAIR_WALK_DOORS)
        plain = cm.coarse_match(skel, traj, door_s=ds)
        tight = cm.coarse_match(skel, traj, door_s=ds, s_h_prior=TRUE["s_h"],
                                s_h_rel_tol=0.20)
        self.assert_contract(tight, skel, traj, ds)
        self.assertEqual(tight["status"], "ok")
        self.assertGreater(tight["margin"], plain["margin"] + 0.15)
        for c in tight["candidates"][1:]:
            self.assertEqual(int(c["transform"]["chi"]), int(tf["chi"]), c)  # no mirror left
        self.assert_transform_close(tight["best"]["transform"], tf,
                                    yaw_tol=0.05, scale_rel=2e-3, offset=0.01)

    def test_no_refusal_ever_carries_a_transform(self):
        """The single invariant every refusal path shares, checked in one place."""
        skel = _stair_skeleton()
        tf, traj, _, ds = _make_walk(STAIR, STAIR_WALK_DOORS)
        cases = [cm.coarse_match(skel, traj, door_s=ds, scale_band=(0.05, 1.0)),
                 cm.coarse_match(skel, traj, door_s=ds, s_h_prior=0.5),
                 cm.coarse_match(skel, traj, door_s=ds, gates={"min_inlier_ratio": 1.5}),
                 cm.coarse_match(skel, traj, door_s=ds, gates={"max_residual": -1.0}),
                 cm.coarse_match(skel, traj, door_s=ds, aniso_band=(0.99, 1.01)),
                 cm.coarse_match(ps.corridor_skeleton(np.zeros((0, 2, 2))), traj)]
        for res in cases:
            ok, errs = ps.validate_match_result(res)
            self.assertTrue(ok, errs)
            self.assertNotEqual(res["status"], "ok", res["hold_reason"])
            self.assertIsNone(res["best"])
            self.assertFalse(res["ok"])
            self.assertIsNotNone(res["hold_reason"])


if __name__ == "__main__":
    unittest.main()

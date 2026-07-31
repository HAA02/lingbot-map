"""Tests for scan2bim.plan_skeleton: the DXF corridor skeleton (centreline / corner /
door graph) and the PLAN-MATCH CANDIDATE SCHEMA that coarse_match.py and
tools/build_coplay.py --plan-match must obey.

Geometry is exercised on SYNTHETIC ezdxf plans (same fixture style as
tests/test_dxf_plan.py:163) with a KNOWN L-shaped corridor of clear width 1.82 m — the
Gasan_7F corridor value — so every extracted number has a ground truth:

      y=12.0  +-----------------------------+      corridor B (along +x), clear 1.82 m
              |                             |      centreline y = 11.09
      y=10.18 |     +-----------------------+
              |     |                              corner at (0.91, 11.09), turn 90 deg
              |     |                              corridor A (along +y), clear 1.82 m
      y=0     +-----+                              centreline x = 0.91
             x=0   x=1.82                  x=10

The 4 m x 3 m room hung off corridor B must NOT become a leg (width out of band), the
A-DOOR insert at (1.82, 4.0) must land on leg A at s = 4.0, and the raw facing-pair end
of leg A must agree with forward_scale.corridor_open_boundary's L_end (the independent
"known value" that the fwdscale cycle established).

NOTE: no real SXX/Gasan DXF exists in this repo, so these are synthetic-only checks;
running plan_skeleton on the real plan is still required before trusting field numbers.
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import ezdxf  # noqa: F401
    _HAS_EZDXF = True
except ImportError:
    _HAS_EZDXF = False

from scan2bim import plan_skeleton as ps

W = 1.82                      # known corridor clear width (m)
Y_INNER = 10.18               # inner corner y (12.0 - W)
CENTER_A = W / 2.0            # 0.91
CENTER_B = 12.0 - W / 2.0     # 11.09
_MM = 1000.0                  # $INSUNITS = 4 (mm)
_WALL = "A-WALL-____-OTLN"
_DOOR = "A-DOOR-____-OTLN"


def _l_corridor_segments(thick: float = 0.0, cap: bool = True, room: bool = True) -> np.ndarray:
    """The L corridor above as (N,2,2) metre segments. thick>0 adds the OUTER wall
    faces (a real DXF wall is two lines), which must not inflate the clear width."""
    segs = [((0.0, 0.0), (0.0, 12.0)),                 # outer wall of corridor A
            ((W, 0.0), (W, Y_INNER)),                  # inner wall of corridor A
            ((W, Y_INNER), (10.0, Y_INNER)),           # inner wall of corridor B
            ((0.0, 12.0), (10.0, 12.0))]               # outer wall of corridor B
    if cap:
        segs.append(((0.0, 0.0), (W, 0.0)))            # dead end at y=0
    if room:                                           # 4 x 3 m room off corridor B
        segs += [((3.0, 12.0), (3.0, 15.0)), ((7.0, 12.0), (7.0, 15.0)),
                 ((3.0, 15.0), (7.0, 15.0))]
    if thick > 0:
        t = thick
        segs += [((-t, 0.0), (-t, 12.0 + t)), ((W + t, 0.0), (W + t, Y_INNER - t)),
                 ((W + t, Y_INNER - t), (10.0, Y_INNER - t)),
                 ((-t, 12.0 + t), (10.0, 12.0 + t))]
    return np.asarray(segs, dtype=np.float64).reshape(-1, 2, 2)


def _write_l_corridor(tmp: Path, doors=((W, 4.0), (30.0, 30.0))) -> Path:
    """Same L corridor as a real ezdxf document (mm units, A-WALL lines + A-DOOR
    block inserts)."""
    doc = ezdxf.new(setup=True)
    doc.header["$INSUNITS"] = 4
    doc.blocks.new(name="DOOR")
    msp = doc.modelspace()
    for (p, q) in _l_corridor_segments():
        msp.add_line((p[0] * _MM, p[1] * _MM), (q[0] * _MM, q[1] * _MM),
                     dxfattribs={"layer": _WALL})
    for (dx, dy) in doors:
        msp.add_blockref("DOOR", (dx * _MM, dy * _MM), dxfattribs={"layer": _DOOR})
    p = tmp / "l_corridor.dxf"
    doc.saveas(p)
    return p


def _leg_along(skel, axis_vec, tol=0.15):
    """Legs whose direction is (anti)parallel to axis_vec."""
    a = np.asarray(axis_vec, dtype=np.float64)
    a = a / np.linalg.norm(a)
    return [g for g in skel["legs"] if abs(float(np.asarray(g["dir"]) @ a)) > 1.0 - tol]


class TestCorridorSkeletonGeometry(unittest.TestCase):
    """Known geometry in -> known centreline/corner/width out (no DXF file needed)."""

    def setUp(self):
        self.skel = ps.corridor_skeleton(_l_corridor_segments())

    def test_two_legs_with_known_clear_width_1_82(self):
        self.assertEqual(len(self.skel["legs"]), 2, self.skel["info"])
        for g in self.skel["legs"]:
            self.assertAlmostEqual(g["width"], W, delta=0.02, msg=g)
        self.assertAlmostEqual(self.skel["info"]["median_width"], W, delta=0.02)

    def test_centrelines_sit_on_the_known_midlines(self):
        legA = _leg_along(self.skel, (0.0, 1.0))
        legB = _leg_along(self.skel, (1.0, 0.0))
        self.assertEqual((len(legA), len(legB)), (1, 1), self.skel["legs"])
        self.assertAlmostEqual(legA[0]["a"][0], CENTER_A, delta=0.02)   # x = 0.91
        self.assertAlmostEqual(legB[0]["a"][1], CENTER_B, delta=0.02)   # y = 11.09

    def test_single_L_corner_node_with_90_deg_turn(self):
        corners = [n for n in self.skel["nodes"] if n["kind"] == "corner"]
        self.assertEqual(len(corners), 1, self.skel["nodes"])
        c = corners[0]
        self.assertAlmostEqual(c["xy"][0], CENTER_A, delta=0.05)
        self.assertAlmostEqual(c["xy"][1], CENTER_B, delta=0.05)
        self.assertAlmostEqual(c["turn_deg"], 90.0, delta=2.0)
        self.assertEqual(sorted(c["legs"]), [0, 1])

    def test_leg_arclengths_reach_the_corner(self):
        legA = _leg_along(self.skel, (0.0, 1.0))[0]
        legB = _leg_along(self.skel, (1.0, 0.0))[0]
        self.assertAlmostEqual(legA["length"], CENTER_B - 0.0, delta=0.1)   # 0 -> 11.09
        self.assertAlmostEqual(legB["length"], 10.0 - CENTER_A, delta=0.1)  # 0.91 -> 10
        self.assertEqual(self.skel["info"]["n_ends"], 2)                    # 2 free ends

    def test_room_is_not_a_corridor_leg(self):
        """The 4 x 3 m room (gap 3.0 m, outside the width band) must not appear."""
        for g in self.skel["legs"]:
            self.assertLess(max(g["a"][1], g["b"][1]), 12.5, f"phantom room leg: {g}")

    def test_thick_walls_keep_the_clear_width(self):
        skel = ps.corridor_skeleton(_l_corridor_segments(thick=0.2, cap=False))
        self.assertEqual(len(skel["legs"]), 2, skel["info"])
        for g in skel["legs"]:
            self.assertAlmostEqual(g["width"], W, delta=0.02, msg=g)   # NOT 2.02/2.22
            self.assertGreater(g["n_pairs"], 1, msg=g)                 # several faces merged

    def test_rotated_plan_recovers_the_same_geometry(self):
        """A plan drawn at 23 deg must give the same widths/lengths/turn: the dominant
        wall orientation is estimated, not assumed axis-aligned."""
        R = ps._rot(23.0)
        segs = _l_corridor_segments().reshape(-1, 2) @ R.T
        skel = ps.corridor_skeleton(segs.reshape(-1, 2, 2))
        self.assertEqual(len(skel["legs"]), 2, skel["info"])
        self.assertAlmostEqual(skel["info"]["theta0_deg"], 23.0, delta=0.5)
        for g in skel["legs"]:
            self.assertAlmostEqual(g["width"], W, delta=0.02, msg=g)
        corners = [n for n in skel["nodes"] if n["kind"] == "corner"]
        self.assertEqual(len(corners), 1, skel["nodes"])
        exp = R @ np.array([CENTER_A, CENTER_B])
        self.assertAlmostEqual(corners[0]["xy"][0], exp[0], delta=0.06)
        self.assertAlmostEqual(corners[0]["xy"][1], exp[1], delta=0.06)
        self.assertAlmostEqual(corners[0]["turn_deg"], 90.0, delta=2.0)
        lens = sorted(g["length"] for g in skel["legs"])
        self.assertAlmostEqual(lens[0], 10.0 - CENTER_A, delta=0.1)
        self.assertAlmostEqual(lens[1], CENTER_B, delta=0.1)

    def test_no_fabrication_on_empty_or_unusable_input(self):
        for segs in (np.zeros((0, 2, 2)),
                     np.asarray([[[0.0, 0.0], [0.0, 8.0]]]),          # single wall
                     _l_corridor_segments()[[5, 6, 7]]):              # room only
            skel = ps.corridor_skeleton(segs)
            self.assertEqual(skel["legs"], [])
            self.assertEqual(skel["nodes"], [])
            self.assertIn("fail", skel["info"])

    def test_skeleton_is_json_serialisable(self):
        s = json.dumps(self.skel)
        self.assertEqual(json.loads(s)["schema"], ps.SKELETON_SCHEMA)

    def test_plan_events_ids_and_kinds(self):
        skel = ps.corridor_skeleton(_l_corridor_segments(), doors=[(W, 4.0)])
        ev = ps.plan_events(skel)
        kinds = [e["kind"] for e in ev]
        self.assertEqual(kinds.count("leg"), 2)
        self.assertEqual(kinds.count("corner"), 1)
        self.assertEqual(kinds.count("door"), 1)
        self.assertEqual([e["id"] for e in ev][:2], ["leg:0", "leg:1"])
        self.assertTrue(all(":" in e["id"] for e in ev))
        json.dumps(ev)


class TestKnownValueCrossChecks(unittest.TestCase):
    """The skeleton's own numbers vs the independently-derived known values."""

    def test_leg_end_matches_forward_scale_L_end(self):
        """leg A's raw facing-pair end (b_wall) is the corridor OPEN BOUNDARY, so it
        must agree with forward_scale.corridor_open_boundary's L_end (step 0.5 m)."""
        from scan2bim.forward_scale import corridor_open_boundary
        segs = _l_corridor_segments()
        skel = ps.corridor_skeleton(segs)
        legA = _leg_along(skel, (0.0, 1.0))[0]
        L_end, info = corridor_open_boundary(segs, axis_origin=legA["a"],
                                             axis_dir=legA["dir"], s_range=(-2.0, 20.0))
        self.assertIsNotNone(L_end, info)
        self.assertLessEqual(float(np.linalg.norm(np.asarray(legA["b_wall"]) - L_end)), 0.6,
                             f"b_wall {legA['b_wall']} vs L_end {info}")
        self.assertAlmostEqual(legA["b_wall"][1], Y_INNER, delta=0.05)

    def test_width_matches_dxf_plan_corridor_widths_near(self):
        """The skeleton width and dxf_plan's trajectory-based width must be the same
        1.82 m, with the skeleton's own centreline used as the trajectory — PER LEG,
        because corridor_widths_near fits one global PCA axis (an L-shaped centreline
        hands it a diagonal axis and it correctly abstains; measured below)."""
        from scan2bim import dxf_plan
        segs = _l_corridor_segments()
        skel = ps.corridor_skeleton(segs)
        for g in skel["legs"]:
            traj = ps.centerline_points(skel, leg_ids=[g["id"]])
            self.assertGreater(len(traj), 20)
            widths, info = dxf_plan.corridor_widths_near(segs, traj)
            self.assertTrue(widths, info)
            self.assertAlmostEqual(min(widths, key=lambda w: abs(w - W)), W, delta=0.05)
            self.assertAlmostEqual(g["width"], min(widths, key=lambda w: abs(w - W)),
                                   delta=0.05)
        both = ps.centerline_points(skel)                     # both legs -> diagonal axis
        self.assertIn("fail", dxf_plan.corridor_widths_near(segs, both)[1])


@unittest.skipUnless(_HAS_EZDXF, "ezdxf not installed")
class TestPlanSkeletonFromDxf(unittest.TestCase):
    """End-to-end on a synthetic DXF file (walls + A-DOOR inserts via dxf_plan)."""

    def test_doors_attach_to_a_leg_with_arclength(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = _write_l_corridor(Path(td))
            skel = ps.plan_skeleton(p)
            self.assertEqual(len(skel["legs"]), 2, skel["info"])
            self.assertEqual(skel["info"]["doors_dxf"]["n_doors_total"], 2)
            self.assertEqual(len(skel["doors"]), 1, skel["doors"])   # (30,30) is not a corridor door
            d = skel["doors"][0]
            legA = _leg_along(skel, (0.0, 1.0))[0]
            self.assertEqual(d["leg"], legA["id"])
            self.assertAlmostEqual(d["s"], 4.0, delta=0.15)          # 4 m from the dead end
            self.assertAlmostEqual(abs(d["offset"]), W / 2.0, delta=0.05)
            self.assertEqual(legA["doors"], [d["id"]])
            self.assertEqual(sum(1 for n in skel["nodes"] if n["kind"] == "door"), 1)

    def test_mm_units_and_widths_read_off_the_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            skel = ps.plan_skeleton(_write_l_corridor(Path(td)))
            self.assertAlmostEqual(skel["info"]["median_width"], W, delta=0.02)
            self.assertEqual(skel["info"]["n_corners"], 1)
            json.dumps(skel)


class TestCandidateSchema(unittest.TestCase):
    """The dev-wire contract: schema shape + the HOLD/REJECT invariants."""

    def _cand(self, score=0.9, n_in=9, n_ev=10, residual=0.2, yaw=12.0):
        tf = ps.make_transform(yaw_deg=yaw, chi=-1, s_h=1.97, s_f=3.53, theta_deg=0.0,
                               translation=[1.0, -2.0])
        return ps.make_candidate(tf, score=score, n_inliers=n_in, n_events=n_ev,
                                 residual=residual, matches=[(0, "leg:0"), (1, "corner:2")],
                                 outliers=[{"kind": "door", "event_index": 7,
                                            "plan_event_id": None, "residual": 2.4,
                                            "reason": "no_plan_door_within_tol"}],
                                 method="corner2+trimmed")

    def test_clear_winner_is_ok_and_schema_valid(self):
        res = ps.finalize_match([self._cand(score=0.90), self._cand(score=0.40, yaw=90.0)])
        ok, errs = ps.validate_match_result(res)
        self.assertTrue(ok, errs)
        self.assertEqual(res["status"], "ok")
        self.assertTrue(res["ok"])
        self.assertEqual(res["best"]["rank"], 0)
        self.assertAlmostEqual(res["margin"], 0.5, places=5)
        json.dumps(res)

    def test_ambiguous_margin_holds_and_returns_no_transform(self):
        """Design invariant: score gap < margin_min -> HOLD, never auto-confirm."""
        res = ps.finalize_match([self._cand(score=0.80), self._cand(score=0.78, yaw=90.0)])
        ok, errs = ps.validate_match_result(res)
        self.assertTrue(ok, errs)
        self.assertEqual(res["status"], "hold")
        self.assertFalse(res["ok"])
        self.assertEqual(res["hold_reason"], "ambiguous_margin")
        self.assertIsNone(res["best"])
        self.assertEqual(len(res["candidates"]), 2)       # the tie is exposed, not hidden

    def test_residual_and_inlier_gates_reject(self):
        res = ps.finalize_match([self._cand(score=0.9, residual=1.2)])
        self.assertEqual((res["status"], res["hold_reason"]), ("reject", "residual_exceeded"))
        self.assertIsNone(res["best"])
        self.assertTrue(ps.validate_match_result(res)[0], ps.validate_match_result(res)[1])
        res2 = ps.finalize_match([self._cand(score=0.9, n_in=3, n_ev=10)])
        self.assertEqual((res2["status"], res2["hold_reason"]),
                         ("reject", "inlier_ratio_below_min"))
        self.assertTrue(ps.validate_match_result(res2)[0])

    def test_no_candidate_holds(self):
        res = ps.finalize_match([])
        self.assertEqual((res["status"], res["hold_reason"]), ("hold", "no_candidate"))
        self.assertTrue(ps.validate_match_result(res)[0])

    def test_single_candidate_has_no_margin(self):
        res = ps.finalize_match([self._cand(score=0.72)])
        self.assertEqual(res["status"], "ok")
        self.assertIsNone(res["margin"])

    def test_gates_are_reported(self):
        res = ps.finalize_match([self._cand()], gates={"margin_min": 0.2})
        self.assertEqual(res["gates"]["margin_min"], 0.2)
        self.assertEqual(res["gates"]["max_residual"], ps.DEFAULT_GATES["max_residual"])

    def test_inlier_ratio_is_derived_not_asserted(self):
        c = self._cand(n_in=7, n_ev=10)
        self.assertAlmostEqual(c["inlier_ratio"], 0.7, places=5)

    def test_validator_catches_contract_violations(self):
        good = ps.finalize_match([self._cand(score=0.9), self._cand(score=0.4, yaw=90.0)])

        bad = json.loads(json.dumps(good)); bad["status"] = "hold"          # ok flag stale
        self.assertFalse(ps.validate_match_result(bad)[0])

        bad = json.loads(json.dumps(good)); bad["candidates"].reverse()      # unsorted
        self.assertFalse(ps.validate_match_result(bad)[0])

        bad = json.loads(json.dumps(good)); bad["candidates"][0]["transform"].pop("chi")
        self.assertFalse(ps.validate_match_result(bad)[0])

        bad = json.loads(json.dumps(good)); bad["candidates"][0]["transform"]["s_h"] = 0.0
        self.assertFalse(ps.validate_match_result(bad)[0])

        bad = json.loads(json.dumps(good)); bad["candidates"][0]["n_inliers"] = 2
        self.assertFalse(ps.validate_match_result(bad)[0])

        bad = json.loads(json.dumps(good)); bad["candidates"][0]["outliers"][0].pop("reason")
        self.assertFalse(ps.validate_match_result(bad)[0])

        bad = json.loads(json.dumps(good)); bad["schema"] = "coplay.plan_match/0.9"
        self.assertFalse(ps.validate_match_result(bad)[0])

        hold = ps.make_hold_result("ambiguous_margin", [self._cand()])
        hold["best"] = hold["candidates"][0]                                 # smuggled transform
        self.assertFalse(ps.validate_match_result(hold)[0])

    def test_unknown_reasons_are_refused(self):
        with self.assertRaises(ValueError):
            ps.make_hold_result("because_i_said_so")
        with self.assertRaises(ValueError):
            ps.make_reject_result("ambiguous_margin")        # that is a HOLD, not a reject


class TestApplyCandidateTransform(unittest.TestCase):
    """The normative transform composition p = R(yaw) A(theta; s_f, s_h) diag(chi,1) q + t."""

    def _apply(self, pts, **kw):
        tf = ps.make_transform(**{"yaw_deg": 0.0, "chi": 1, "s_h": 1.0, "s_f": 1.0,
                                  "theta_deg": 0.0, "translation": [0.0, 0.0], **kw})
        return ps.apply_candidate_transform(tf, pts)

    def test_identity(self):
        q = np.array([[1.0, 0.0], [0.0, 1.0], [-2.0, 3.0]])
        np.testing.assert_allclose(self._apply(q), q, atol=1e-9)

    def test_yaw_rotates(self):
        np.testing.assert_allclose(self._apply([[1.0, 0.0]], yaw_deg=90.0),
                                   [[0.0, 1.0]], atol=1e-9)

    def test_chi_mirrors_x_before_anything_else(self):
        np.testing.assert_allclose(self._apply([[1.0, 2.0]], chi=-1), [[-1.0, 2.0]], atol=1e-9)
        # chi first, then yaw: mirror then rotate 90 deg
        np.testing.assert_allclose(self._apply([[1.0, 0.0]], chi=-1, yaw_deg=90.0),
                                   [[0.0, -1.0]], atol=1e-9)

    def test_anisotropic_scale_stretches_only_the_theta_axis(self):
        out = self._apply([[1.0, 0.0], [0.0, 1.0]], s_f=2.0, s_h=1.0, theta_deg=0.0)
        np.testing.assert_allclose(out, [[2.0, 0.0], [0.0, 1.0]], atol=1e-9)
        out90 = self._apply([[1.0, 0.0], [0.0, 1.0]], s_f=2.0, s_h=1.0, theta_deg=90.0)
        np.testing.assert_allclose(out90, [[1.0, 0.0], [0.0, 2.0]], atol=1e-9)

    def test_translation_is_applied_last(self):
        np.testing.assert_allclose(self._apply([[1.0, 0.0]], yaw_deg=90.0, s_f=2.0, s_h=2.0,
                                               translation=[5.0, -1.0]),
                                   [[5.0, 1.0]], atol=1e-9)

    def test_matches_forward_scale_tensor_convention(self):
        """The anisotropy must be exactly forward_scale.anisotropic_scale_tensor, so
        placement code and matcher cannot drift apart."""
        from scan2bim.forward_scale import anisotropic_scale_tensor
        q = np.array([[1.3, -0.7], [2.0, 4.0]])
        theta, s_f, s_h, yaw, chi, t = 31.0, 3.53, 1.97, -17.0, -1, np.array([2.0, 9.0])
        A = anisotropic_scale_tensor(np.deg2rad(theta), s_f, s_h)
        qm = q * np.array([chi, 1.0])
        expect = (qm @ A.T) @ ps._rot(yaw).T + t
        tf = ps.make_transform(yaw_deg=yaw, chi=chi, s_h=s_h, s_f=s_f, theta_deg=theta,
                               translation=t)
        np.testing.assert_allclose(ps.apply_candidate_transform(tf, q), expect, atol=1e-9)


if __name__ == "__main__":
    unittest.main()

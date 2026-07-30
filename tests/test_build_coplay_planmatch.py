"""Tests for tools/build_coplay.py --plan-match / --accept-plan-match (P1-Wire-b).

Two contracts from docs/TEAM_coplay-planmatch-01.md are pinned here:

  D4, no regression — --plan-match unspecified / "off" must be byte-identical to the
      existing place_rigid output (a default-value mistake would hit D4 directly), and
      the CLI defaults that guarantee it are locked against a silent future edit.

  DESIGN INVARIANT (2), no auto-confirmation — "BIM이 부정확해도 자동확정하지 않는다".
      This is the adversarial half: the fixture below is a match that succeeds
      PERFECTLY (status 'ok', residual 0, transform recovered exactly), i.e. the most
      tempting case there is for a pipeline to just apply it. --plan-match auto alone
      must STILL confirm nothing: candidates land on disk with applied=False, the
      returned reginfo stays mode 'rigid' with an unchanged anchor, and pose_json is
      byte-identical to the --plan-match off build. Only an explicit
      --accept-plan-match <candidate_id> may move the placement — and it is refused
      (loudly, with the candidate JSON still on disk for inspection) when the match is
      a 'hold' or the id does not exist.

GROUND TRUTH BY CONSTRUCTION, no fixtures needed: the plan is a synthetic corridor
written to a temporary DXF, and the recon walk is that corridor's own centreline mapped
into recon units with the INVERSE of a known transform — the harness from
tests/test_coarse_match.py (_corridor_walls / _make_walk), reused unchanged. So "the
accepted candidate really was applied" can be checked against where the walk is KNOWN
to belong on the plan (asserted to 1 cm), not merely against a changed field.
"""
import glob
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tools.build_coplay as bc  # noqa: E402
from scan2bim import plan_skeleton as ps  # noqa: E402
from test_build_coplay_rigid import (  # noqa: E402  (reuse the same fixture loaders — no duplication)
    _GASAN_GLOB, _S_H_OVERRIDE, _UPLOAD, _load_model, _load_recon,
)
from test_coarse_match import _corridor_walls, _make_walk  # noqa: E402  (matcher harness)
from test_plan_skeleton import CENTER_A, CENTER_B, _l_corridor_segments  # noqa: E402

try:
    import ezdxf
    _HAS_EZDXF = True
except ImportError:                                        # pragma: no cover
    _HAS_EZDXF = False

#: A 4-leg corridor whose leg lengths (6 / 20 / 6 / 8 m) break the reverse-mirrored
#: fit: with a free (s_f, s_h) pair the backwards traversal would need two different
#: forward scales at once, so this plan is UNAMBIGUOUS even with no door evidence
#: (place_rigid matches with door_s=None). Verified below: status 'ok', margin 0.30
#: vs the 0.08 gate, transform recovered exactly. tests/test_coarse_match.py's own
#: STAIR fixture is deliberately NOT used: without doors it ties (margin 0.0).
ASYM_SPINE = [(0.91, 0.0), (0.91, 6.0), (20.91, 6.0), (20.91, 12.0), (28.91, 12.0)]

#: The L corridor that is ambiguous BY CONSTRUCTION (2 legs + 1 corner + free s_f/s_h
#: fit forwards and backwards-mirrored at residual 0 — tests/test_coarse_match.py
#: TestAmbiguityHolds). Used to produce a 'hold' the accept path must refuse.
L_WALK = [(CENTER_A, 0.0), (CENTER_A, CENTER_B), (10.0, CENTER_B)]


def _write_wall_dxf(segments, path):
    """Wall segments (metres) -> a minimal A-WALL DXF in mm ($INSUNITS=4), the shape
    dxf_plan.load_wall_segments reads (same construction as tests/test_dxf_plan.py)."""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    for a, b in np.asarray(segments, dtype=np.float64).reshape(-1, 2, 2):
        msp.add_line((float(a[0]) * 1000.0, float(a[1]) * 1000.0),
                     (float(b[0]) * 1000.0, float(b[1]) * 1000.0),
                     dxfattribs={"layer": "A-WALL-____-OTLN"})
    doc.header["$INSUNITS"] = 4
    doc.saveas(path)
    return Path(path)


def _poses_from_xz(traj_xz):
    """Recon XZ walk -> reconstruction [R|t] rows (12 floats) whose viewer_pose centre
    is exactly (x, bob, z) with R = I. R = I makes place_rigid's gravity alignment the
    identity, so the matcher is handed EXACTLY `traj_xz` (its Cg0[:, [0, 2]]) — the
    walk whose true placement on the plan is known."""
    rows = []
    for i, (x, z) in enumerate(np.asarray(traj_xz, dtype=np.float64)):
        y = 0.02 * float(np.sin(0.7 * i))                  # keep a metric vertical bob
        rows.append([1.0, 0.0, 0.0, -float(x),
                     0.0, 1.0, 0.0, float(y),
                     0.0, 0.0, 1.0, float(z)])
    return rows


def _scan_cloud(traj_xz, n=1500, seed=0):
    """A recon-unit point cloud around the walk with a floor->ceiling vertical span, so
    place_rigid's s_v / floor-level estimators have something real to chew on. Sign
    convention matches place_rigid's own P[:, 1] *= -1; P[:, 2] *= -1 flip."""
    rng = np.random.RandomState(seed)
    xz = np.asarray(traj_xz, dtype=np.float64)
    k = rng.randint(0, len(xz), n)
    off = rng.normal(0.0, 0.4, (n, 2))
    y = rng.uniform(-0.75, 0.65, n)
    return np.column_stack([xz[k, 0] + off[:, 0], -y, -(xz[k, 1] + off[:, 1])])


class _PlanMatchFixture(unittest.TestCase):
    """Synthetic plan + walk + a model box, wired straight into place_rigid. No Gasan /
    upload fixtures and no FXX file: fxx_file=None makes place_rigid fall back to its
    documented model_ceiling-PCA yaw + centroid anchor, which is all the baseline needs
    to be a stable comparison target."""

    SPINE = ASYM_SPINE
    WALK = None                                            # None -> walk the whole spine

    @classmethod
    def setUpClass(cls):
        if not _HAS_EZDXF:
            raise unittest.SkipTest("ezdxf not installed")
        cls._td = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._td.name)
        cls.dxf = _write_wall_dxf(_corridor_walls(cls.SPINE), cls.tmp / "plan.dxf")
        walk = cls.SPINE if cls.WALK is None else cls.WALK
        cls.tf, cls.traj, cls.plan_walk, _ = _make_walk(walk)
        cls.poses = _poses_from_xz(cls.traj)
        cls.scan = _scan_cloud(cls.traj)
        rng = np.random.RandomState(1)
        cls.ceil = np.column_stack([rng.uniform(0.0, 30.0, 400), np.full(400, 3.0),
                                    rng.uniform(0.0, 15.0, 400)])
        cls.bbox = (np.array([0.0, 0.0, 0.0]), np.array([30.0, 3.0, 15.0]))

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    @classmethod
    def _place(cls, **kw):
        # _S_H_OVERRIDE (2.30) is deliberately NOT the fixture's true s_h (1.97): the
        # baseline's lateral scale must be distinguishable from the candidate's, or
        # "the accepted transform is what placed the walk" could not be told apart from
        # the heuristic placement. The matcher is unaffected (it sees the pre-scale Cg0).
        return bc.place_rigid(cls.poses, cls.scan, cls.ceil, cls.bbox, None,
                              horizontal_scale_override=_S_H_OVERRIDE, dxf_path=cls.dxf, **kw)


class TestAutoAloneConfirmsNothing(_PlanMatchFixture):
    """DESIGN INVARIANT (2), the hard case: a PERFECT match must still not be applied
    without --accept-plan-match."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.out = cls.tmp / "auto.plan_match.json"
        cls.pj_off, cls.info_off = cls._place(plan_match="off")
        cls.pj_auto, cls.info_auto = cls._place(plan_match="auto", plan_match_out=cls.out)
        cls.disk = json.loads(cls.out.read_text())

    def test_the_fixture_really_is_a_perfect_unambiguous_match(self):
        """Guards the test itself: if this fixture ever degraded into a hold/reject, the
        invariant assertions below would pass for the wrong reason (nothing to apply)."""
        pm = self.info_auto["plan_match"]
        self.assertEqual(pm["schema"], ps.PLAN_MATCH_SCHEMA)
        self.assertEqual(pm["status"], "ok", (pm["hold_reason"], pm["info"]))
        self.assertGreaterEqual(pm["margin"], pm["gates"]["margin_min"])
        self.assertEqual(pm["best"]["residual"], 0.0)
        self.assertEqual(pm["best"]["inlier_ratio"], 1.0)
        ok, errs = ps.validate_match_result({k: v for k, v in pm.items()
                                            if k not in ("applied", "accepted_candidate_id",
                                                         "persisted_to", "skeleton_info")})
        self.assertTrue(ok, errs)
        for k in ("chi", "yaw_deg", "s_f", "s_h"):         # recovered, not approximated
            self.assertAlmostEqual(float(pm["best"]["transform"][k]), float(self.tf[k]),
                                   delta=0.02, msg=pm["best"]["transform"])

    def test_placement_is_byte_identical_to_plan_match_off(self):
        """The whole point: candidates exist, and the build is still the off build."""
        self.assertEqual(self.pj_auto, self.pj_off)
        self.assertEqual({k: v for k, v in self.info_auto.items() if k != "plan_match"},
                         self.info_off)
        self.assertNotIn("plan_match", self.info_off)

    def test_reginfo_carries_no_confirmation_marker(self):
        pm = self.info_auto["plan_match"]
        self.assertEqual(self.info_auto["mode"], "rigid")           # NOT 'rigid+plan_match'
        self.assertFalse(self.info_auto["anchor"].startswith("plan_match_candidate:"),
                         self.info_auto["anchor"])
        self.assertEqual(self.info_auto["anchor"], self.info_off["anchor"])
        self.assertIs(pm["applied"], False)
        self.assertIsNone(pm["accepted_candidate_id"])

    def test_candidate_json_exists_and_is_addressable_but_unapplied(self):
        """The artifact a human adjudicates: on disk, complete, and NOT a confirmation.
        Every candidate carries a candidate_id --accept-plan-match can name."""
        self.assertTrue(self.out.exists())
        self.assertEqual(self.disk["schema"], ps.PLAN_MATCH_SCHEMA)
        self.assertEqual(self.disk["status"], "ok")
        self.assertIs(self.disk["applied"], False)
        self.assertIsNone(self.disk["accepted_candidate_id"])
        ids = [c["candidate_id"] for c in self.disk["candidates"]]
        self.assertTrue(ids)
        self.assertEqual(len(set(ids)), len(ids), ids)
        self.assertIn("0", ids)

    def test_diagnostics_are_exposed_for_adjudication(self):
        """reginfo 진단 노출: enough to judge WHY, not just that something matched."""
        pm = self.info_auto["plan_match"]
        for k in ("status", "candidates", "hold_reason", "margin", "gates", "info"):
            self.assertIn(k, pm)
        self.assertEqual(pm["persisted_to"], str(self.out))
        json.dumps(pm)                                              # must survive the wire

    def test_auto_without_an_out_path_still_confirms_nothing(self):
        """No candidate file requested -> diagnostics only, and still no confirmation
        (the invariant must not depend on the persistence argument)."""
        pj, info = self._place(plan_match="auto")
        self.assertEqual(pj, self.pj_off)
        self.assertEqual(info["mode"], "rigid")
        self.assertIs(info["plan_match"]["applied"], False)
        self.assertNotIn("persisted_to", info["plan_match"])


class TestExplicitAcceptIsTheOnlyWayIn(_PlanMatchFixture):
    """...and when it IS given, the accepted candidate's own transform is what places
    the walk — checked against the known ground-truth position on the plan."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.out = cls.tmp / "accepted.plan_match.json"
        cls.pj_off, cls.info_off = cls._place(plan_match="off")
        cls.pj, cls.info = cls._place(plan_match="auto", plan_match_out=cls.out,
                                      accept_plan_match="0")
        cls.disk = json.loads(cls.out.read_text())

    def test_reginfo_marks_the_accepted_candidate(self):
        self.assertEqual(self.info["mode"], "rigid+plan_match")
        self.assertEqual(self.info["anchor"], "plan_match_candidate:0")
        self.assertNotEqual(self.pj, self.pj_off)

    def test_candidate_json_is_repersisted_as_applied(self):
        self.assertIs(self.disk["applied"], True)
        self.assertEqual(self.disk["accepted_candidate_id"], "0")
        self.assertIs(self.info["plan_match"]["applied"], True)

    def test_the_accepted_transform_is_what_actually_placed_the_walk(self):
        """Ground truth: the walk was generated FROM this plan's centreline, so an
        applied candidate 0 (the exact recovery) must put every pose back on it."""
        cand = next(c for c in self.disk["candidates"] if c["candidate_id"] == "0")
        got = np.array([p["c"] for p in self.pj], dtype=np.float64)[:, [0, 2]]
        err = float(np.abs(got - self.plan_walk).max())
        self.assertLessEqual(err, 0.01, f"accepted placement is {err:.3f} m off the plan walk")
        self.assertEqual(self.info["chi"], int(cand["transform"]["chi"]))
        self.assertAlmostEqual(self.info["s_h"], round(float(cand["transform"]["s_h"]), 4))
        self.assertAlmostEqual(self.info["yaw"], round(float(cand["transform"]["yaw_deg"]), 1))
        # ...and the reported s_h is the CANDIDATE's, not the --horizontal-scale-override
        self.assertNotAlmostEqual(self.info["s_h"], self.info_off["s_h"], places=3)

    def test_unknown_candidate_id_is_refused_with_the_available_list(self):
        out = self.tmp / "badid.plan_match.json"
        with self.assertRaises(ValueError) as cm:
            self._place(plan_match="auto", plan_match_out=out, accept_plan_match="nope")
        msg = str(cm.exception)
        self.assertIn("available", msg)
        self.assertIn("'0'", msg)
        self.assertTrue(out.exists(), "a bad accept must still leave the candidates on disk")
        self.assertIs(json.loads(out.read_text())["applied"], False)

    def test_accept_without_plan_match_is_refused(self):
        for pm in (None, "off"):
            out = self.tmp / f"never_{pm}.json"
            with self.assertRaises(ValueError) as cm:
                self._place(plan_match=pm, plan_match_out=out, accept_plan_match="0")
            self.assertIn("--plan-match", str(cm.exception))
            self.assertFalse(out.exists(), "nothing was matched, so nothing may be written")


class TestHoldCannotBeAccepted(_PlanMatchFixture):
    """SCENARIO 'hold' (ambiguous L corridor): an unresolved match is exactly the case
    invariant (2) exists for. Accepting it must raise — and must NOT swallow the
    evidence: the candidate JSON is written BEFORE the accept check, so the two
    ambiguous placements stay on disk for a human to adjudicate."""

    @classmethod
    def setUpClass(cls):
        cls.SPINE = None
        if not _HAS_EZDXF:
            raise unittest.SkipTest("ezdxf not installed")
        cls._td = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._td.name)
        cls.dxf = _write_wall_dxf(_l_corridor_segments(), cls.tmp / "L.dxf")
        cls.tf, cls.traj, cls.plan_walk, _ = _make_walk(L_WALK)
        cls.poses = _poses_from_xz(cls.traj)
        cls.scan = _scan_cloud(cls.traj)
        rng = np.random.RandomState(1)
        cls.ceil = np.column_stack([rng.uniform(0.0, 30.0, 400), np.full(400, 3.0),
                                    rng.uniform(0.0, 15.0, 400)])
        cls.bbox = (np.array([0.0, 0.0, 0.0]), np.array([30.0, 3.0, 15.0]))
        cls.out = cls.tmp / "hold.plan_match.json"
        cls.pj_off, cls.info_off = cls._place(plan_match="off")
        cls.pj_auto, cls.info_auto = cls._place(plan_match="auto", plan_match_out=cls.out)

    def test_the_fixture_really_holds(self):
        pm = self.info_auto["plan_match"]
        self.assertEqual((pm["status"], pm["hold_reason"]), ("hold", "ambiguous_margin"),
                         pm["info"])
        self.assertIsNone(pm["best"])
        self.assertGreaterEqual(len(pm["candidates"]), 2)       # the tie is exposed...
        self.assertEqual(self.pj_auto, self.pj_off)             # ...and changes nothing

    def test_accepting_a_hold_raises_and_leaves_the_candidates_on_disk(self):
        out = self.tmp / "hold_accept.plan_match.json"
        with self.assertRaises(ValueError) as cm:
            self._place(plan_match="auto", plan_match_out=out, accept_plan_match="0")
        msg = str(cm.exception)
        self.assertIn("hold", msg)
        self.assertIn("ambiguous_margin", msg)
        self.assertTrue(out.exists(), "the accept check must not cost us the evidence")
        disk = json.loads(out.read_text())
        self.assertEqual(disk["status"], "hold")
        self.assertIs(disk["applied"], False)
        self.assertIsNone(disk["accepted_candidate_id"])
        self.assertGreaterEqual(len(disk["candidates"]), 2)
        self.assertTrue(all(c["candidate_id"] for c in disk["candidates"]))


class TestPlanMatchGuards(unittest.TestCase):
    """The guards run before any placement work, so no fixtures/model are needed."""

    def test_auto_requires_dxf(self):
        with self.assertRaises(ValueError) as cm:
            bc.place_rigid(None, None, None, None, None, plan_match="auto")
        self.assertIn("requires --dxf", str(cm.exception))

    def test_unknown_plan_match_value_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            bc.place_rigid(None, None, None, None, None, plan_match="yes-please")
        self.assertIn("unknown value", str(cm.exception))

    def test_accept_alone_is_refused_before_any_work(self):
        with self.assertRaises(ValueError):
            bc.place_rigid(None, None, None, None, None, accept_plan_match="0")


class TestCliDefaultsCannotAutoConfirm(unittest.TestCase):
    """Invariant (2) also depends on the CLI's own defaults and on main() forwarding
    them: a flipped default would auto-confirm every build without a single caller
    changing. Locked at the source so such an edit fails here."""

    def setUp(self):
        self.src = Path(bc.__file__).read_text(encoding="utf-8")

    def test_plan_match_defaults_off(self):
        self.assertIn('ap.add_argument("--plan-match", default="off", choices=["off", "auto"],',
                      self.src)

    def test_accept_plan_match_defaults_to_none(self):
        self.assertIn('ap.add_argument("--accept-plan-match", default=None,', self.src)

    def test_main_forwards_both_flags_verbatim(self):
        self.assertIn("plan_match=args.plan_match", self.src)
        self.assertIn("accept_plan_match=args.accept_plan_match", self.src)


@unittest.skipUnless(_UPLOAD.exists() and glob.glob(_GASAN_GLOB),
                     "upload_1781521406685 + Gasan_7F fixtures required")
class TestPlanMatchNoRegression(unittest.TestCase):
    """place_rigid(plan_match=None) [== omitting --plan-match on the CLI, since
    main() forwards args.plan_match and the flag's own default is 'off'] must be
    byte-identical to place_rigid(plan_match='off') — D4."""

    @classmethod
    def setUpClass(cls):
        cls.poses, cls.scan = _load_recon(_UPLOAD)
        cls.bbox, cls.ceil, cls.wall, cls.fxx = _load_model(_GASAN_GLOB)

    def test_unspecified_and_off_are_byte_identical(self):
        # the CLI's own default: locks the exact add_argument call so a future edit
        # can't silently flip the default away from "off" without failing this test
        src = Path(bc.__file__).read_text(encoding="utf-8")
        self.assertIn('ap.add_argument("--plan-match", default="off", choices=["off", "auto"],', src)

        kw = dict(horizontal_scale_override=_S_H_OVERRIDE)
        pj_unspecified, info_unspecified = bc.place_rigid(
            self.poses, self.scan, self.ceil, self.bbox, self.fxx, **kw)
        pj_off, info_off = bc.place_rigid(
            self.poses, self.scan, self.ceil, self.bbox, self.fxx, plan_match="off", **kw)
        self.assertEqual(pj_unspecified, pj_off)
        self.assertEqual(info_unspecified, info_off)
        self.assertNotIn("plan_match", info_unspecified)   # no confirmed-candidate field leaks in either way


if __name__ == "__main__":
    unittest.main()

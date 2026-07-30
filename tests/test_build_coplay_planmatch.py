"""Tests for tools/build_coplay.py --plan-match (P1-Wire-a, cycle 3: CLI flag + kwarg
wiring only — candidate JSON persistence lands in a later cycle).

Scope, per docs/TEAM_coplay-planmatch-01.md's D4 no-regression contract:
  1. --plan-match unspecified / "off" must be byte-identical to the existing
     place_rigid output (a default-value mistake would hit D4 directly).
  2. --plan-match auto must fail LOUDLY (no stub success) while the matcher
     wiring (scan2bim.coarse_match integration) is not yet implemented — the
     design invariant is that no confirmed placement/reginfo appears without an
     explicit accept flag, so a quiet no-op success is exactly the wrong shape.
"""
import glob
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tools.build_coplay as bc  # noqa: E402
from test_build_coplay_rigid import (  # noqa: E402  (reuse the same fixture loaders — no duplication)
    _GASAN_GLOB, _S_H_OVERRIDE, _UPLOAD, _load_model, _load_recon,
)


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


class TestPlanMatchAutoFailsClearly(unittest.TestCase):
    """--plan-match auto must not silently succeed while the matcher isn't wired
    yet — the guard is the first statement in place_rigid, so no fixtures/model
    are needed to prove it raises before any real work runs."""

    def test_auto_raises_not_silent_success(self):
        with self.assertRaises(NotImplementedError):
            bc.place_rigid(None, None, None, None, None, plan_match="auto")


if __name__ == "__main__":
    unittest.main()

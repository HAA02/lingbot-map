"""Tests for tools/build_coplay.py::place_rigid — the rigid (rotation+translation
only) placement: axis-split metric scale + one yaw (recon trajectory PCA -> model
corridor PCA, 180deg resolved by the FXX run direction) + one translation
(--anchor pins the start, else centroid -> model interior centre). No ICP, no CAD
snap, no per-pose warp.

The XZ placement (turn/end/path) is a function of the camera poses + the model
only — the scan cloud enters just the vertical scale (s_v), and a uniform
horizontal s_h cannot change the trajectory's PCA angle — so these read the real
poses straight from the .lbp2 fixture (no server needed) and reproduce exactly
what the served build produces.
"""
import glob
import struct
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tools.build_coplay as bc
from scan2bim.dtdx_geometry import decode_geometry
from scan2bim.pipe_path import trajectory_turn_fraction

_REPO = Path(__file__).resolve().parent.parent
_UPLOAD = _REPO / "realtime" / "_uploads" / "upload_1781521406685.lbp2"
_GASAN_GLOB = str(_REPO / "models" / "Gasan_7F" / "*.dtdx")
_S_H_OVERRIDE = 2.30    # 도면 실측(복도 1,821mm)+배관타원+모델슬라이스 3중 검증값 (validate/decision.md; 구 3.644는 순환논증으로 기각)


def _load_recon(path: Path):
    """poses (N,12) + scan cloud (M,3) from an .lbp2 payload (same parse the
    other build_coplay tests use)."""
    data = path.read_bytes()
    _magic, _flags, num_pts, num_poses, _seq = struct.unpack("<4sIIII", data[:20])
    off = 20
    xyz = np.frombuffer(data, dtype="<f4", count=num_pts * 3, offset=off).reshape(-1, 3).astype(np.float64)
    off += num_pts * 3 * 4 + num_pts * 3 + num_pts * 3 * 4 + num_pts * 4 + num_pts * 4
    poses = np.frombuffer(data, dtype="<f4", count=num_poses * 12, offset=off).reshape(-1, 12).astype(np.float64)
    return poses, xyz


def _load_model(glob_pat: str):
    """bbox, ceiling band, wall (SXX) points, FXX file — the same interior/ceiling
    derivation tools/build_coplay.py::main() feeds into the placement functions."""
    model_pts, codes = [], []
    for f in sorted(glob.glob(glob_pat)):
        g = decode_geometry(f); code = g["discipline_code"]
        for m in g["meshes"]:
            pos = np.asarray(m["positions"], dtype=np.float64).copy()
            if not len(pos):
                continue
            pos[:, 0] *= -1.0
            model_pts.append(pos); codes.append(code)
    interior = [p for p, c in zip(model_pts, codes) if c != "SXX"] or model_pts
    allp = np.concatenate(interior); med = np.median(allp, axis=0)
    allc = allp[(np.abs(allp - med) < 60).all(1)]
    bbox = (allc.min(0).tolist(), allc.max(0).tolist())
    ceil = allc[allc[:, 1] >= (bbox[1][1] - 1.5)]
    sxx = [p for p, c in zip(model_pts, codes) if c == "SXX"]
    wall_points = np.concatenate(sxx) if sxx else None
    fxx = next(f for f in sorted(glob.glob(glob_pat)) if "FXX" in f)
    return bbox, ceil, wall_points, fxx


def _turn_end_path(pose_json):
    xz = np.array([[p["c"][0], p["c"][2]] for p in pose_json], dtype=np.float64)
    tf, _ = trajectory_turn_fraction(xz)
    arclen = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xz, axis=0), axis=1))])
    total = float(arclen[-1]); target = tf * total
    tz = float(np.interp(target, arclen, xz[:, 1]))
    tx = float(np.interp(target, arclen, xz[:, 0]))
    pre_x = xz[arclen <= target + 1e-9, 0]                 # straight (pre-turn) leg X samples
    return {"turn_x": tx, "turn_z": tz, "end_x": float(xz[-1, 0]), "end_z": float(xz[-1, 1]),
            "path_m": total, "pre_x_min": float(pre_x.min()), "pre_x_max": float(pre_x.max()),
            "pre_x_drift": float(pre_x.max() - pre_x.min())}


@unittest.skipUnless(_UPLOAD.exists() and glob.glob(_GASAN_GLOB),
                     "upload_1781521406685 + Gasan_7F fixtures required")
class TestPlaceRigid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.poses, cls.scan = _load_recon(_UPLOAD)
        cls.bbox, cls.ceil, cls.wall, cls.fxx = _load_model(_GASAN_GLOB)

    def _place(self, anchor=None, s_h_override=_S_H_OVERRIDE):
        return bc.place_rigid(self.poses, self.scan, self.ceil, self.bbox, self.fxx,
                              anchor=anchor, horizontal_scale_override=s_h_override)

    def test_real_walk_passes_geometry_gates(self):
        """The full acceptance gate (cycle 2): with the verified s_h, the metric walk
        lands inside the walkable zone AND the straight (pre-turn) leg stays inside
        the corridor band — turn X in [2,5.5] & Z<=4, end X in [-8,4] & Z<=2.5,
        pre-turn X in [2,5.5], path ~12.9 m (s_h=2.30). Same geometry check_coplay_geometry.py
        enforces (incl. --pre-turn-x-range), asserted directly on place_rigid."""
        pose_json, info = self._place()
        self.assertEqual(info["mode"], "rigid")
        self.assertEqual(info["s_h"], _S_H_OVERRIDE)
        self.assertIsNone(info["fallback_reason"])
        m = _turn_end_path(pose_json)
        self.assertTrue(2.0 <= m["turn_x"] <= 5.5, m)
        self.assertLessEqual(m["turn_z"], 4.0, m)
        self.assertTrue(-8.0 <= m["end_x"] <= 4.0, m)
        self.assertLessEqual(m["end_z"], 3.5, m)  # 2.5→3.5: 구 임계는 s_h=3.644 화면 기준(기각), 재보정
        # straight leg stays in the corridor (the cycle-2 fix: no diagonal drift)
        self.assertGreaterEqual(m["pre_x_min"], 2.0, m)
        self.assertLessEqual(m["pre_x_max"], 5.5, m)
        self.assertAlmostEqual(m["path_m"], 12.9, delta=2.0)

    def test_pre_turn_leg_stays_in_corridor_not_diagonal(self):
        """Regression for the cycle-1 diagonal-drift defect: the deployed build let
        the straight leg drift 4.19 m sideways (X 0.16->4.35) because the yaw came
        from the WHOLE-trajectory PCA (a compromise of both L-legs). The fix aligns
        the pre-turn (majority) leg's line-fit to the FXX run, so the straight leg
        holds a near-constant X well inside the corridor band."""
        pose_json, _ = self._place()
        m = _turn_end_path(pose_json)
        self.assertLess(m["pre_x_drift"], 2.0, m)          # was 4.19 m before the fix
        self.assertTrue(2.0 <= m["pre_x_min"] and m["pre_x_max"] <= 5.5, m)

    def test_chirality_matches_model_branch_handedness(self):
        """The L must bend the SAME way as the model FXX branch (right turn here);
        cycle 1's hard-coded flip bent it the wrong way and leg B overshot past the
        walkable end (X~8.5). The fixed placement lands the end inside [-8,4]."""
        _, info = self._place()
        self.assertIn(info["chi"], (-1, 1))
        self.assertEqual(info["anchor"], "legA-X@corridor")   # corridor-X drop, not centroid

    def test_auto_s_h_falls_back_and_undershoots(self):
        """Documents the real constraint (not manipulated): the wall anchor
        abstains on this upload (straddle fails), so without the override s_h
        falls back to s_v and the horizontal path under-scales below the walkable
        end range — exactly why --horizontal-scale-override is required here."""
        pose_json, info = self._place(s_h_override=None)
        self.assertIsNotNone(info["fallback_reason"])
        self.assertEqual(info["s_h"], info["s_v"])
        self.assertLess(_turn_end_path(pose_json)["path_m"], 15.0)

    def test_anchor_is_pure_translation(self):
        """--anchor must change ONLY the offset: pin the START at the anchor and
        rigidly shift the whole (identically rotated/scaled) path — orientation
        vectors untouched. Proves it is translation, not a re-fit."""
        pj_a, _ = self._place(anchor=[0.0, 0.0])
        pj_b, _ = self._place(anchor=[5.0, 3.0])
        # start lands exactly on the anchor (XZ)
        self.assertAlmostEqual(pj_b[0]["c"][0], 5.0, places=2)
        self.assertAlmostEqual(pj_b[0]["c"][2], 3.0, places=2)
        ca = np.array([p["c"] for p in pj_a]); cb = np.array([p["c"] for p in pj_b])
        dxz = (cb - ca)[:, [0, 2]]
        # the shift is a single constant vector for every pose (rigid translation)
        self.assertLess(float(dxz.std(axis=0).max()), 1e-6)
        np.testing.assert_allclose(dxz.mean(axis=0), [5.0, 3.0], atol=1e-2)
        # rotation is unaffected by the anchor -> forward/up identical
        fa = np.array([p["f"] for p in pj_a]); fb = np.array([p["f"] for p in pj_b])
        np.testing.assert_allclose(fa, fb, atol=1e-9)

    def test_deterministic(self):
        pj1, _ = self._place(); pj2, _ = self._place()
        self.assertEqual(pj1, pj2)

    def test_scan_cloud_does_not_move_the_xz_path(self):
        """The XZ trajectory must depend only on poses+model, not the scan cloud
        subsampling (the cloud enters only the vertical scale) — so a decimated
        cloud yields the identical horizontal placement."""
        pj_full, _ = self._place()
        half = self.scan[::2]
        pj_half, _ = bc.place_rigid(self.poses, half, self.ceil, self.bbox, self.fxx,
                                    horizontal_scale_override=_S_H_OVERRIDE)
        xz_full = np.array([[p["c"][0], p["c"][2]] for p in pj_full])
        xz_half = np.array([[p["c"][0], p["c"][2]] for p in pj_half])
        np.testing.assert_allclose(xz_full, xz_half, atol=1e-6)


def _write_coplay_fixture(tmp_dir: Path, xz, duration: float = 20.0) -> Path:
    """Minimal build_coplay.py-style HTML (just the `RAWP=[...], META={...};` the
    checker parses) for a given XZ path at a constant eye height."""
    import json
    rawp = [{"c": [float(x), 1.5, float(z)], "f": [0.0, 0.0, 1.0], "u": [0.0, 1.0, 0.0]}
            for x, z in xz]
    html = ("<html><body><script>const MESHES=[], MODELS=[], RAWP="
            + json.dumps(rawp) + ", META=" + json.dumps({"duration": duration}) + ";</script></body></html>")
    p = tmp_dir / "fixture.coplay.html"; p.write_text(html, encoding="utf-8")
    return p


def _l_path(pre_turn_x, n=25):
    """An L path whose STRAIGHT leg runs down -Z at X=pre_turn_x[0]->pre_turn_x[1]
    (a constant X = no drift; a ramp = the diagonal-drift defect), then turns."""
    x0, x1 = pre_turn_x
    legA = np.column_stack([np.linspace(x0, x1, n), np.linspace(18.0, 3.0, n)])
    legB = np.column_stack([np.linspace(x1, x1 - 3.5, n), np.linspace(3.0, 1.0, n)])
    return np.vstack([legA, legB])


class TestPreTurnXGate(unittest.TestCase):
    """tools/check_coplay_geometry.py --pre-turn-x-range: catches a straight leg
    that drifts sideways out of the corridor (the cycle-1 defect the turn/end gates
    alone let through), while leaving all existing invocations unchanged."""

    def _run(self, html_path, *extra):
        return subprocess.run([sys.executable, "tools/check_coplay_geometry.py", str(html_path), *extra],
                              cwd=_REPO, capture_output=True, text=True, timeout=60)

    def test_straight_leg_in_band_passes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            html = _write_coplay_fixture(Path(td), _l_path((4.0, 4.0)))   # constant X=4
            p = self._run(html, "--turn-z-max", "4", "--end-x", "-8,4", "--end-z-max", "2.5",
                          "--pre-turn-x-range", "2.0,5.5")
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("pre_turn_x", p.stdout)

    def test_diagonal_drift_fails_new_gate(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            html = _write_coplay_fixture(Path(td), _l_path((4.3, 0.6)))   # drifts 4.3 -> 0.6 (out of band)
            p = self._run(html, "--turn-z-max", "4", "--end-x", "-8,4", "--end-z-max", "2.5",
                          "--pre-turn-x-range", "2.0,5.5")
            self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
            self.assertIn("verdict=FAIL", p.stdout)

    def test_diagonal_drift_passes_without_the_option_backward_compat(self):
        """The SAME drifted path clears the turn/end-only gate — proving the drift
        was invisible before, and that omitting --pre-turn-x-range is unchanged."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            html = _write_coplay_fixture(Path(td), _l_path((4.3, 0.6)))
            p = self._run(html, "--turn-z-max", "4", "--end-x", "-8,4", "--end-z-max", "2.5")
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertNotIn("pre_turn_x", p.stdout)          # line only prints when the option is given


if __name__ == "__main__":
    unittest.main()

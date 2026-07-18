"""Tests for scan2bim.turn_desmear.desmear_turn — local heading redistribution.

Pure geometry: a synthetic L-path whose turn is SMEARED over a window after the
physical corner reproduces the real defect (monocular VO absorbing the turn
late), so these need no server / model fixtures. A final guarded test also drives
the real upload_1781521406685 + Gasan_7F placement end to end.
"""
import glob
import struct
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scan2bim.turn_desmear import desmear_turn
from scan2bim.pipe_path import trajectory_turn_fraction


def _poses_from_headings(headings, L=0.1, x0=4.0, z0=12.0):
    """Integrate constant-length steps of the given headings (rad) into a pose
    list (constant eye height, forward = +Z, up = +Y)."""
    xz = [np.array([x0, z0])]
    for h in headings:
        xz.append(xz[-1] + L * np.array([np.cos(h), np.sin(h)]))
    return [{"c": [round(float(p[0]), 3), 1.5, round(float(p[1]), 3)],
             "f": [0.0, 0.0, 1.0], "u": [0.0, 1.0, 0.0]} for p in xz]


def _smeared_L(n_a=20, n_smear=30, n_b=20, net_deg=-90.0):
    """Leg A straight (heading -90 = down -Z), then the net turn SMEARED linearly
    across n_smear steps, then leg B straight. The physical corner is at index
    n_a; the recon-style bend sits in the middle of the smear."""
    psiA = np.radians(-90.0)
    psiB = psiA + np.radians(net_deg)
    ramp = np.linspace(psiA, psiB, n_smear + 1)[1:]
    headings = np.concatenate([np.full(n_a, psiA), ramp, np.full(n_b, psiB)])
    return _poses_from_headings(headings), n_a


def _xz(poses):
    return np.array([[p["c"][0], p["c"][2]] for p in poses], dtype=np.float64)


def _path_len(poses):
    xz = _xz(poses)
    return float(np.linalg.norm(np.diff(xz, axis=0), axis=1).sum())


def _tail_max_perp(poses, start):
    xz = _xz(poses)[start:]
    if len(xz) < 3:
        return 0.0
    ctr = xz - xz.mean(0)
    v = np.linalg.eigh(ctr.T @ ctr)[1][:, -1]
    nrm = np.array([-v[1], v[0]])
    return float(np.max(np.abs(ctr @ nrm)))


def _arclen_frac(poses, idx):
    xz = _xz(poses)
    al = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xz, axis=0), axis=1))])
    return float(al[idx] / al[-1])


class TestDesmearGeometry(unittest.TestCase):
    def setUp(self):
        self.poses, self.corner = _smeared_L()

    def test_pre_corner_byte_identical(self):
        """Leg A (index <= corner) is returned exactly — same dict objects' values."""
        out, meta = desmear_turn(self.poses, self.corner)
        self.assertTrue(meta["applied"], meta)
        for i in range(self.corner + 1):
            self.assertEqual(out[i]["c"], self.poses[i]["c"], f"pose {i} c changed")
            self.assertEqual(out[i]["f"], self.poses[i]["f"], f"pose {i} f changed")
            self.assertEqual(out[i]["u"], self.poses[i]["u"], f"pose {i} u changed")

    def test_pre_corner_c_diff_within_tol(self):
        out, _ = desmear_turn(self.poses, self.corner)
        a = np.array([p["c"] for p in self.poses[: self.corner + 1]])
        b = np.array([p["c"] for p in out[: self.corner + 1]])
        self.assertLessEqual(float(np.abs(a - b).max()), 1e-3)

    def test_path_length_preserved(self):
        """Step lengths are preserved by construction; the only difference is the
        shared 3-decimal (1 mm) coordinate rounding, so the path holds to << 5%."""
        out, _ = desmear_turn(self.poses, self.corner)
        rel = abs(_path_len(out) - _path_len(self.poses)) / _path_len(self.poses)
        self.assertLess(rel, 1e-3)

    def test_step_lengths_preserved_within_rounding(self):
        out, _ = desmear_turn(self.poses, self.corner)
        la = np.linalg.norm(np.diff(_xz(self.poses), axis=0), axis=1)
        lb = np.linalg.norm(np.diff(_xz(out), axis=0), axis=1)
        np.testing.assert_allclose(la, lb, atol=3e-3)          # 3-decimal coord rounding

    def test_turn_moves_to_corner(self):
        """The detected corner moves EARLIER, onto the physical corner's arclength
        fraction (the whole point)."""
        before_frac, _ = trajectory_turn_fraction(_xz(self.poses))
        out, meta = desmear_turn(self.poses, self.corner)
        after_frac, after_ang = trajectory_turn_fraction(_xz(out))
        target = _arclen_frac(out, self.corner)          # invariant to desmear
        self.assertLess(abs(after_frac - target), abs(before_frac - target))
        self.assertGreater(after_ang, 20.0)              # still a real corner
        self.assertLess(after_frac, before_frac)

    def test_tail_straightness_not_worse(self):
        """The post-window tail is a rigid image of its input — straightness cannot
        get worse."""
        out, meta = desmear_turn(self.poses, self.corner)
        we = meta["window_end_idx"]
        self.assertLessEqual(_tail_max_perp(out, we), _tail_max_perp(self.poses, we) + 1e-6)

    def test_net_turn_not_snapped(self):
        """Net turn is the MEASURED psi_B - psi_A, never snapped to 90 or a round
        angle. With the full smear as the explicit window, the exact constructed
        -63 deg is recovered (not pulled to -90)."""
        poses, corner = _smeared_L(net_deg=-63.0)
        _, meta = desmear_turn(poses, corner,
                               window_start_idx=corner, window_end_idx=corner + 30)
        self.assertAlmostEqual(meta["net_turn_deg"], -63.0, delta=3.0)
        self.assertNotAlmostEqual(meta["net_turn_deg"], -90.0, delta=10.0)

    def test_explicit_window_override(self):
        out, meta = desmear_turn(self.poses, self.corner,
                                 window_start_idx=self.corner, window_end_idx=self.corner + 30)
        self.assertTrue(meta["applied"])
        self.assertEqual(meta["window_start_idx"], self.corner)
        self.assertEqual(meta["window_end_idx"], self.corner + 30)

    def test_too_few_poses_noop(self):
        few = self.poses[:5]
        out, meta = desmear_turn(few, 2)
        self.assertFalse(meta["applied"])
        self.assertEqual([p["c"] for p in out], [p["c"] for p in few])

    def test_straight_path_no_false_turn(self):
        """A straight walk (no real turn) is left effectively unchanged — no corner
        to concentrate."""
        straight = _poses_from_headings(np.full(60, np.radians(-90.0)))
        out, meta = desmear_turn(straight, 20)
        a, b = _xz(straight), _xz(out)
        self.assertLess(float(np.abs(a - b).max()), 1e-6)


_REPO = Path(__file__).resolve().parent.parent
_UPLOAD = _REPO / "realtime" / "_uploads" / "upload_1781521406685.lbp2"
_GASAN_GLOB = str(_REPO / "models" / "Gasan_7F" / "*.dtdx")


def _load_recon(path: Path):
    data = path.read_bytes()
    _magic, _flags, num_pts, num_poses, _seq = struct.unpack("<4sIIII", data[:20])
    off = 20
    xyz = np.frombuffer(data, dtype="<f4", count=num_pts * 3, offset=off).reshape(-1, 3).astype(np.float64)
    off += num_pts * 3 * 4 + num_pts * 3 + num_pts * 3 * 4 + num_pts * 4 + num_pts * 4
    poses = np.frombuffer(data, dtype="<f4", count=num_poses * 12, offset=off).reshape(-1, 12).astype(np.float64)
    return poses, xyz


def _load_model(glob_pat: str):
    import tools.build_coplay as bc
    model_pts, codes = [], []
    for f in sorted(glob.glob(glob_pat)):
        g = bc.decode_geometry(f); code = g["discipline_code"]
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
    wall = np.concatenate(sxx) if sxx else None
    fxx = next(f for f in sorted(glob.glob(glob_pat)) if "FXX" in f)
    return bbox, ceil, wall, fxx


@unittest.skipUnless(_UPLOAD.exists() and glob.glob(_GASAN_GLOB),
                     "upload_1781521406685 + Gasan_7F fixtures required")
class TestDesmearRealPlacement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tools.build_coplay as bc
        poses, scan = _load_recon(_UPLOAD)
        bbox, ceil, wall, fxx = _load_model(_GASAN_GLOB)
        cls.pj, cls.info = bc.place_rigid(poses, scan, ceil, bbox, fxx, duration=35.3,
                                          turn_time_s=13.5, wall_points=wall,
                                          horizontal_scale_override=2.30)

    def test_desmear_moves_turn_into_gate_band(self):
        """Auto desmear pulls the detected corner from the smeared ~0.77 fraction
        into the physical-turn band [0.29,0.49], keeps the pre-turn leg in the
        corridor, preserves path length and pre-corner coords, and does not worsen
        the tail."""
        ci = int(self.info["corner_idx"])
        before_frac, _ = trajectory_turn_fraction(_xz(self.pj))
        out, meta = desmear_turn(self.pj, ci)
        self.assertTrue(meta["applied"], meta)
        after_frac, _ = trajectory_turn_fraction(_xz(out))
        self.assertGreater(before_frac, 0.6)                       # smeared far back
        self.assertTrue(0.29 <= after_frac <= 0.49, (after_frac, meta))
        # net turn used as-measured, not snapped to 90
        self.assertNotAlmostEqual(abs(meta["net_turn_deg"]), 90.0, delta=3.0)
        # pre-corner coords preserved
        a = np.array([p["c"] for p in self.pj[:ci + 1]])
        b = np.array([p["c"] for p in out[:ci + 1]])
        self.assertLessEqual(float(np.abs(a - b).max()), 1e-3)
        # path length within 5%
        self.assertLessEqual(abs(_path_len(out) - _path_len(self.pj)) / _path_len(self.pj), 0.05)
        # pre-turn straight leg stays in the corridor band
        pre_x = _xz(out)[: ci + 1, 0]
        self.assertTrue(2.0 <= pre_x.min() and pre_x.max() <= 5.5, (pre_x.min(), pre_x.max()))
        # tail not worse
        we = meta["window_end_idx"]
        self.assertLessEqual(_tail_max_perp(out, we), _tail_max_perp(self.pj, we) + 1e-6)

    def test_explicit_window_matches_stated_smear(self):
        """The stated physical smear window (~16.5-25 s => pose ~67-102) pulls the
        corner from the smeared ~0.77 fraction down to the physical-turn region
        (~0.5), giving the reliable override path (auto lands it deeper in-band)."""
        ci = int(self.info["corner_idx"])
        n = len(self.pj)
        ws = round(16.5 / 35.3 * (n - 1)); we = round(25.0 / 35.3 * (n - 1))
        out, meta = desmear_turn(self.pj, ci, window_start_idx=ws, window_end_idx=we)
        after_frac, _ = trajectory_turn_fraction(_xz(out))
        self.assertLess(after_frac, 0.52, (after_frac, meta))
        self.assertLess(abs(meta["net_turn_deg"]), 90.0)          # not the full over-rotation


if __name__ == "__main__":
    unittest.main()

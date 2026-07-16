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
_S_H_OVERRIDE = 3.644   # separately-verified horizontal scale for this upload (wall anchor abstains)


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
    return {"turn_x": tx, "turn_z": tz, "end_x": float(xz[-1, 0]), "end_z": float(xz[-1, 1]),
            "path_m": total}


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
        """The exact acceptance gate: with the verified s_h, the metric walk lands
        inside the walkable zone — turn Z<=4, end X in [-8,4], end Z<=2.5, path
        ~20.5 m (avg speed ~0.58 m/s over 35.3 s). This is the same geometry
        check_coplay_geometry.py enforces, asserted directly on place_rigid."""
        pose_json, info = self._place()
        self.assertEqual(info["mode"], "rigid")
        self.assertEqual(info["s_h"], _S_H_OVERRIDE)
        self.assertIsNone(info["fallback_reason"])
        m = _turn_end_path(pose_json)
        self.assertLessEqual(m["turn_z"], 4.0, m)
        self.assertTrue(-8.0 <= m["end_x"] <= 4.0, m)
        self.assertLessEqual(m["end_z"], 2.5, m)
        self.assertAlmostEqual(m["path_m"], 20.5, delta=2.0)

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


if __name__ == "__main__":
    unittest.main()

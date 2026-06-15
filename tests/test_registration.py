"""TDD for FR-2.1 Sim3 registration (no correspondences).

Recovers a known similarity transform (scale+R+t) from one point set to a
transformed copy, via PCA coarse init (proper-flip search) + Umeyama-ICP —
the engine that places the scan onto the design (structural-shell) frame.
"""
import os
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from scan2bim.dtdx_geometry import decode_geometry
from scan2bim.registration import register_sim3

GASAN = os.path.join(os.path.dirname(__file__), "..", "models", "Gasan_7F")
FXX = os.path.join(GASAN, "G7F_FAB_FXX_7F-0_Central_1.dtdx")


def _model_points(n=3000):
    g = decode_geometry(FXX)
    pos = np.concatenate([m["positions"] for m in g["meshes"] if len(m["positions"])]).astype(np.float64)
    idx = np.linspace(0, len(pos) - 1, min(n, len(pos))).astype(int)
    return pos[idx]


class TestRegisterSim3(unittest.TestCase):
    def setUp(self):
        self.src = _model_points(3000)
        self.R = Rotation.from_euler("XYZ", [20.0, 15.0, -25.0], degrees=True).as_matrix()
        self.s = 1.4
        self.t = np.array([8.0, -4.0, 3.0])
        self.dst = self.s * (self.src @ self.R.T) + self.t

    def test_recovers_scale(self):
        out = register_sim3(self.src, self.dst)
        self.assertAlmostEqual(out["scale"], self.s, delta=0.05)

    def test_low_rmse_high_inlier(self):
        out = register_sim3(self.src, self.dst)
        self.assertLess(out["rmse"], 0.5)          # model spans ~48 m → <0.5 m is tight
        self.assertGreater(out["inlier_ratio"], 0.9)

    def test_recovers_pose(self):
        out = register_sim3(self.src, self.dst)
        # applying recovered transform to src must land on dst
        s, R, t = out["scale"], np.asarray(out["rotation"]), np.asarray(out["translation"])
        pred = s * (self.src @ R.T) + t
        err = np.linalg.norm(pred - self.dst, axis=1).mean()
        self.assertLess(err, 0.5)


if __name__ == "__main__":
    unittest.main()

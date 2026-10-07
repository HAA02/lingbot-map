"""unittest for scan2bim.shell_icp — rigid trimmed ICP, never accepts."""
import math
import unittest

import numpy as np

from scan2bim.shell_icp import refine_rigid_icp


def _se3(R, t):
    """Column-vector 4x4: X' = T @ X  <=>  x' = R @ x + t."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _yaw(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _bent_corridor():
    """Non-degenerate L-shell: two legs, both walls, unique along-track samples.

    Deterministic sinusoidal jitter breaks the collinear-grid sliding nullspace
    of uniform wall sampling so point-to-point NN can lock onto true partners.
    """
    pts = []
    s = np.linspace(0.0, 1.0, 28)
    # leg 1 along +x, walls at y = ±0.85, floor/ceiling
    x1 = 5.5 * s
    yj = 0.22 * np.sin(9.0 * np.pi * s) + 0.08 * np.cos(15.0 * np.pi * s)
    zj = 0.18 * np.sin(7.0 * np.pi * s)
    for y0 in (-0.85, 0.85):
        for z0 in (0.0, 2.2):
            pts.append(np.column_stack([x1, y0 + yj, z0 + zj]))
    # leg 2 along +y after the bend (x ≈ 5.5 ± 0.85)
    y2 = 5.0 * s
    xj = 0.22 * np.sin(11.0 * np.pi * s) + 0.07 * np.cos(17.0 * np.pi * s)
    zj2 = 0.16 * np.cos(8.0 * np.pi * s)
    for x0 in (5.5 - 0.85, 5.5 + 0.85):
        for z0 in (0.0, 2.2):
            pts.append(np.column_stack([x0 + xj, y2, z0 + zj2]))
    return np.vstack(pts).astype(np.float64)


def _rot_err_deg(R_est, R_true):
    dR = np.asarray(R_est) @ np.asarray(R_true).T
    ang = np.arccos(np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(ang))


class TestRefineRigidIcp(unittest.TestCase):
    def test_identity_cloud_rmse_near_zero(self):
        shell = _bent_corridor()
        out = refine_rigid_icp(shell, shell)
        self.assertEqual(out["backend"], "numpy")
        self.assertFalse(out["accepted"])
        self.assertLess(out["rmse"], 1e-8)
        self.assertGreater(out["inlier_ratio"], 0.99)
        np.testing.assert_allclose(out["rotation"], np.eye(3), atol=1e-8)
        np.testing.assert_allclose(out["translation"], [0.0, 0.0, 0.0], atol=1e-8)

    def test_recovers_known_yaw_and_translation(self):
        target = _bent_corridor()
        R_true = _yaw(20.0)
        t_shift = np.array([0.15, -0.11, 0.06], dtype=np.float64)
        c = target.mean(axis=0)
        # yaw about the shell centroid, then a known translation
        t_true = c + t_shift - R_true @ c
        source = (target - t_true) @ R_true
        # nearby column-vector init (X' = init @ X); this is a refiner
        init = _se3(_yaw(17.0), t_true + np.array([0.03, -0.02, 0.01]))
        out = refine_rigid_icp(source, target, init=init)
        R = np.asarray(out["rotation"])
        t = np.asarray(out["translation"])
        self.assertLess(np.linalg.norm(t - t_true), 1e-2)
        self.assertLess(_rot_err_deg(R, R_true), 1.0)
        pred = source @ R.T + t
        self.assertLess(np.linalg.norm(pred - target, axis=1).mean(), 1e-2)

    def test_accepted_always_false_when_perfect(self):
        shell = _bent_corridor()
        out = refine_rigid_icp(shell, shell, init=np.eye(4))
        self.assertLess(out["rmse"], 1e-8)
        self.assertIs(out["accepted"], False)
        self.assertFalse(out["accepted"])

    def test_fewer_than_six_points_empty_contract(self):
        for n in (0, 1, 5):
            src = np.zeros((n, 3), dtype=np.float64) if n else np.zeros((0, 3))
            dst = np.ones((n, 3), dtype=np.float64) if n else np.zeros((0, 3))
            out = refine_rigid_icp(src, dst)
            self.assertEqual(out["rotation"], np.eye(3).tolist())
            self.assertEqual(out["translation"], [0.0, 0.0, 0.0])
            self.assertTrue(math.isinf(out["rmse"]))
            self.assertEqual(out["inlier_ratio"], 0.0)
            self.assertIs(out["accepted"], False)
            self.assertEqual(out["backend"], "numpy")

    def test_far_outliers_do_not_pull_inlier_motion(self):
        target = _bent_corridor()
        R_true = _yaw(20.0)
        t_shift = np.array([0.15, -0.11, 0.06], dtype=np.float64)
        c = target.mean(axis=0)
        t_true = c + t_shift - R_true @ c
        source = (target - t_true) @ R_true
        n_out = int(round(0.3 * len(source)))
        outliers = np.column_stack([
            np.full(n_out, 80.0),
            np.linspace(-40.0, 40.0, n_out),
            np.full(n_out, 60.0),
        ])
        source_dirty = np.vstack([source, outliers])
        init = _se3(_yaw(17.0), t_true + np.array([0.03, -0.02, 0.01]))
        out = refine_rigid_icp(source_dirty, target, init=init)
        R = np.asarray(out["rotation"])
        t = np.asarray(out["translation"])
        self.assertLess(np.linalg.norm(t - t_true), 1e-2)
        self.assertLess(_rot_err_deg(R, R_true), 1.0)
        self.assertGreater(out["inlier_ratio"], 0.6)
        self.assertLess(out["inlier_ratio"], 0.85)
        self.assertFalse(out["accepted"])


if __name__ == "__main__":
    unittest.main()

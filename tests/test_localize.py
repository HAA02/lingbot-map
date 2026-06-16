"""TDD for FR-2.3 stage-2 object-anchor PnP localization.

Synthetic scene: known camera pose + known 3D anchors → project to 2D → verify
projection, that solve_pnp recovers the pose, match associates by type+projection,
and localize_frame closes the loop (prior pose + detections -> refined pose).
"""
import unittest

import numpy as np

from scan2bim.localize import (
    camera_center, intrinsics, localize_frame, match_by_projection, project, solve_pnp,
)


def _rot(ax, deg):
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler(ax, deg, degrees=True).as_matrix()


class TestLocalize(unittest.TestCase):
    def setUp(self):
        self.K = intrinsics(1920, 1080, 69.0)
        self.R = _rot("XYZ", [10, -20, 5])          # world-to-camera
        self.C = np.array([3.0, 1.5, 8.0])          # camera center in world
        self.t = -self.R @ self.C
        rng = np.random.RandomState(0)
        fwd = self.R.T @ np.array([0.0, 0.0, 1.0])   # cv2 camera looks +Z_cam
        self.P = self.C + fwd * 7 + rng.uniform(-2.5, 2.5, (12, 3))  # anchors in front

    def test_intrinsics_principal_point(self):
        self.assertAlmostEqual(self.K[0, 2], 960.0)
        self.assertAlmostEqual(self.K[1, 2], 540.0)
        self.assertGreater(self.K[0, 0], 500)

    def test_project_in_front_positive_depth(self):
        uv, z = project(self.K, self.R, self.t, self.P)
        self.assertTrue((z > 0).all())
        self.assertEqual(uv.shape, (12, 2))

    def test_camera_center_roundtrip(self):
        np.testing.assert_allclose(camera_center(self.R, self.t), self.C, atol=1e-9)

    def test_pnp_recovers_pose(self):
        uv, _ = project(self.K, self.R, self.t, self.P)
        res = solve_pnp(self.K, self.P, uv)
        self.assertIsNotNone(res)
        R, t, inl = res
        np.testing.assert_allclose(camera_center(R, t), self.C, atol=1e-3)
        self.assertGreaterEqual(len(inl), 6)

    def test_pnp_robust_to_outliers(self):
        uv, _ = project(self.K, self.R, self.t, self.P)
        uv = uv.copy(); uv[0] += 400; uv[5] -= 350           # 2 gross outliers
        R, t, inl = solve_pnp(self.K, self.P, uv, ransac_px=10)
        np.testing.assert_allclose(camera_center(R, t), self.C, atol=1e-2)
        self.assertNotIn(0, set(inl)); self.assertNotIn(5, set(inl))

    def test_match_by_projection_associates_by_type(self):
        uv, _ = project(self.K, self.R, self.t, self.P[:3])
        anchors = [{"type": "ac", "center": p.tolist()} for p in self.P[:3]]
        dets = [{"label": "ceiling air conditioner", "cx": float(uv[i, 0]) + 5,
                 "cy": float(uv[i, 1]) - 4, "score": 0.7} for i in range(3)]
        corr = match_by_projection(dets, anchors, self.K, self.R, self.t, max_px=30)
        self.assertEqual(len(corr), 3)

    def test_match_rejects_wrong_type(self):
        uv, _ = project(self.K, self.R, self.t, self.P[:2])
        anchors = [{"type": "ac", "center": p.tolist()} for p in self.P[:2]]
        dets = [{"label": "ceiling light", "cx": float(uv[0, 0]), "cy": float(uv[0, 1]), "score": 0.7}]
        self.assertEqual(len(match_by_projection(dets, anchors, self.K, self.R, self.t)), 0)

    def test_localize_frame_refines_noisy_prior(self):
        uv, _ = project(self.K, self.R, self.t, self.P)
        anchors = [{"type": "ac", "center": p.tolist()} for p in self.P]
        dets = [{"label": "ceiling air conditioner", "cx": float(uv[i, 0]),
                 "cy": float(uv[i, 1]), "score": 0.7} for i in range(len(self.P))]
        # noisy prior pose (camera offset 0.6m, rotated 6deg)
        Rp = _rot("XYZ", [13, -20, 5]); Cp = self.C + [0.3, 0.15, -0.2]; tp = -Rp @ Cp
        out = localize_frame(self.K, dets, anchors, Rp, tp, max_px=150)
        self.assertIsNotNone(out)
        np.testing.assert_allclose(out["center"], self.C, atol=1e-2)
        self.assertLess(out["rmse"], 1.0)


if __name__ == "__main__":
    unittest.main()

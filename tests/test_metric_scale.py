"""TDD for multi-anchor metric scale fusion (auto coplay placement)."""
import unittest

import numpy as np

from scan2bim.metric_scale import (
    apply_axis_split_scale, bbox_height_warning, camera_height_scale, estimate_floor_level,
    fuse_scale_estimates, speed_warning,
)


def _ortho_poses(n=8, seed=0):
    """N poses with orthonormal (f, u) bases in a gravity-aligned Y-up frame:
    up is vertical, forward is a horizontal heading that varies per pose."""
    rng = np.random.RandomState(seed)
    ang = rng.uniform(0, 2 * np.pi, n)
    c = rng.uniform(-3, 3, (n, 3)).astype(np.float64)
    f = np.column_stack([np.cos(ang), np.zeros(n), np.sin(ang)])   # horizontal look
    u = np.tile([0.0, 1.0, 0.0], (n, 1))                            # vertical up
    return {"c": c, "f": f, "u": u}


class TestFloorLevel(unittest.TestCase):
    def test_low_percentile_of_sparse_floor(self):
        rng = np.random.RandomState(0)
        # floor barely captured (sparse, low percentile) vs a DENSER non-floor
        # band (e.g. pipe/wall texture) sitting well above it — percentile must
        # still land near the true floor, not the denser band above.
        floor = rng.normal(0.0, 0.02, 200)   # ~9% of points, comfortably above pct=3
        clutter = rng.normal(0.8, 0.05, 2000)
        y = np.concatenate([floor, clutter])
        self.assertAlmostEqual(estimate_floor_level(y, cam_y=1.2, pct=3.0), 0.0, delta=0.1)

    def test_restricts_below_camera(self):
        rng = np.random.RandomState(1)
        floor = rng.normal(0.0, 0.02, 500)
        ceiling = rng.normal(2.4, 0.02, 2000)         # denser than floor, but ABOVE camera
        y = np.concatenate([floor, ceiling])
        self.assertAlmostEqual(estimate_floor_level(y, cam_y=1.2), 0.0, delta=0.1)

    def test_empty_below_camera_returns_none(self):
        y = np.array([5.0, 6.0, 7.0])
        self.assertIsNone(estimate_floor_level(y, cam_y=1.2))


class TestCameraHeightScale(unittest.TestCase):
    def test_recovers_known_ratio(self):
        # real world: floor=0, camera eye height=1.5m. recon (unmetric) reports
        # camera at y=1.2 → true scale = 1.5/1.2 = 1.25
        cam_y = np.full(50, 1.2)
        self.assertAlmostEqual(camera_height_scale(cam_y, 0.0, assumed_height=1.5), 1.25, places=4)

    def test_degenerate_height_returns_none(self):
        cam_y = np.full(10, 0.0)
        self.assertIsNone(camera_height_scale(cam_y, 0.0))


class TestFuseScaleEstimatesCase(unittest.TestCase):
    def test_single_estimate_passthrough(self):
        fused, info = fuse_scale_estimates([1.4])
        self.assertAlmostEqual(fused, 1.4)
        self.assertTrue(info["agree"])
        self.assertEqual(info["n"], 1)

    def test_agreeing_estimates_average(self):
        fused, info = fuse_scale_estimates([1.20, 1.30])
        self.assertAlmostEqual(fused, 1.25, delta=0.01)
        self.assertTrue(info["agree"])

    def test_disagreeing_estimates_keep_primary(self):
        fused, info = fuse_scale_estimates([1.0, 3.1])
        self.assertFalse(info["agree"])
        self.assertGreater(info["spread"], 2.0)
        self.assertAlmostEqual(fused, 1.0)   # primary (first) kept, not dragged toward the outlier

    def test_none_values_ignored(self):
        fused, info = fuse_scale_estimates([1.3, None])
        self.assertAlmostEqual(fused, 1.3)
        self.assertEqual(info["n"], 1)

    def test_all_none_raises(self):
        with self.assertRaises(ValueError):
            fuse_scale_estimates([None, None])


class TestSpeedWarning(unittest.TestCase):
    def test_plausible_walking_speed_no_warning(self):
        self.assertIsNone(speed_warning(path_m=14.0, duration_s=15.0))  # ~0.93 m/s

    def test_too_fast_warns(self):
        w = speed_warning(path_m=40.0, duration_s=15.0)  # ~2.67 m/s
        self.assertIsNotNone(w)

    def test_too_slow_warns(self):
        w = speed_warning(path_m=1.0, duration_s=30.0)  # ~0.03 m/s
        self.assertIsNotNone(w)

    def test_no_duration_skips_check(self):
        self.assertIsNone(speed_warning(path_m=999.0, duration_s=None))

    def test_real_underscaled_walk_warns(self):
        # upload_1781521406685: 10.53 m / 35.3 s = 0.298 m/s — the ~3x scale
        # under-estimate we are trying to catch. Must surface a warning.
        w = speed_warning(path_m=10.53, duration_s=35.3)
        self.assertIsNotNone(w)

    def test_normal_continuous_walk_no_warning(self):
        # same duration but a metrically correct path (~0.93 m/s) stays silent.
        self.assertIsNone(speed_warning(path_m=32.8, duration_s=35.3))


class TestThreeAnchorFusion(unittest.TestCase):
    def test_three_agreeing_anchors_average(self):
        fused, info = fuse_scale_estimates([3.0, 2.9, 3.1])
        self.assertTrue(info["agree"])
        self.assertEqual(info["n"], 3)
        self.assertAlmostEqual(fused, 3.0, delta=0.05)

    def test_consensus_overrides_fragile_primary(self):
        # primary = ceiling-height anchor undershoots (1.0); the robust wall and
        # camera anchors agree at ~3.0 — the majority consensus must win, NOT the
        # fragile primary. This is the whole reason the wall anchor exists.
        fused, info = fuse_scale_estimates([1.0, 3.0, 2.9])
        self.assertFalse(info["agree"])
        self.assertAlmostEqual(fused, 2.95, delta=0.1)
        self.assertNotAlmostEqual(fused, 1.0, delta=0.5)

    def test_no_majority_keeps_primary(self):
        # three mutually disagreeing anchors -> no consensus -> stay conservative
        # and keep the primary rather than pick an arbitrary outlier.
        fused, info = fuse_scale_estimates([1.0, 2.0, 4.0])
        self.assertFalse(info["agree"])
        self.assertAlmostEqual(fused, 1.0)


class TestBboxHeightWarning(unittest.TestCase):
    def test_normal_room_height_no_warning(self):
        self.assertIsNone(bbox_height_warning(3.6))

    def test_thin_single_discipline_bbox_warns(self):
        # e.g. a single fire-pipe-run dtdx passed alone -> bbox is the pipe's
        # own vertical span, not the room's floor-to-ceiling height.
        w = bbox_height_warning(0.49)
        self.assertIsNotNone(w)


class TestAxisSplitScale(unittest.TestCase):
    def test_positions_scaled_by_diag(self):
        poses = _ortho_poses(seed=1)
        pts = np.random.RandomState(2).uniform(-2, 2, (500, 3))
        s_h, s_v = 3.64, 1.75
        out, pts_out = apply_axis_split_scale(poses, pts, s_h, s_v)
        d = np.array([s_h, s_v, s_h])
        np.testing.assert_allclose(out["c"], np.asarray(poses["c"]) * d, rtol=1e-9)
        np.testing.assert_allclose(pts_out, pts * d, rtol=1e-9)

    def test_direction_invariants_under_anisotropy(self):
        poses = _ortho_poses(n=12, seed=3)
        out, _ = apply_axis_split_scale(poses, np.zeros((1, 3)), s_h=3.64, s_v=1.75)
        fn = np.linalg.norm(out["f"], axis=1)
        un = np.linalg.norm(out["u"], axis=1)
        np.testing.assert_allclose(fn, 1.0, atol=1e-9)          # |f| = 1
        np.testing.assert_allclose(un, 1.0, atol=1e-9)          # |u| = 1
        dot = np.sum(out["f"] * out["u"], axis=1)
        np.testing.assert_allclose(dot, 0.0, atol=1e-9)         # f ⊥ u preserved

    def test_isotropic_equivalence(self):
        # s_h == s_v is a plain uniform scale: positions * s, orthonormal
        # directions unchanged (matches the existing isotropic placement path).
        poses = _ortho_poses(seed=4)
        pts = np.random.RandomState(5).uniform(-1, 1, (300, 3))
        s = 2.3
        out, pts_out = apply_axis_split_scale(poses, pts, s_h=s, s_v=s)
        np.testing.assert_allclose(out["c"], np.asarray(poses["c"]) * s, rtol=1e-9)
        np.testing.assert_allclose(pts_out, pts * s, rtol=1e-9)
        np.testing.assert_allclose(out["f"], poses["f"], atol=1e-9)   # unchanged
        np.testing.assert_allclose(out["u"], poses["u"], atol=1e-9)

    def test_preserves_forward_horizontal_heading(self):
        # anchoring on up keeps forward's XZ azimuth (both X and Z scale by s_h),
        # so the look direction still follows the horizontally-scaled path.
        poses = _ortho_poses(n=10, seed=6)
        f_in = np.asarray(poses["f"])
        az_in = np.arctan2(f_in[:, 2], f_in[:, 0])
        out, _ = apply_axis_split_scale(poses, np.zeros((1, 3)), s_h=3.64, s_v=1.75)
        f_out = out["f"]
        az_out = np.arctan2(f_out[:, 2], f_out[:, 0])
        np.testing.assert_allclose(np.cos(az_out - az_in), 1.0, atol=1e-9)

    def test_recovers_anisotropic_corridor_dimensions(self):
        # recon corridor: horizontal compressed 1/s_h, vertical 1/s_v vs the model.
        # diag(s_h, s_v, s_h) must restore both the corridor width and its height.
        rng = np.random.RandomState(7)
        s_h, s_v = 3.64, 1.75
        model_w, model_h = 3.2, 3.5
        hw = (model_w / s_h) / 2.0        # recon half-width

        def wall(x0, n=2000):
            return np.column_stack([x0 + rng.normal(0, 0.004, n),
                                    rng.uniform(0, model_h / s_v, n),
                                    rng.uniform(0, 3.0, n)])

        recon = np.vstack([wall(-hw), wall(hw)])
        poses = {"c": np.array([[0.0, 0.7, 0.0]]),
                 "f": np.array([[0.0, 0.0, 1.0]]), "u": np.array([[0.0, 1.0, 0.0]])}
        _, pts_scaled = apply_axis_split_scale(poses, recon, s_h, s_v)
        xext = float(pts_scaled[:, 0].max() - pts_scaled[:, 0].min())
        yext = float(pts_scaled[:, 1].max() - pts_scaled[:, 1].min())
        self.assertAlmostEqual(xext, model_w, delta=model_w * 0.05)
        self.assertAlmostEqual(yext, model_h, delta=model_h * 0.05)


if __name__ == "__main__":
    unittest.main()

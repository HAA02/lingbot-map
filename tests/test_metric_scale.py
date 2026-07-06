"""TDD for multi-anchor metric scale fusion (auto coplay placement)."""
import unittest

import numpy as np

from scan2bim.metric_scale import (
    bbox_height_warning, camera_height_scale, estimate_floor_level, fuse_scale_estimates, speed_warning,
)


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


class TestBboxHeightWarning(unittest.TestCase):
    def test_normal_room_height_no_warning(self):
        self.assertIsNone(bbox_height_warning(3.6))

    def test_thin_single_discipline_bbox_warns(self):
        # e.g. a single fire-pipe-run dtdx passed alone -> bbox is the pipe's
        # own vertical span, not the room's floor-to-ceiling height.
        w = bbox_height_warning(0.49)
        self.assertIsNotNone(w)


if __name__ == "__main__":
    unittest.main()

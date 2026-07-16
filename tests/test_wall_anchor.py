"""TDD for corridor-width metric scale anchor (wall_anchor).

Corridor width (facing vertical walls) is an INDEPENDENT scale anchor: unlike the
ceiling-height and camera-height anchors it does not depend on the camera-carry
height assumption nor on the scan capturing the full floor->ceiling span, so it
stays valid on "looking up at the pipes" footage where the other two are weak.

Synthetic clouds are gravity-aligned Y-up: columns are [X, Y, Z], Y is up.
"""
import unittest

import numpy as np

from scan2bim.wall_anchor import estimate_wall_scale


def _wall(x0, length, height, n, noise, rng, axis="z"):
    """A thin vertical wall slab at X=x0 (or Z=x0), spanning the full height."""
    a = rng.uniform(0.0, length, n)
    y = rng.uniform(0.0, height, n)
    off = rng.normal(0.0, noise, n)
    if axis == "z":                       # wall runs along Z, normal along X
        return np.column_stack([x0 + off, y, a])
    return np.column_stack([a, y, x0 + off])  # wall runs along X, normal along Z


def _slab(y0, xlo, xhi, length, n, noise, rng):
    """A horizontal floor/ceiling slab at Y=y0."""
    x = rng.uniform(xlo, xhi, n)
    z = rng.uniform(0.0, length, n)
    y = y0 + rng.normal(0.0, noise, n)
    return np.column_stack([x, y, z])


def make_corridor(width=2.4, length=12.0, height=2.7, n_wall=3000, noise=0.01,
                  extra_walls=(), seed=0):
    """Two facing walls at X=+-width/2 running along Z, plus floor+ceiling.

    extra_walls: iterable of X positions for additional distractor walls
    (e.g. a room wall beyond the corridor)."""
    rng = np.random.RandomState(seed)
    parts = [
        _wall(-width / 2, length, height, n_wall, noise, rng),
        _wall(+width / 2, length, height, n_wall, noise, rng),
        _slab(0.0, -width / 2, width / 2, length, n_wall, noise, rng),
        _slab(height, -width / 2, width / 2, length, n_wall, noise, rng),
    ]
    for x in extra_walls:
        parts.append(_wall(x, length, height, n_wall, noise, rng))
    return np.vstack(parts)


def _cam_along_z(length=12.0, m=60):
    """Camera trajectory walking down the corridor centre (X~0, Z increasing)."""
    z = np.linspace(1.0, length - 1.0, m)
    x = np.zeros(m)
    return np.column_stack([x, z])


class TestCorridorDetection(unittest.TestCase):
    def test_detects_corridor_width_within_5pct(self):
        pts = make_corridor(width=2.4, seed=1)
        scale, info = estimate_wall_scale(pts, [2.4], cam_xz=_cam_along_z())
        self.assertIsNotNone(scale)
        self.assertAlmostEqual(info["recon_width"], 2.4, delta=2.4 * 0.05)

    def test_recovers_scale_from_ratio(self):
        # recon corridor is 0.8 units wide, real model corridor is 2.4 m -> 3.0x.
        pts = make_corridor(width=0.8, length=4.0, seed=2)
        scale, info = estimate_wall_scale(pts, [2.4], cam_xz=_cam_along_z(length=4.0))
        self.assertIsNotNone(scale)
        self.assertAlmostEqual(scale, 3.0, delta=3.0 * 0.05)

    def test_scale_independent_of_camera_height_assumption(self):
        # sanity: the wall anchor never consults camera height, only the gap.
        pts = make_corridor(width=1.2, length=6.0, seed=7)
        scale, info = estimate_wall_scale(pts, [2.4], cam_xz=_cam_along_z(length=6.0))
        self.assertIsNotNone(scale)
        self.assertAlmostEqual(scale, 2.0, delta=2.0 * 0.06)   # 2.4 / 1.2


class TestFallbackContract(unittest.TestCase):
    def test_open_space_returns_none(self):
        rng = np.random.RandomState(3)
        pts = np.column_stack([
            rng.uniform(-2.0, 2.0, 30000),
            rng.uniform(0.0, 2.7, 30000),
            rng.uniform(0.0, 10.0, 30000),
        ])
        scale, info = estimate_wall_scale(pts, [2.4], cam_xz=None)
        self.assertIsNone(scale)
        self.assertIn("fail", info)

    def test_only_floor_and_ceiling_no_walls_returns_none(self):
        rng = np.random.RandomState(4)
        floor = _slab(0.0, -3.0, 3.0, 10.0, 3000, 0.01, rng)
        ceil = _slab(2.7, -3.0, 3.0, 10.0, 3000, 0.01, rng)
        pts = np.vstack([floor, ceil])
        scale, info = estimate_wall_scale(pts, [2.4], cam_xz=_cam_along_z())
        self.assertIsNone(scale)

    def test_no_model_widths_returns_none(self):
        pts = make_corridor(seed=5)
        scale, info = estimate_wall_scale(pts, [], cam_xz=_cam_along_z())
        self.assertIsNone(scale)

    def test_too_few_points_returns_none(self):
        pts = np.random.RandomState(6).uniform(-1, 1, (10, 3))
        scale, info = estimate_wall_scale(pts, [2.4])
        self.assertIsNone(scale)


class TestPairSelection(unittest.TestCase):
    def test_prefers_trajectory_straddling_corridor_over_room_wall(self):
        # corridor walls at X=+-1.2 (width 2.4); a distractor room wall at X=+4.
        # camera walks the corridor centre -> must pick the 2.4 gap, not 5.2.
        pts = make_corridor(width=2.4, length=12.0, extra_walls=(4.0,), seed=8)
        scale, info = estimate_wall_scale(pts, [2.4], cam_xz=_cam_along_z())
        self.assertIsNotNone(scale)
        self.assertEqual(info["n_walls"], 3)
        self.assertTrue(info["straddles_trajectory"])
        self.assertAlmostEqual(info["recon_width"], 2.4, delta=0.2)


class TestStraddleEnforcement(unittest.TestCase):
    def test_non_straddling_pair_returns_none(self):
        # both walls on the SAME side of the camera path (X=1.0 and X=2.0, camera
        # at X=0) -> not the corridor actually walked, so no scale is returned.
        rng = np.random.RandomState(12)
        parts = [
            _wall(1.0, 12.0, 2.7, 3000, 0.01, rng),
            _wall(2.0, 12.0, 2.7, 3000, 0.01, rng),
            _slab(0.0, 0.5, 2.5, 12.0, 3000, 0.01, rng),
            _slab(2.7, 0.5, 2.5, 12.0, 3000, 0.01, rng),
        ]
        pts = np.vstack(parts)
        scale, info = estimate_wall_scale(pts, [2.4], cam_xz=_cam_along_z())
        self.assertIsNone(scale)
        self.assertEqual(info.get("fail"), "no straddling pair")

    def test_no_trajectory_cannot_verify_straddle_returns_none(self):
        # without a camera trajectory there is nothing to straddle -> no result.
        pts = make_corridor(width=2.4, seed=13)
        scale, info = estimate_wall_scale(pts, [2.4], cam_xz=None)
        self.assertIsNone(scale)


class TestTrajectoryRadiusPrefilter(unittest.TestCase):
    def test_prefilter_drops_far_points_and_detects_corridor(self):
        # the corridor actually walked (near the path) plus a large mass of clutter
        # far down-corridor, well beyond where the camera went. The proximity filter
        # keeps only the near-path corridor and recovers it cleanly. (On real,
        # ceiling-facing scans this is what turns a drowned scan into a clean spike.)
        rng = np.random.RandomState(14)
        corridor = make_corridor(width=2.4, length=12.0, seed=14)   # walls at X=+-1.2
        far = np.column_stack([
            rng.uniform(-8.0, 8.0, 50000),
            rng.uniform(0.0, 2.7, 50000),
            rng.uniform(30.0, 60.0, 50000),   # far beyond the walked Z range (~1..11)
        ])
        pts = np.vstack([corridor, far])
        cam = _cam_along_z()
        scale, info = estimate_wall_scale(pts, [2.4], cam_xz=cam, trajectory_radius=2.7)
        self.assertIsNotNone(scale)
        self.assertTrue(info["straddles_trajectory"])
        self.assertAlmostEqual(info["recon_width"], 2.4, delta=0.2)
        # the far clutter (majority of points) was excluded by the radius filter
        self.assertIn("n_after_radius", info)
        self.assertLessEqual(info["n_after_radius"], len(corridor))
        self.assertLess(info["n_after_radius"], len(pts) * 0.5)

    def test_radius_none_is_backward_compatible(self):
        pts = make_corridor(width=2.4, seed=15)
        cam = _cam_along_z()
        a, _ = estimate_wall_scale(pts, [2.4], cam_xz=cam)
        b, _ = estimate_wall_scale(pts, [2.4], cam_xz=cam, trajectory_radius=None)
        self.assertEqual(a, b)


class TestInfoDiagnostics(unittest.TestCase):
    def test_info_has_diagnostic_fields(self):
        pts = make_corridor(width=2.4, seed=9)
        scale, info = estimate_wall_scale(pts, [2.4], cam_xz=_cam_along_z())
        for key in ("n_walls", "recon_width", "model_width", "scale",
                    "straddles_trajectory", "confidence"):
            self.assertIn(key, info)
        self.assertGreater(info["confidence"], 0.0)
        self.assertLessEqual(info["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()

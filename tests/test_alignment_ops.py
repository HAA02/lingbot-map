"""Gasan coverage ops: start-hint 2-click + correspondence Sim3 gates.

These are the contracts behind /coverage 「시작점 지정」 then 「대응점 찍기」.
No GPU, no upload fixtures.
"""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from lingbot_map.bim.alignment import (
    RMSE_GREEN_M,
    RMSE_YELLOW_M,
    START_DIR_MIN_SEP_M,
    alignment_permits_coverage_analysis,
    auto_geometric_alignment_record,
    correspondence_spread,
    solve_scan_to_model_alignment,
    start_direction_ok,
    yaw_candidates_from_user_direction,
)


def _circ_diff(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def _pairs(src, dst):
    return [{"scan": s.tolist(), "model": d.tolist()} for s, d in zip(src, dst)]


class TestStartHintTwoClicks(unittest.TestCase):
    def test_min_sep_matches_ui(self):
        self.assertEqual(START_DIR_MIN_SEP_M, 0.3)

    def test_too_close_rejected(self):
        self.assertFalse(start_direction_ok([1.0, 2.0, 3.0], [1.2, 2.0, 3.0]))

    def test_far_enough_accepted(self):
        self.assertTrue(start_direction_ok([1.0, 2.0, 3.0], [1.0 + START_DIR_MIN_SEP_M, 2.0, 9.0]))
        self.assertTrue(start_direction_ok([0.0, 0.0, 0.0], [0.0, START_DIR_MIN_SEP_M, 0.0]))


class TestYawFromUserDirection(unittest.TestCase):
    def test_small_set_not_full_sweep(self):
        yaws = yaw_candidates_from_user_direction(40.0, 10.0)
        self.assertEqual(len(yaws), 10)
        self.assertLess(len(yaws), 18)

    def test_only_around_heading_and_flip(self):
        yaws = yaw_candidates_from_user_direction(40.0, 10.0)
        bases = (30.0, 210.0)
        for y in yaws:
            nearest = min(_circ_diff(y, b) for b in bases)
            self.assertLessEqual(nearest, 8.01)

    def test_no_orthogonal_flood(self):
        yaws = yaw_candidates_from_user_direction(0.0, 0.0)
        for y in yaws:
            self.assertGreater(min(_circ_diff(y, 90.0), _circ_diff(y, 270.0)), 70.0)


class TestCorrespondenceSim3(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.src = rng.normal(size=(6, 3)) * 2.0
        self.s = 1.97
        self.R = Rotation.from_euler("Z", 37.0, degrees=True).as_matrix()
        self.t = np.array([4.2, -1.1, 0.8])
        self.dst = self.s * (self.src @ self.R.T) + self.t

    def test_fewer_than_four_is_red(self):
        out = solve_scan_to_model_alignment(_pairs(self.src[:3], self.dst[:3]))
        self.assertEqual(out["quality"], "red")
        self.assertEqual(out["n"], 3)
        self.assertFalse(alignment_permits_coverage_analysis(out))

    def test_four_spread_points_green_and_recover_sim3(self):
        src = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.5, 0.0], [0.4, 0.3, 1.8]])
        dst = self.s * (src @ self.R.T) + self.t
        out = solve_scan_to_model_alignment(_pairs(src, dst))
        self.assertEqual(out["quality"], "green")
        self.assertLess(out["rmse_m"], RMSE_GREEN_M)
        self.assertAlmostEqual(out["scale"], self.s, places=5)
        self.assertTrue(out["correspondence_spread"]["ok"])
        self.assertTrue(alignment_permits_coverage_analysis(out))

    def test_collinear_four_is_red(self):
        src = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        dst = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
        spread = correspondence_spread(src)
        self.assertLess(spread["rank"], 2)
        out = solve_scan_to_model_alignment(_pairs(src, dst))
        self.assertEqual(out["quality"], "red")
        self.assertFalse(out["correspondence_spread"]["ok"])
        self.assertFalse(alignment_permits_coverage_analysis(out))

    def test_noisy_pairs_yellow_still_allows_coverage(self):
        src = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.5, 0.0], [0.4, 0.3, 1.8], [1.2, 1.1, 0.6]])
        dst = self.s * (src @ self.R.T) + self.t
        dst = dst.copy()
        dst[0] += np.array([0.18, 0.0, 0.0])
        dst[1] += np.array([0.0, 0.16, 0.0])
        out = solve_scan_to_model_alignment(_pairs(src, dst))
        self.assertIn(out["quality"], ("green", "yellow"))
        self.assertLessEqual(out["rmse_m"], RMSE_YELLOW_M)
        self.assertTrue(alignment_permits_coverage_analysis(out))


class TestAutoGeometricScaleRecord(unittest.TestCase):
    def test_searched_scale_is_not_replaced_by_default_one(self):
        al = {
            "scale": 1.97,
            "rotation": np.eye(3).tolist(),
            "translation": [4.0, -0.5, 1.0],
        }
        rec = auto_geometric_alignment_record(al, "yellow")
        self.assertEqual(rec["scale"], 1.97)
        self.assertNotEqual(rec["scale"], 1.0)
        pts = np.array([[1.0, 0.0, 0.0]])
        pred = rec["scale"] * (np.asarray(rec["rotation"]) @ pts.T).T + rec["translation"]
        expected = 1.97 * pts + np.array(rec["translation"])
        np.testing.assert_allclose(pred, expected)

    def test_red_auto_record_blocks_coverage(self):
        al = {"scale": 0.5, "rotation": np.eye(3).tolist(), "translation": [0, 0, 0]}
        rec = auto_geometric_alignment_record(al, "red")
        self.assertFalse(alignment_permits_coverage_analysis(rec))


class TestCoverageGate(unittest.TestCase):
    def test_missing_and_red_blocked(self):
        self.assertFalse(alignment_permits_coverage_analysis(None))
        self.assertFalse(alignment_permits_coverage_analysis({"quality": "red"}))
        self.assertFalse(alignment_permits_coverage_analysis({}))

    def test_green_yellow_ok(self):
        self.assertTrue(alignment_permits_coverage_analysis({"quality": "green"}))
        self.assertTrue(alignment_permits_coverage_analysis({"quality": "yellow"}))


if __name__ == "__main__":
    unittest.main()

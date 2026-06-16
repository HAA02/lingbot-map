"""TDD for scan2bim.pipeline.analyze_progress — the end-to-end chain
(register → coverage → ledger → color attribution)."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from scan2bim.pipeline import analyze_progress, apply_sim3

RED_INDEX = [{"discipline": "fire", "discipline_code": "FXX", "material_id": 1,
              "rgb": (1.0, 0.0, 0.0), "alpha": 1.0, "kind": "chromatic",
              "hue": "red", "usable": True}]


def _grid(center, n=200, seed=0):
    return np.random.default_rng(seed).uniform(-0.3, 0.3, (n, 3)) + np.asarray(center, float)


class TestAnalyzeProgress(unittest.TestCase):
    def setUp(self):
        self.A = _grid([0, 0, 0], seed=1)
        self.B = _grid([5, 0, 0], seed=2)
        self.C = _grid([10, 0, 0], seed=3)  # never scanned
        self.elements = [{"guid": "A", "points": self.A},
                         {"guid": "B", "points": self.B},
                         {"guid": "C", "points": self.C}]

    def test_coverage_ledger_and_color(self):
        scan = np.vstack([self.A, self.B])
        colors = np.tile([230, 20, 20], (len(scan), 1)).astype(np.uint8)
        out = analyze_progress(self.elements, scan, colors, RED_INDEX, "2026-06-16", radius=0.15)
        st = out["ledger"]["statuses"]
        self.assertEqual(st["A"], "observed")
        self.assertEqual(st["B"], "observed")
        self.assertEqual(st["C"], "not_observed")
        red = next(m for m in out["color_matches"] if m["hue"] == "red")
        self.assertIn("fire", {x["discipline"] for x in red["matches"]})

    def test_alignment_places_scan(self):
        # scan delivered in a rotated/scaled frame; alignment brings it back
        s = 1.3
        R = Rotation.from_euler("XYZ", [12, -20, 8], degrees=True).as_matrix()
        t = np.array([4.0, -2.0, 1.0])
        model_pts = np.vstack([self.A, self.B])
        scan_raw = (model_pts - t) @ R / s  # inverse of s*R*x+t
        colors = np.tile([230, 20, 20], (len(scan_raw), 1)).astype(np.uint8)
        align = {"scale": s, "rotation": R.tolist(), "translation": t.tolist()}
        # sanity: applying alignment recovers model points
        self.assertLess(np.linalg.norm(apply_sim3(scan_raw, align) - model_pts, axis=1).mean(), 1e-6)
        out = analyze_progress(self.elements, scan_raw, colors, RED_INDEX, "2026-06-16",
                               alignment=align, radius=0.15)
        self.assertEqual(out["ledger"]["statuses"]["A"], "observed")
        self.assertEqual(out["ledger"]["statuses"]["C"], "not_observed")


if __name__ == "__main__":
    unittest.main()

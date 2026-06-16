"""TDD for FR-2.2 color↔system matching — the text-free semantic bridge.

Scan points carry real RGB (red fire pipes, etc). The design carries per-system
material colors. Segment the scan by hue and map each chromatic cluster to the
design disciplines/systems sharing that hue. Achromatic (gray) points are not
usable for color matching and must be excluded.
"""
import os
import unittest

import numpy as np

from scan2bim.colormap import build_color_index
from scan2bim.dtdx import load_dtdx
from scan2bim.matching import match_to_design, segment_by_color

GASAN = os.path.join(os.path.dirname(__file__), "..", "models", "Gasan_7F")
FXX = os.path.join(GASAN, "G7F_FAB_FXX_7F-0_Central_1.dtdx")


class TestColorMatching(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        red = np.tile([230, 20, 20], (1000, 1))
        blue = np.tile([20, 20, 230], (600, 1))
        gray = np.tile([128, 128, 128], (800, 1))
        self.colors = np.vstack([red, blue, gray]).astype(np.uint8)
        self.points = rng.normal(size=(2400, 3))

    def test_segments_chromatic_only(self):
        clusters = segment_by_color(self.points, self.colors)
        hues = {c["hue"] for c in clusters}
        self.assertIn("red", hues)
        self.assertIn("blue", hues)
        self.assertNotIn("_gray", hues)  # achromatic excluded

    def test_cluster_counts(self):
        clusters = segment_by_color(self.points, self.colors)
        red = next(c for c in clusters if c["hue"] == "red")
        self.assertEqual(red["count"], 1000)
        self.assertEqual(len(red["centroid"]), 3)

    def test_red_matches_fire_discipline(self):
        idx = build_color_index([load_dtdx(FXX)])
        clusters = segment_by_color(self.points, self.colors)
        matched = match_to_design(clusters, idx)
        red = next(m for m in matched if m["hue"] == "red")
        disciplines = {mm["discipline"] for mm in red["matches"]}
        self.assertIn("fire", disciplines)

    def test_tiny_clusters_filtered(self):
        # 5 stray orange points must not become a cluster
        colors = np.vstack([np.tile([230, 20, 20], (500, 1)),
                            np.tile([240, 140, 20], (5, 1))]).astype(np.uint8)
        pts = np.zeros((505, 3))
        clusters = segment_by_color(pts, colors, min_count=20)
        self.assertEqual({c["hue"] for c in clusters}, {"red"})


if __name__ == "__main__":
    unittest.main()

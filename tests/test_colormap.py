"""TDD for FR-1.2 design color map — classify per-material color as
chromatic (usable for color↔system matching) vs achromatic (gray, not usable).

Ground truth (FXX/HVAC inventory, 2026-06-15): fire red[1,0,0] + blue[0,0,1]
are chromatic; HVAC's single [0.5,0.5,0.5] is achromatic.
"""
import glob
import os
import unittest

from scan2bim.dtdx import load_dtdx
from scan2bim.colormap import classify_color, build_color_index

GASAN = os.path.join(os.path.dirname(__file__), "..", "models", "Gasan_7F")


class TestClassifyColor(unittest.TestCase):
    def test_red_is_chromatic(self):
        self.assertEqual(classify_color((1.0, 0.0, 0.0)), ("chromatic", "red"))

    def test_blue_is_chromatic(self):
        self.assertEqual(classify_color((0.0, 0.0, 1.0)), ("chromatic", "blue"))

    def test_gray_is_achromatic(self):
        kind, hue = classify_color((0.5, 0.5, 0.5))
        self.assertEqual(kind, "achromatic")
        self.assertIsNone(hue)

    def test_near_black_and_white_achromatic(self):
        self.assertEqual(classify_color((0.0, 0.0, 0.0))[0], "achromatic")
        self.assertEqual(classify_color((0.95, 0.95, 0.95))[0], "achromatic")


class TestColorIndex(unittest.TestCase):
    def test_fire_has_usable_red(self):
        m = load_dtdx(os.path.join(GASAN, "G7F_FAB_FXX_7F-0_Central_1.dtdx"))
        idx = build_color_index([m])
        reds = [e for e in idx if e["discipline"] == "fire" and e["hue"] == "red" and e["usable"]]
        self.assertTrue(reds, "fire discipline should expose a usable red color")

    def test_hvac_only_achromatic(self):
        m = load_dtdx(os.path.join(GASAN, "G7F_FAB_HXX_7F-0_Central_1.dtdx"))
        idx = build_color_index([m])
        self.assertTrue(idx, "hvac should have at least one material entry")
        self.assertTrue(all(not e["usable"] for e in idx),
                        "hvac (single gray) must be all non-usable for color matching")


if __name__ == "__main__":
    unittest.main()

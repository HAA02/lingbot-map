"""TDD for FR-2.3 object anchors — per-instance model anchors + type classify.

Verifies each discipline yields object-instance centers (linkMesh/base pivots)
inside the model bounds, that HXX produces AC-like and SXX produces column-like
anchors, and that the X-flip matches the co-play / authoring-viewer frame.
"""
import glob
import os
import unittest

from scan2bim.anchors import (
    AC, COLUMN, anchor_inventory, classify_anchor, model_anchors,
)

GASAN = os.path.join(os.path.dirname(__file__), "..", "models", "Gasan_7F")
FILES = sorted(glob.glob(os.path.join(GASAN, "*.dtdx")))
HXX = next(f for f in FILES if "HXX" in f)
SXX = next(f for f in FILES if "SXX" in f)


class TestAnchors(unittest.TestCase):
    def test_extracts_instances(self):
        a = model_anchors(HXX)
        self.assertGreater(len(a), 10)
        for x in a[:5]:
            self.assertEqual(len(x["center"]), 3)
            self.assertEqual(len(x["size"]), 3)

    def test_centers_in_bounds(self):
        a = model_anchors(SXX)
        ys = [c["center"][1] for c in a]
        self.assertTrue(all(20 < y < 30 for y in ys))  # Y = vertical, ~floor-ceiling band

    def test_flip_mirrors_x(self):
        on = model_anchors(SXX, flip_x=True)
        off = model_anchors(SXX, flip_x=False)
        self.assertAlmostEqual(on[0]["center"][0], -off[0]["center"][0], places=2)

    def test_hvac_yields_ac(self):
        cat = anchor_inventory([HXX])
        self.assertIn(AC, cat)
        self.assertGreater(len(cat[AC]), 0)

    def test_structure_yields_columns(self):
        cat = anchor_inventory([SXX])
        self.assertIn(COLUMN, cat)
        self.assertGreater(len(cat[COLUMN]), 5)

    def test_classify_column_by_height(self):
        # tall + small footprint near ceiling -> column
        self.assertEqual(classify_anchor("SXX", [0.4, 3.0, 0.4], [0, 26, 0], 27.5), COLUMN)
        # wide flat in hvac -> ac
        self.assertEqual(classify_anchor("HXX", [0.84, 0.3, 0.84], [0, 27.3, 0], 27.3), AC)


if __name__ == "__main__":
    unittest.main()

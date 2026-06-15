"""TDD for FR-1.1b .dtdx geometry decode + instance placement.

Verifies the binary blob (vp/tp absolute offsets, vc*12 float32 verts,
materialCount submesh index arrays) decodes to a world-space triangle soup,
with base meshes + linkMesh instances placed so the model spans the floor
(not collapsed at origin), and fire triangles carry the red material color.
"""
import os
import unittest

import numpy as np

from scan2bim.dtdx_geometry import decode_geometry

GASAN = os.path.join(os.path.dirname(__file__), "..", "models", "Gasan_7F")
FXX = os.path.join(GASAN, "G7F_FAB_FXX_7F-0_Central_1.dtdx")


class TestDtdxGeometry(unittest.TestCase):
    def setUp(self):
        self.g = decode_geometry(FXX)

    def test_has_triangles(self):
        self.assertGreater(self.g["triangle_count"], 100)

    def test_positions_finite(self):
        for m in self.g["meshes"]:
            if len(m["positions"]):
                self.assertTrue(np.isfinite(m["positions"]).all())

    def test_instances_spread_over_floor(self):
        # base shapes are ~0.2x0.2x2 m pipes; with instances placed the model
        # must span many metres (a real 7F fire layout), not sit at origin.
        mn, mx = self.g["bbox"]
        span = np.asarray(mx) - np.asarray(mn)
        self.assertGreater(float(span.max()), 5.0, "instances must spread the model over the floor")

    def test_fire_red_present(self):
        colors = {tuple(round(c, 2) for c in m["color"]) for m in self.g["meshes"] if len(m["positions"])}
        self.assertIn((1.0, 0.0, 0.0), colors, "fire red triangles expected")


if __name__ == "__main__":
    unittest.main()

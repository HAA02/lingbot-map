"""TDD for FR-1.1 .dtdx ingestion (scan2bim.dtdx).

Ground truth from direct msgpack decode of the Gasan_7F FXX (fire) model
(2026-06-15): 4 materials incl. red[1,0,0] + blue[0,0,1], 76 base meshes,
224 element instances (linkMesh), 609 connectors, 728 attr records.
"""
import glob
import os
import unittest

from scan2bim.dtdx import load_dtdx, discipline_of

GASAN = os.path.join(os.path.dirname(__file__), "..", "models", "Gasan_7F")
FXX = os.path.join(GASAN, "G7F_FAB_FXX_7F-0_Central_1.dtdx")


class TestDtdxHeader(unittest.TestCase):
    def test_signature_and_version(self):
        m = load_dtdx(FXX)
        self.assertEqual(m.signature, 0xFF09)
        self.assertEqual(m.version, 0x00200000)


class TestDtdxDiscipline(unittest.TestCase):
    def test_discipline_parsed_from_filename(self):
        self.assertEqual(discipline_of("G7F_FAB_FXX_7F-0_Central_1.dtdx"), ("FXX", "fire"))
        self.assertEqual(discipline_of("G7F_FAB_SXX_7F-0_Central_1.dtdx"), ("SXX", "structure"))
        self.assertEqual(discipline_of("G7F_FAB_AXX_7F-0_Central_1.dtdx"), ("AXX", "architecture"))

    def test_model_carries_discipline(self):
        m = load_dtdx(FXX)
        self.assertEqual(m.discipline_code, "FXX")
        self.assertEqual(m.discipline, "fire")


class TestDtdxMaterialsColorCoded(unittest.TestCase):
    def test_four_materials(self):
        m = load_dtdx(FXX)
        self.assertEqual(len(m.materials), 4)

    def test_fire_red_and_blue_present(self):
        m = load_dtdx(FXX)
        colors = {tuple(round(c, 2) for c in mat.diffuse_rgb) for mat in m.materials}
        self.assertIn((1.0, 0.0, 0.0), colors, "fire system red material expected")
        self.assertIn((0.0, 0.0, 1.0), colors, "blue material expected")


class TestDtdxInventory(unittest.TestCase):
    def test_counts_match_ground_truth(self):
        m = load_dtdx(FXX)
        self.assertEqual(len(m.meshes), 76)
        self.assertEqual(len(m.elements), 224)   # linkMesh instances
        self.assertEqual(len(m.connectors), 609)

    def test_attr_and_guids(self):
        m = load_dtdx(FXX)
        self.assertEqual(len(m.attr), 728)
        # attr carries IfcGUID-style 22-char base64 identifiers
        self.assertGreater(len(m.guids), 100)
        self.assertTrue(all(len(g) >= 20 for g in list(m.guids)[:20]))


class TestAllSixDisciplinesLoad(unittest.TestCase):
    def test_all_files_parse(self):
        files = sorted(glob.glob(os.path.join(GASAN, "*.dtdx")))
        self.assertEqual(len(files), 6)
        for f in files:
            m = load_dtdx(f)
            self.assertGreater(len(m.materials), 0, f"{os.path.basename(f)} has materials")
            self.assertGreaterEqual(len(m.elements), 0)


if __name__ == "__main__":
    unittest.main()

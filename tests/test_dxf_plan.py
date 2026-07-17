"""Tests for scan2bim.dxf_plan (DXF floor plan as the corridor-width source) and its
wiring into tools.build_coplay._resolve_horizontal_scale (--dxf).

A hand-written minimal ASCII DXF fixture exercises the raw parser + mm->m scaling;
synthetic corridors exercise the width/transform math; a small ezdxf-built doc
covers the A-DOOR INSERT path. ezdxf is a lazy import inside dxf_plan, so the whole
module is skipped when it is not installed.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import ezdxf  # noqa: F401
    _HAS_EZDXF = True
except ImportError:
    _HAS_EZDXF = False

from scan2bim import dxf_plan as dp

# A hand-written R12 ASCII DXF: $INSUNITS=4 (mm) + two A-WALL lines 2 m apart
# (X=0 and X=2000 mm, spanning Y 0..10000 mm) = a 2.0 m corridor along Y.
_HAND_DXF = """0
SECTION
2
HEADER
9
$INSUNITS
70
4
0
ENDSEC
0
SECTION
2
ENTITIES
0
LINE
8
A-WALL-____-OTLN
10
0.0
20
0.0
30
0.0
11
0.0
21
10000.0
31
0.0
0
LINE
8
A-WALL-____-OTLN
10
2000.0
20
0.0
30
0.0
11
2000.0
21
10000.0
31
0.0
0
ENDSEC
0
EOF
"""


def _write(tmp: Path, text: str) -> Path:
    p = tmp / "plan.dxf"; p.write_text(text, encoding="utf-8"); return p


@unittest.skipUnless(_HAS_EZDXF, "ezdxf not installed")
class TestLoadWallSegments(unittest.TestCase):
    def test_hand_written_ascii_dxf_mm_to_m(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            segs = dp.load_wall_segments(_write(Path(td), _HAND_DXF))
            self.assertEqual(segs.shape, (2, 2, 2))
            # mm -> m: 10000 mm -> 10 m, 2000 mm -> 2 m
            self.assertAlmostEqual(segs[:, :, 1].max(), 10.0, places=3)
            self.assertAlmostEqual(sorted(segs[:, 0, 0])[-1], 2.0, places=3)

    def test_missing_layers_returns_empty(self):
        import tempfile
        no_wall = _HAND_DXF.replace("A-WALL-____-OTLN", "M-PIPE-CENTER")
        with tempfile.TemporaryDirectory() as td:
            segs = dp.load_wall_segments(_write(Path(td), no_wall))
            self.assertEqual(len(segs), 0)


@unittest.skipUnless(_HAS_EZDXF, "ezdxf not installed")
class TestCorridorWidths(unittest.TestCase):
    def _corridor(self, width_m=2.0, length_m=10.0, x0=0.0):
        """Two walls along Y at X=x0 and X=x0+width, densely sampled as segments."""
        ys = np.linspace(0, length_m, 40)
        segs = []
        for x in (x0, x0 + width_m):
            for i in range(len(ys) - 1):
                segs.append([[x, ys[i]], [x, ys[i + 1]]])
        return np.asarray(segs, dtype=np.float64)

    def test_two_walls_give_clear_width(self):
        segs = self._corridor(width_m=2.0)
        traj = np.column_stack([np.full(20, 1.0), np.linspace(1, 9, 20)])   # centreline at X=1
        widths, info = dp.corridor_widths_near(segs, traj, radius=2.5)
        self.assertTrue(widths, info)
        self.assertAlmostEqual(min(widths, key=lambda w: abs(w - 2.0)), 2.0, delta=0.05)

    def test_offset_walker_still_measures_full_width(self):
        """The walker hugging one wall must still recover the full clear width."""
        segs = self._corridor(width_m=1.8)
        traj = np.column_stack([np.full(20, 0.2), np.linspace(1, 9, 20)])   # hugging the X=0 wall
        widths, info = dp.corridor_widths_near(segs, traj, radius=2.5)
        self.assertTrue(widths, info)
        self.assertAlmostEqual(min(widths, key=lambda w: abs(w - 1.8)), 1.8, delta=0.05)


@unittest.skipUnless(_HAS_EZDXF, "ezdxf not installed")
class TestPlanTransform(unittest.TestCase):
    def test_recovers_pure_translation(self):
        rng = np.random.RandomState(0)
        base = rng.uniform(-10, 10, (400, 2))
        dxf_seg = np.stack([base, base + rng.uniform(-0.2, 0.2, base.shape)], axis=1)
        shift = np.array([2.93, -1.5])
        dtdx = base + shift                                                 # same walls, translated
        tf = dp.estimate_plan_transform(dxf_seg, dtdx)
        self.assertTrue(tf["ok"], tf)
        self.assertAlmostEqual(tf["translation"][0], shift[0], delta=0.1)
        self.assertAlmostEqual(tf["translation"][1], shift[1], delta=0.1)
        self.assertLess(tf["residual"], 0.5)

    def test_refuses_forced_fit_when_unrelated(self):
        rng = np.random.RandomState(1)
        dxf_seg = np.stack([rng.uniform(-2, 2, (200, 2))] * 2, axis=1)
        dtdx = rng.uniform(40, 60, (200, 2))                               # nowhere near -> no honest fit
        tf = dp.estimate_plan_transform(dxf_seg, dtdx)
        self.assertFalse(tf["ok"], tf)


@unittest.skipUnless(_HAS_EZDXF, "ezdxf not installed")
class TestDoorPositions(unittest.TestCase):
    def test_door_insert_near_trajectory(self):
        import tempfile
        doc = ezdxf.new(setup=True)
        doc.blocks.new(name="DOOR")
        msp = doc.modelspace()
        msp.add_blockref("DOOR", (1000.0, 5000.0), dxfattribs={"layer": "A-DOOR-____-OTLN"})  # mm -> (1,5) m
        msp.add_blockref("DOOR", (40000.0, 40000.0), dxfattribs={"layer": "A-DOOR-____-OTLN"})  # far away
        doc.header["$INSUNITS"] = 4
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "d.dxf"; doc.saveas(p)
            traj = np.column_stack([np.full(10, 1.0), np.linspace(1, 9, 10)])
            near, info = dp.door_positions_near(p, traj, radius=2.0)
            self.assertEqual(info["n_doors_total"], 2)
            self.assertEqual(len(near), 1)                                  # only the (1,5) door is near
            self.assertAlmostEqual(near[0, 0], 1.0, places=2)


class TestResolveDxfBranch(unittest.TestCase):
    """tools.build_coplay._resolve_horizontal_scale DXF branch: s_h = DXF width /
    recon gap, band-selected — and no-regression when dxf_widths is not given."""

    def _recon_corridor(self, gap=1.0, n=4000, seed=0):
        """A recon gravity-aligned cloud (Pg) with two facing walls `gap` apart plus
        a centreline trajectory; returns (Pg, cam_xz, vext)."""
        rng = np.random.RandomState(seed)
        z = rng.uniform(0, 5, n)
        y = rng.uniform(0, 2.2, n)                                          # full floor->ceiling span
        wall = np.where(rng.rand(n) < 0.5, -gap / 2, gap / 2) + rng.normal(0, 0.01, n)
        Pg = np.column_stack([wall, y, z])
        cam = np.column_stack([np.zeros(30), np.linspace(0.5, 4.5, 30)])    # walk the centre along Z
        vext = float(np.percentile(Pg[:, 1], 97) - np.percentile(Pg[:, 1], 3))
        return Pg, cam, vext

    def test_recon_gap_and_band_selected_s_h(self):
        import tools.build_coplay as bc
        Pg, cam, vext = self._recon_corridor(gap=1.0)
        gap, ginfo = bc._recon_corridor_gap(Pg, cam, vext)
        self.assertIsNotNone(gap, ginfo)
        self.assertAlmostEqual(gap, 1.0, delta=0.12)
        # DXF widths: 2.0 -> s_h ~2.0 (in band), 3.6 -> ~3.6 (out of [1.8,2.8]) -> 2.0 chosen
        s_h, fb, info = bc._resolve_horizontal_scale(Pg, cam, None, vext, s_v=1.75,
                                                     dxf_widths=[2.0, 3.6], s_h_band=(1.8, 2.8))
        self.assertIsNone(fb, info)
        self.assertEqual(info["source"], "dxf_corridor")
        self.assertEqual(info["selected_width"], 2.0)
        self.assertAlmostEqual(s_h, 2.0, delta=0.25)

    def test_no_dxf_widths_is_unchanged(self):
        """dxf_widths=None must not touch the existing SXX/hint/override path."""
        import tools.build_coplay as bc
        Pg, cam, vext = self._recon_corridor(gap=1.0)
        s_h, fb, info = bc._resolve_horizontal_scale(Pg, cam, None, vext, s_v=1.75)
        self.assertNotEqual(info.get("source"), "dxf_corridor")

    def test_none_in_band_falls_back_to_s_v(self):
        import tools.build_coplay as bc
        Pg, cam, vext = self._recon_corridor(gap=1.0)
        s_h, fb, info = bc._resolve_horizontal_scale(Pg, cam, None, vext, s_v=1.75,
                                                     dxf_widths=[5.0], s_h_band=(1.8, 2.8))   # 5.0/1.0=5 out
        self.assertEqual(s_h, 1.75)
        self.assertIsNotNone(fb)


class TestCliWiring(unittest.TestCase):
    """--dxf is plumbed into both CLIs and the placement path, and absence is a
    no-op (the parameter defaults to None everywhere)."""

    def test_build_coplay_and_validate_expose_dxf(self):
        import inspect
        import tools.build_coplay as bc
        import tools.validate_wall_anchor as vwa
        self.assertTrue(hasattr(bc, "_dxf_corridor_scale_inputs"))
        self.assertTrue(hasattr(bc, "_recon_corridor_gap"))
        self.assertIn("dxf_widths", inspect.signature(bc.place_rigid).parameters)
        self.assertIn("dxf_widths", inspect.signature(bc._resolve_horizontal_scale).parameters)
        self.assertIn("dxf_widths", inspect.signature(vwa.compute_metric_scale).parameters)

    def test_dxf_absent_leaves_place_registered_pipe_signatures_untouched(self):
        """place_registered / place_pipe_auto must NOT have grown a dxf param (their
        bodies are frozen this cycle) — only place_rigid opts into DXF."""
        import inspect
        import tools.build_coplay as bc
        self.assertNotIn("dxf_widths", inspect.signature(bc.place_registered).parameters)
        self.assertNotIn("dxf_widths", inspect.signature(bc.place_pipe_auto).parameters)


if __name__ == "__main__":
    unittest.main()

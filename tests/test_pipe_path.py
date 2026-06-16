"""TDD for auto pipe-path centerline detection."""
import glob
import os
import unittest

import numpy as np

from scan2bim.pipe_path import (
    main_pipe_run, main_pipe_run_L, run_centerline, run_L_polyline, trajectory_turn_fraction,
)

FXX = next(f for f in glob.glob(os.path.join(os.path.dirname(__file__), "..", "models", "Gasan_7F", "*.dtdx"))
           if "FXX" in f)


class TestPipePath(unittest.TestCase):
    def test_synthetic_run(self):
        rng = np.random.RandomState(0)
        # dominant run at X=5 along Z[0,30] + a sparser parallel run at X=8 + noise
        main = np.column_stack([5 + rng.normal(0, 0.1, 800), rng.uniform(0, 30, 800)])
        side = np.column_stack([8 + rng.normal(0, 0.1, 200), rng.uniform(5, 15, 200)])
        A, B = run_centerline(np.vstack([main, side]))
        xs = sorted([A[0], B[0]])
        self.assertAlmostEqual((A[0] + B[0]) / 2, 5.0, delta=0.6)        # picked the dense X=5 lane
        zspan = abs(A[1] - B[1])
        self.assertGreater(zspan, 25)                                     # spans the run length

    def test_real_fxx_main_run(self):
        run = main_pipe_run(FXX)
        self.assertEqual(run.shape, (2, 2))
        midx = (run[0, 0] + run[1, 0]) / 2
        self.assertTrue(3.5 < midx < 7.5)                                # main fire main ~X5 (display frame)
        self.assertGreater(abs(run[0, 1] - run[1, 1]), 25)               # long Z run

    def test_endpoints_finite(self):
        run = main_pipe_run(FXX)
        self.assertTrue(np.isfinite(run).all())

    def test_L_polyline_synthetic(self):
        rng = np.random.RandomState(1)
        # main run X=5, Z[0,30] + branch at corner Z=5 going -X to X=0
        main = np.column_stack([5 + rng.normal(0, 0.08, 4000), rng.uniform(0, 30, 4000)])
        branch = np.column_stack([rng.uniform(0, 5, 300), 8 + rng.normal(0, 0.08, 300)])
        poly = run_L_polyline(np.vstack([main, branch]))
        self.assertEqual(len(poly), 3)                       # start, corner, branch_end
        # start far from corner (high Z end, since branch is at Z=5 near the low end)
        self.assertGreater(abs(poly[0][1] - poly[1][1]), 10)
        # branch_end departs in X from the corner
        self.assertGreater(abs(poly[2][0] - poly[1][0]), 2)

    def test_L_polyline_real_fxx(self):
        poly = main_pipe_run_L(FXX)
        self.assertIn(len(poly), (2, 3))
        self.assertTrue(np.isfinite(poly).all())

    def test_turn_fraction_detects_corner(self):
        # L-trajectory: straight 0..20 along Z, then turn left along X for 5 (turn at ~80%)
        seg1 = np.column_stack([np.full(40, 5.0), np.linspace(0, 20, 40)])
        seg2 = np.column_stack([np.linspace(5, 0, 10), np.full(10, 20.0)])
        frac, ang = trajectory_turn_fraction(np.vstack([seg1, seg2]))
        self.assertGreater(ang, 60)                         # detects the ~90° turn
        self.assertTrue(0.6 < frac < 0.95)                  # turn near the end

    def test_turn_fraction_picks_different_branch(self):
        # two branches; turn_fraction near the start should pick the earlier corner
        rng = np.random.RandomState(2)
        main = np.column_stack([5 + rng.normal(0, 0.08, 4000), rng.uniform(0, 30, 4000)])
        b_near = np.column_stack([rng.uniform(0, 5, 300), 25 + rng.normal(0, 0.08, 300)])   # near A-start corner
        b_far = np.column_stack([rng.uniform(0, 5, 300), 6 + rng.normal(0, 0.08, 300)])
        xz = np.vstack([main, b_near, b_far])
        p_lowfrac = run_L_polyline(xz, turn_fraction=0.2)
        p_highfrac = run_L_polyline(xz, turn_fraction=0.9)
        # different turn fractions select corners at different Z
        self.assertNotAlmostEqual(p_lowfrac[1][1], p_highfrac[1][1], delta=3)


if __name__ == "__main__":
    unittest.main()

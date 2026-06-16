"""TDD for auto pipe-path centerline detection."""
import glob
import os
import unittest

import numpy as np

from scan2bim.pipe_path import main_pipe_run, run_centerline

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


if __name__ == "__main__":
    unittest.main()

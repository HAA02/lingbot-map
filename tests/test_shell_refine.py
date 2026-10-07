"""Shell ICP must pass the explicit-accept gate before it counts as placed."""
import unittest

import numpy as np

from scan2bim.shell_refine import propose_shell_refine


class TestProposeShellRefine(unittest.TestCase):
    def test_perfect_overlap_stays_hold_until_explicit_accept(self):
        rng = np.random.default_rng(0)
        target = rng.normal(size=(40, 3))
        held = propose_shell_refine(target, target)
        self.assertEqual(held["status"], "hold")
        self.assertIsNone(held["accepted_id"])
        self.assertEqual(held["best_id"], "shell-icp")
        self.assertFalse(held["fit"]["accepted"])
        self.assertLess(held["fit"]["rmse"], 1e-8)

        ok = propose_shell_refine(target, target, accept_id="shell-icp")
        self.assertEqual(ok["status"], "ok")
        self.assertEqual(ok["accepted_id"], "shell-icp")
        self.assertFalse(ok["fit"]["accepted"])

    def test_unknown_accept_id_rejects(self):
        extra = np.array(
            [[0.2, 0.1, 0.0], [1.0, 0.4, 0.2], [0.3, 1.2, 0.5],
             [1.4, 0.2, 1.1], [0.6, 1.5, 0.4], [1.2, 1.1, 1.3]],
            dtype=np.float64,
        )
        out = propose_shell_refine(extra, extra, accept_id="other")
        self.assertEqual(out["status"], "reject")
        self.assertIsNone(out["accepted_id"])
        self.assertEqual(out["best_id"], "shell-icp")

"""TDD for FR-3 daily progress ledger.

Per-element coverage (fraction of the element seen by the registered scan) →
status (not_observed / in_progress / observed). A dated ledger accumulates
statuses; day-over-day diff surfaces newly-advanced elements (자동 실적).
"""
import unittest

import numpy as np

from scan2bim.progress import (build_ledger, classify_status, diff_ledgers,
                               element_coverage)


class TestElementCoverage(unittest.TestCase):
    def setUp(self):
        self.elemA = np.random.default_rng(1).uniform(0, 1, (200, 3))
        self.elemB = self.elemA + np.array([10.0, 0, 0])  # far from any scan

    def test_observed_when_scan_covers(self):
        scan = self.elemA + np.random.default_rng(2).normal(0, 0.02, self.elemA.shape)
        cov = element_coverage(self.elemA, scan, radius=0.1)
        self.assertGreater(cov, 0.9)
        self.assertEqual(classify_status(cov), "observed")

    def test_not_observed_when_absent(self):
        scan = self.elemA  # covers A only
        cov = element_coverage(self.elemB, scan, radius=0.1)
        self.assertLess(cov, 0.1)
        self.assertEqual(classify_status(cov), "not_observed")

    def test_partial_is_in_progress(self):
        self.assertEqual(classify_status(0.3), "in_progress")


class TestLedger(unittest.TestCase):
    def test_counts(self):
        led = build_ledger("2026-06-16", {"A": 0.9, "B": 0.0, "C": 0.3})
        self.assertEqual(led["counts"]["observed"], 1)
        self.assertEqual(led["counts"]["not_observed"], 1)
        self.assertEqual(led["counts"]["in_progress"], 1)

    def test_diff_detects_new_completion(self):
        l1 = build_ledger("2026-06-16", {"A": 0.9, "B": 0.0})
        l2 = build_ledger("2026-06-17", {"A": 0.9, "B": 0.8})
        d = diff_ledgers(l1, l2)
        self.assertIn("B", d["new_observed"])
        self.assertTrue(any(x["guid"] == "B" for x in d["advanced"]))
        self.assertEqual(d["regressed"], [])


if __name__ == "__main__":
    unittest.main()

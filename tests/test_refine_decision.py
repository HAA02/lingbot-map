"""Tests for scan2bim.refine_decision — explicit-accept gate for refine candidates."""
import copy
import unittest

from scan2bim.refine_decision import decide_refine


def _c(cid, rmse, inlier=0.8):
    return {"candidate_id": cid, "rmse": rmse, "inlier_ratio": inlier}


class TestDecideRefine(unittest.TestCase):
    def test_empty_candidates_hold(self):
        # Rule 1
        out = decide_refine([])
        self.assertEqual(out["status"], "hold")
        self.assertIsNone(out["accepted_id"])
        self.assertIsNone(out["best_id"])
        self.assertFalse(out["ambiguous"])
        self.assertEqual(out["candidates"], [])

    def test_best_id_lowest_rmse_then_inlier_then_lex(self):
        # Rule 2
        cands = [_c("z", 0.20, 0.9), _c("a", 0.10, 0.5), _c("m", 0.15, 0.99)]
        out = decide_refine(cands)
        self.assertEqual(out["best_id"], "a")

        tied_rmse = [_c("b", 0.10, 0.6), _c("a", 0.10, 0.9), _c("c", 0.10, 0.9)]
        out = decide_refine(tied_rmse)
        # same rmse -> higher inlier; a and c both 0.9 -> lex "a"
        self.assertEqual(out["best_id"], "a")

        same_all_but_id = [_c("m", 0.05, 0.7), _c("k", 0.05, 0.7)]
        out = decide_refine(same_all_but_id)
        self.assertEqual(out["best_id"], "k")

    def test_ambiguous_relative_margin(self):
        # Rule 3
        close = [_c("best", 1.00), _c("near", 1.04)]  # 0.04 < default 0.05
        out = decide_refine(close)
        self.assertTrue(out["ambiguous"])
        self.assertEqual(out["best_id"], "best")

        far = [_c("best", 1.00), _c("far", 1.10)]  # 0.10 >= 0.05
        out = decide_refine(far)
        self.assertFalse(out["ambiguous"])

        equal = [_c("x", 0.3, 0.8), _c("y", 0.3, 0.7)]
        out = decide_refine(equal)
        self.assertTrue(out["ambiguous"])

        single = [_c("only", 0.1)]
        out = decide_refine(single)
        self.assertFalse(out["ambiguous"])

    def test_no_accept_id_always_hold(self):
        # Rule 4 — invariant: never auto-ok
        out = decide_refine([_c("only", 0.01, 0.99)])
        self.assertEqual(out["status"], "hold")
        self.assertIsNone(out["accepted_id"])
        self.assertEqual(out["best_id"], "only")
        self.assertFalse(out["ambiguous"])

        out = decide_refine([_c("a", 0.1), _c("b", 0.5)], accept_id=None)
        self.assertEqual(out["status"], "hold")
        self.assertIsNone(out["accepted_id"])

    def test_explicit_accept_ok(self):
        # Rule 5 — ambiguous does not block explicit accept
        cands = [_c("p", 1.00), _c("q", 1.02)]
        out = decide_refine(cands, accept_id="p")
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["accepted_id"], "p")
        self.assertTrue(out["ambiguous"])
        self.assertEqual(out["best_id"], "p")

    def test_unknown_accept_id_reject(self):
        # Rule 6 — no silent fallback to hold or best_id
        cands = [_c("p", 0.1), _c("q", 0.8)]
        out = decide_refine(cands, accept_id="nope")
        self.assertEqual(out["status"], "reject")
        self.assertIsNone(out["accepted_id"])
        self.assertEqual(out["best_id"], "p")

    def test_explicit_accept_worse_candidate(self):
        cands = [_c("good", 0.05, 0.95), _c("worse", 0.40, 0.40)]
        out = decide_refine(cands, accept_id="worse")
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["accepted_id"], "worse")
        self.assertEqual(out["best_id"], "good")

    def test_input_list_and_dicts_unchanged(self):
        cands = [_c("b", 0.20), _c("a", 0.10)]
        snapshot = copy.deepcopy(cands)
        obj_ids = [id(c) for c in cands]
        out = decide_refine(cands, accept_id="b")
        self.assertEqual(cands, snapshot)
        self.assertEqual([id(c) for c in cands], obj_ids)
        self.assertEqual([c["candidate_id"] for c in out["candidates"]], ["b", "a"])
        self.assertIsNot(out["candidates"], cands)


if __name__ == "__main__":
    unittest.main()

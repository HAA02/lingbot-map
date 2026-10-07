"""Geometric refine candidate gate: never auto-confirm a video-only alignment.

On repetitive geometry a refine (ICP / TEASER / COLMAP) can yield several
plausible poses. This module ranks them and applies the design invariant:
candidates may be stored; only an explicit accept_id confirms. It does not run ICP.
"""


def _rank_key(c: dict) -> tuple:
    return (float(c["rmse"]), -float(c["inlier_ratio"]), str(c["candidate_id"]))


def decide_refine(
    candidates: list[dict],
    *,
    accept_id: str | None = None,
    margin: float = 0.05,
) -> dict:
    """Rank refine candidates and decide hold / ok / reject.

    Never auto-confirms: accept_id is None always yields hold, even for a
    single unambiguous candidate. Does not mutate ``candidates`` or its dicts.
    """
    kept = list(candidates)
    if not kept:
        return {
            "status": "hold",
            "accepted_id": None,
            "best_id": None,
            "ambiguous": False,
            "candidates": [],
        }

    ranked = sorted(kept, key=_rank_key)
    best = ranked[0]
    best_id = best["candidate_id"]

    ambiguous = False
    if len(ranked) >= 2:
        rmse_best = float(best["rmse"])
        rmse_second = float(ranked[1]["rmse"])
        ambiguous = (rmse_second - rmse_best) / max(rmse_best, 1e-9) < float(margin)

    if accept_id is None:
        status = "hold"
        accepted = None
    elif any(c["candidate_id"] == accept_id for c in kept):
        status = "ok"
        accepted = accept_id
    else:
        status = "reject"
        accepted = None

    return {
        "status": status,
        "accepted_id": accepted,
        "best_id": best_id,
        "ambiguous": ambiguous,
        "candidates": kept,
    }

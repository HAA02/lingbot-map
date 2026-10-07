"""Join rigid shell ICP to the explicit-accept gate.

ICP only proposes a candidate. decide_refine never auto-confirms it.
"""
from __future__ import annotations

from scan2bim.refine_decision import decide_refine
from scan2bim.shell_icp import refine_rigid_icp


def propose_shell_refine(
    source,
    target,
    init=None,
    *,
    candidate_id: str = "shell-icp",
    accept_id: str | None = None,
    margin: float = 0.05,
    **icp_kw,
) -> dict:
    """Refine source onto target, then hold unless accept_id names this candidate.

    The fit dict keeps accepted=False. Status becomes \"ok\" only when the caller
    passes accept_id equal to candidate_id.
    """
    fit = refine_rigid_icp(source, target, init, **icp_kw)
    candidate = {
        "candidate_id": candidate_id,
        "rmse": fit["rmse"],
        "inlier_ratio": fit["inlier_ratio"],
        "rotation": fit["rotation"],
        "translation": fit["translation"],
    }
    decision = decide_refine([candidate], accept_id=accept_id, margin=margin)
    decision["fit"] = fit
    return decision

"""FR-3 daily progress ledger — turn registered coverage into 자동 실적.

Per design element, coverage = fraction of the element's sampled surface seen
by the (registered) scan within a radius. Coverage → status. A dated ledger
records every element's status; diffing consecutive days surfaces what newly
advanced (e.g. installed/observed today) for automatic progress registration.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.spatial import cKDTree

# coverage thresholds
_OBSERVED = 0.5
_PARTIAL = 0.1
_ORDER = {"not_observed": 0, "in_progress": 1, "observed": 2}


def element_coverage(element_points, scan_points, radius: float = 0.15) -> float:
    """Fraction of element sample points with a scan point within `radius`."""
    elem = np.asarray(element_points, dtype=np.float64)
    scan = np.asarray(scan_points, dtype=np.float64)
    if len(elem) == 0 or len(scan) == 0:
        return 0.0
    d, _ = cKDTree(scan).query(elem, workers=-1)
    return float((d <= radius).mean())


def classify_status(coverage: float, *, observed: float = _OBSERVED, partial: float = _PARTIAL) -> str:
    if coverage >= observed:
        return "observed"
    if coverage >= partial:
        return "in_progress"
    return "not_observed"


def build_ledger(date: str, element_coverage_by_guid: dict, **thresholds) -> dict:
    """Dated ledger of per-element status from {guid: coverage}."""
    statuses = {g: classify_status(c, **thresholds) for g, c in element_coverage_by_guid.items()}
    return {
        "date": date,
        "statuses": statuses,
        "coverage": {g: round(float(c), 3) for g, c in element_coverage_by_guid.items()},
        "counts": dict(Counter(statuses.values())),
    }


def diff_ledgers(prev: dict, cur: dict) -> dict:
    """Day-over-day change. advanced/regressed list status transitions;
    new_observed lists guids that reached 'observed' today."""
    advanced, regressed = [], []
    prev_st = prev.get("statuses", {})
    for g, s in cur.get("statuses", {}).items():
        ps = prev_st.get(g, "not_observed")
        if _ORDER[s] > _ORDER.get(ps, 0):
            advanced.append({"guid": g, "from": ps, "to": s})
        elif _ORDER[s] < _ORDER.get(ps, 0):
            regressed.append({"guid": g, "from": ps, "to": s})
    return {
        "advanced": advanced,
        "regressed": regressed,
        "new_observed": [x["guid"] for x in advanced if x["to"] == "observed"],
    }

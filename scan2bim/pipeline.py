"""scan2bim end-to-end orchestration — ties the modules into the PRD chain:

    (registered) scan  →  per-element coverage  →  daily ledger (자동 실적)
                       →  color↔system attribution (텍스트 없이)

apply_sim3 places the scan onto the design frame using an alignment from
register_sim3 (FR-2.1). analyze_progress then computes per-element coverage
(FR-3) and color-cluster→system matches (FR-2.2) in one call.
"""
from __future__ import annotations

import numpy as np

from scan2bim.matching import match_to_design, segment_by_color
from scan2bim.progress import build_ledger, element_coverage


def apply_sim3(points, alignment) -> np.ndarray:
    s = float(alignment["scale"])
    R = np.asarray(alignment["rotation"], dtype=np.float64)
    t = np.asarray(alignment["translation"], dtype=np.float64)
    return s * (np.asarray(points, dtype=np.float64) @ R.T) + t


def analyze_progress(elements, scan_points, scan_colors, color_index, date,
                     *, alignment=None, radius: float = 0.15) -> dict:
    """One-shot progress analysis.

    elements: [{guid, points (Nx3, model frame), ...}]
    scan_points/scan_colors: reconstructed scan (recon frame); alignment places
    it onto the model frame. Returns {date, ledger, color_matches}.
    """
    scan = apply_sim3(scan_points, alignment) if alignment else np.asarray(scan_points, dtype=np.float64)
    covs = {e["guid"]: element_coverage(e["points"], scan, radius) for e in elements}
    ledger = build_ledger(date, covs)
    color_matches = []
    if scan_colors is not None and len(scan_colors):
        color_matches = match_to_design(segment_by_color(scan_points, scan_colors), color_index)
    return {"date": date, "ledger": ledger, "color_matches": color_matches}

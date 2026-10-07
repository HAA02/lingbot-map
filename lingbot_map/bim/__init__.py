"""BIM alignment, projection, and progress analysis modules.

This package absorbs the BIM-VAMS workflow into lingbot-map: take a
lingbot-map scene export plus a BIM (geometry + metadata), align the two
coordinate systems with manual correspondence points, project BIM objects
into the camera frames, and report per-object installation status.

See `docs/bim-vams-absorption-plan.md` for the design.
"""
from .metadata import BimObject, BimModel, load_bim_metadata
from .alignment import (
    Sim3,
    solve_sim3_umeyama,
    AlignmentResult,
    start_direction_ok,
    yaw_candidates_from_user_direction,
    solve_scan_to_model_alignment,
    alignment_permits_coverage_analysis,
    auto_geometric_alignment_record,
)
from .projection import project_object_to_frame, ProjectionResult
from .progress import (
    aggregate_evidence,
    decide_status,
    infer_topology,
    build_progress_report,
)

__all__ = [
    "BimObject", "BimModel", "load_bim_metadata",
    "Sim3", "solve_sim3_umeyama", "AlignmentResult",
    "start_direction_ok", "yaw_candidates_from_user_direction",
    "solve_scan_to_model_alignment", "alignment_permits_coverage_analysis",
    "auto_geometric_alignment_record",
    "project_object_to_frame", "ProjectionResult",
    "aggregate_evidence", "decide_status", "infer_topology",
    "build_progress_report",
]

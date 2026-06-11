"""Per-object evidence aggregation, status decision, and progress report.

This is the MVP decision layer. Inputs are projection-based observation
candidates; the visual-match part is left as a hook so that an external
detector can plug in later. For now we use *projection consistency* as the
primary direct evidence signal, with a confidence floor controlled by the
caller.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .metadata import BimModel
from .projection import ProjectionResult


Status = str  # "installed" | "detected" | "inferred" | "not_detected" | "unknown" | "needs_review"


@dataclass
class ObjectEvidence:
    guid: str
    frames_seen: int = 0
    best_score: float = 0.0
    mean_score: float = 0.0
    frame_numbers: list[int] = field(default_factory=list)
    detector_scores: list[float] = field(default_factory=list)


def aggregate_evidence(
    projections: Iterable[ProjectionResult],
    detector: Callable[[ProjectionResult], float] | None = None,
) -> dict[str, ObjectEvidence]:
    """Aggregate per-frame projections into per-object evidence.

    `detector(projection) -> match_score in [0,1]` is an optional hook for a
    real visual matcher. If omitted, the projection score is used directly.
    """
    out: dict[str, ObjectEvidence] = defaultdict(lambda: ObjectEvidence(guid=""))
    for p in projections:
        ev = out[p.guid]
        ev.guid = p.guid
        if not p.visible:
            continue
        d = detector(p) if detector is not None else p.score
        ev.frames_seen += 1
        ev.frame_numbers.append(p.frame_number)
        ev.detector_scores.append(float(d))
        if d > ev.best_score:
            ev.best_score = float(d)
    for ev in out.values():
        if ev.detector_scores:
            ev.mean_score = sum(ev.detector_scores) / len(ev.detector_scores)
    return dict(out)


@dataclass
class StatusDecision:
    guid: str
    status: Status
    confidence: float
    method: str            # "direct_observation" | "topology_inference" | "absent"
    path: list[str] = field(default_factory=list)
    evidence_frames: list[int] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "guid": self.guid,
            "status": self.status,
            "confidence": float(self.confidence),
            "method": self.method,
            "path": list(self.path),
            "evidence_frames": list(self.evidence_frames),
            "notes": self.notes,
        }


def decide_status(
    model: BimModel,
    evidence: dict[str, ObjectEvidence],
    visibility_by_guid: dict[str, int] | None = None,
    *,
    install_thresh: float = 0.6,
    detect_thresh: float = 0.3,
    min_frames: int = 2,
) -> dict[str, StatusDecision]:
    """Decide per-object status from direct evidence (no topology yet)."""
    decisions: dict[str, StatusDecision] = {}
    for guid, obj in model.objects.items():
        ev = evidence.get(guid)
        seen = visibility_by_guid.get(guid, 0) if visibility_by_guid else (ev.frames_seen if ev else 0)

        if ev is None or ev.frames_seen == 0:
            if seen == 0:
                decisions[guid] = StatusDecision(
                    guid=guid, status="unknown", confidence=0.0,
                    method="absent", notes="never in view",
                )
            else:
                decisions[guid] = StatusDecision(
                    guid=guid, status="not_detected", confidence=0.5,
                    method="direct_observation",
                    notes=f"visible in {seen} frames but no positive evidence",
                )
            continue

        if ev.frames_seen >= min_frames and ev.mean_score >= install_thresh:
            status = "installed"
            conf = min(1.0, 0.5 + 0.5 * ev.mean_score)
        elif ev.best_score >= detect_thresh:
            status = "detected"
            conf = ev.best_score
        else:
            status = "needs_review"
            conf = ev.mean_score

        decisions[guid] = StatusDecision(
            guid=guid, status=status, confidence=float(conf),
            method="direct_observation",
            evidence_frames=list(ev.frame_numbers),
        )
    return decisions


def infer_topology(
    model: BimModel,
    decisions: dict[str, StatusDecision],
    *,
    max_hops: int = 2,
    confidence_decay: float = 0.6,
) -> dict[str, StatusDecision]:
    """Promote `not_detected`/`unknown`/`needs_review` to `inferred` when
    both endpoints of a short connector path are `installed`/`detected`.

    Conservative: only fires when the object lies on a path of length<=max_hops
    between two confirmed neighbors.
    """
    confirmed = {g for g, d in decisions.items() if d.status in ("installed", "detected")}

    for guid, dec in decisions.items():
        if dec.status in ("installed", "detected"):
            continue
        obj = model.objects.get(guid)
        if not obj:
            continue

        # BFS up to max_hops from `guid`, looking for two distinct confirmed objects on opposite sides.
        # Simplified rule: if any two neighbors (within max_hops) are confirmed, treat as inferred.
        found: list[str] = []
        seen = {guid}
        frontier = [(guid, 0)]
        while frontier and len(found) < 2:
            node, depth = frontier.pop(0)
            if depth >= max_hops:
                continue
            for nb in model.neighbors(node):
                if nb in seen:
                    continue
                seen.add(nb)
                if nb in confirmed:
                    found.append(nb)
                    if len(found) >= 2:
                        break
                frontier.append((nb, depth + 1))

        if len(found) >= 2:
            base_conf = min(decisions[found[0]].confidence, decisions[found[1]].confidence)
            decisions[guid] = StatusDecision(
                guid=guid,
                status="inferred",
                confidence=float(base_conf * confidence_decay),
                method="topology_inference",
                path=found,
                evidence_frames=dec.evidence_frames,
                notes=f"inferred from {found[0]}..{found[1]}",
            )

    return decisions


def build_progress_report(
    model: BimModel,
    decisions: dict[str, StatusDecision],
) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    by_system: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_zone: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    review_queue: list[dict[str, Any]] = []

    for guid, dec in decisions.items():
        counts[dec.status] += 1
        obj = model.objects.get(guid)
        if obj:
            by_system[obj.system or "_unassigned"][dec.status] += 1
            by_zone[obj.zone or "_unassigned"][dec.status] += 1
            by_category[obj.category][dec.status] += 1
        if dec.status in ("needs_review", "not_detected"):
            review_queue.append({
                "guid": guid,
                "status": dec.status,
                "confidence": dec.confidence,
                "notes": dec.notes,
            })

    total = max(1, len(decisions))
    installed_like = counts.get("installed", 0) + counts.get("inferred", 0)

    def _rates(group: dict[str, dict[str, int]]) -> dict[str, dict[str, Any]]:
        out = {}
        for key, c in group.items():
            t = max(1, sum(c.values()))
            i = c.get("installed", 0) + c.get("inferred", 0)
            out[key] = {"counts": dict(c), "install_rate": i / t}
        return out

    review_queue.sort(key=lambda r: r["confidence"])

    return {
        "model_id": model.model_id,
        "total_objects": len(decisions),
        "overall_install_rate": installed_like / total,
        "status_counts": dict(counts),
        "by_system": _rates(by_system),
        "by_zone": _rates(by_zone),
        "by_category": _rates(by_category),
        "review_queue": review_queue,
    }

# lingbot_map.bim — BIM absorption layer

Implements the MVP from `docs/bim-vams-absorption-plan.md`:

1. **Export standardization** — `lingbot_map.export.scene_export.export_scene` writes
   `scene_export.json` + `point_cloud_sampled.ply` from a `predictions` dict.
2. **Sim(3) alignment** — `alignment.solve_sim3_umeyama` (≥3 correspondences) returns
   `(scale, R, t)`, RMSE, residuals, and a `green / yellow / review` quality tag.
3. **BIM metadata** — `metadata.load_bim_metadata` parses `{guid, category, system, zone,
   center|start|end|bbox, connectors}` records.
4. **Projection** — `projection.project_object_to_frame` projects each BIM object's
   sample points into a frame and reports a `ProjectionResult` (bbox, depth, score).
5. **Decision + topology** — `progress.aggregate_evidence` → `decide_status` →
   `infer_topology` produces per-object `StatusDecision` with provenance.
6. **Report** — `progress.build_progress_report` rolls up overall / per-system /
   per-zone / per-category install rates and a sorted review queue.

## End-to-end CLI

```bash
python tools/analyze_bim_progress.py \
    --scene out/scene_export.json \
    --bim   data/bim_metadata.json \
    --correspondences data/correspondences.json \
    --output out/
```

Writes `alignment.json`, `object_statuses.json`, `progress_report.json`. Every
status carries its decision `method` (`direct_observation` | `topology_inference`)
and the supporting `evidence_frames` (or the topology `path`), so every report
number is traceable to a BIM GUID and source frame.

## Status taxonomy

| status         | meaning                                                  |
| -------------- | -------------------------------------------------------- |
| `installed`    | direct evidence above threshold across ≥`min_frames`     |
| `detected`     | best score over `detect_thresh` but mean below install   |
| `inferred`     | not directly seen; two confirmed neighbors within hops   |
| `not_detected` | should have been visible (≥1 candidate frame) but wasn't |
| `unknown`      | never in view / unobservable                             |
| `needs_review` | scores ambiguous; queued for manual inspection           |

## What is not in MVP

Automatic global BIM localization, per-frame visual detectors, geometry-aware
occlusion, mobile/real-time pipeline, and an interactive 3D viewer. The
detector is a `Callable[[ProjectionResult], float]` hook in
`aggregate_evidence(..., detector=...)` so a real visual matcher can be plugged
in without changing the decision/topology layer.

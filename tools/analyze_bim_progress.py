"""CLI: turn a lingbot scene_export + BIM metadata into a progress report.

Usage:
    python tools/analyze_bim_progress.py \
        --scene out/scene_export.json \
        --bim   data/bim_metadata.json \
        --correspondences data/correspondences.json \
        --output out/

Produces in `--output`:
    alignment.json       Sim(3) transform + RMSE + per-point residuals
    object_statuses.json per-object status + evidence
    progress_report.json overall + by-system / by-zone / by-category rates
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# Allow running as a script from the repo root.
_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_THIS)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from lingbot_map.bim.metadata import load_bim_metadata
from lingbot_map.bim.alignment import solve_sim3_umeyama, load_correspondences, Sim3
from lingbot_map.bim.projection import project_object_to_frame
from lingbot_map.bim.progress import (
    aggregate_evidence, decide_status, infer_topology, build_progress_report,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="LingBot scene + BIM -> progress report")
    ap.add_argument("--scene", required=True, help="path to scene_export.json")
    ap.add_argument("--bim", required=True, help="path to BIM metadata JSON")
    ap.add_argument("--correspondences", required=True,
                    help="path to correspondences JSON (lingbot<->bim pairs)")
    ap.add_argument("--output", required=True, help="output directory")
    ap.add_argument("--image-size", type=str, default=None,
                    help="override frame image size HxW (e.g. 518x518); "
                         "otherwise read from scene_export.source")
    ap.add_argument("--margin-px", type=float, default=0.0)
    ap.add_argument("--install-thresh", type=float, default=0.6)
    ap.add_argument("--detect-thresh", type=float, default=0.3)
    ap.add_argument("--min-frames", type=int, default=2)
    ap.add_argument("--max-hops", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with open(args.scene) as f:
        scene = json.load(f)
    bim = load_bim_metadata(args.bim)

    if args.image_size:
        H, W = (int(x) for x in args.image_size.lower().split("x"))
    else:
        src = scene.get("source", {})
        size = src.get("image_size") or [518, 518]
        H, W = int(size[0]), int(size[1])

    # 1) alignment
    src_pts, dst_pts = load_correspondences(args.correspondences)
    alignment = solve_sim3_umeyama(src_pts, dst_pts)
    print(f"[align] n={alignment.n_correspondences} rmse={alignment.rmse:.4f} "
          f"quality={alignment.quality} scale={alignment.transform.scale:.4f}")
    alignment.source["correspondences_path"] = args.correspondences
    with open(os.path.join(args.output, "alignment.json"), "w") as f:
        json.dump(alignment.to_dict(), f, indent=2)

    if alignment.quality == "review":
        print(f"[align] WARNING quality=review (RMSE={alignment.rmse:.3f}m). "
              "Proceeding but statuses should be reviewed manually.")

    T: Sim3 = alignment.transform

    # 2) project BIM objects into each frame (in BIM coords)
    projections = []
    visibility_count: dict[str, int] = {}
    for frame in scene["frames"]:
        K = np.array(frame["intrinsic"], dtype=np.float64)
        c2w_lingbot = np.array(frame["extrinsic_c2w"], dtype=np.float64)
        # ensure (3,4)
        if c2w_lingbot.shape == (4, 4):
            c2w_lingbot = c2w_lingbot[:3, :4]
        c2w_bim = T.apply_pose_c2w(c2w_lingbot)
        fnum = int(frame["frame_number"])
        for obj in bim:
            r = project_object_to_frame(
                obj, K, c2w_bim, (H, W), fnum, margin_px=args.margin_px,
            )
            projections.append(r)
            if r.visible:
                visibility_count[obj.guid] = visibility_count.get(obj.guid, 0) + 1

    # 3) per-object evidence aggregation
    evidence = aggregate_evidence(projections)
    decisions = decide_status(
        bim, evidence, visibility_by_guid=visibility_count,
        install_thresh=args.install_thresh,
        detect_thresh=args.detect_thresh,
        min_frames=args.min_frames,
    )

    # 4) topology inference
    decisions = infer_topology(bim, decisions, max_hops=args.max_hops)

    # 5) write outputs
    with open(os.path.join(args.output, "object_statuses.json"), "w") as f:
        json.dump(
            {"objects": [d.to_dict() for d in decisions.values()]},
            f, indent=2,
        )

    report = build_progress_report(bim, decisions)
    with open(os.path.join(args.output, "progress_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"[report] total={report['total_objects']} "
          f"install_rate={report['overall_install_rate']*100:.1f}% "
          f"counts={report['status_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

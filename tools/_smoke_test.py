"""End-to-end smoke test: synthetic scene -> alignment -> projection -> report.

Run from repo root:
    python tools/_smoke_test.py
"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingbot_map.export.scene_export import export_scene, load_scene
from lingbot_map.bim.metadata import load_bim_metadata
from lingbot_map.bim.alignment import solve_sim3_umeyama
from lingbot_map.bim.projection import project_object_to_frame
from lingbot_map.bim.progress import (
    aggregate_evidence, decide_status, infer_topology, build_progress_report,
)


def main():
    tmp = tempfile.mkdtemp(prefix="lingbot-smoke-")
    print(f"workdir: {tmp}")

    # --- fake predictions (3 frames looking down +Z) ---
    N, H, W = 3, 64, 64
    K = np.array([[80.0, 0, W / 2], [0, 80.0, H / 2], [0, 0, 1]])
    extr = np.tile(np.eye(4)[:3, :4], (N, 1, 1)).astype(np.float64)
    for i in range(N):
        extr[i, 0, 3] = i * 0.5  # camera slides along +X
    intr = np.tile(K[None], (N, 1, 1))
    wp = np.random.default_rng(0).normal(size=(N, H, W, 3)).astype(np.float32)
    wpc = np.ones((N, H, W), dtype=np.float32) * 0.5
    images = (np.random.default_rng(1).random((N, H, W, 3)) * 255).astype(np.uint8)

    predictions = {
        "extrinsic": extr, "intrinsic": intr,
        "world_points": wp, "world_points_conf": wpc, "images": images,
    }
    exp = export_scene(
        predictions, output_dir=tmp, scene_id="smoke",
        source_info={"image_size": [H, W]},
    )
    assert os.path.exists(os.path.join(tmp, "scene_export.json"))
    assert os.path.exists(os.path.join(tmp, "point_cloud_sampled.ply"))
    print(f"[export] frames={len(exp.frames)} pc={exp.point_cloud['num_points']}")

    # --- fake BIM: 3 objects forming a chain A-B-C ---
    bim_data = {
        "model_id": "smoke-bim",
        "coordinate_system": "bim_world",
        "objects": [
            {"guid": "A", "category": "Pipe", "system": "S1", "zone": "Z1",
             "center": [0, 0, 5], "connectors": ["B"]},
            {"guid": "B", "category": "Pipe", "system": "S1", "zone": "Z1",
             "center": [0.5, 0, 5], "connectors": ["A", "C"]},
            {"guid": "C", "category": "Pipe", "system": "S1", "zone": "Z1",
             "center": [1.0, 0, 5], "connectors": ["B"]},
            {"guid": "D", "category": "Pipe", "system": "S2", "zone": "Z2",
             "center": [100, 100, 100], "connectors": []},
        ],
    }
    bim_path = os.path.join(tmp, "bim.json")
    with open(bim_path, "w") as f:
        json.dump(bim_data, f)
    bim = load_bim_metadata(bim_path)

    # --- identity correspondences (lingbot == bim coords for the smoke test) ---
    pairs = {
        "pairs": [
            {"lingbot": [0, 0, 5], "bim": [0, 0, 5]},
            {"lingbot": [1, 0, 5], "bim": [1, 0, 5]},
            {"lingbot": [0, 1, 5], "bim": [0, 1, 5]},
            {"lingbot": [0, 0, 6], "bim": [0, 0, 6]},
        ],
    }
    src = np.array([p["lingbot"] for p in pairs["pairs"]])
    dst = np.array([p["bim"] for p in pairs["pairs"]])
    align = solve_sim3_umeyama(src, dst)
    print(f"[align] rmse={align.rmse:.6f} quality={align.quality} scale={align.transform.scale:.4f}")
    assert align.rmse < 1e-6
    assert align.quality == "green"

    # --- projection ---
    scene = load_scene(os.path.join(tmp, "scene_export.json"))
    T = align.transform
    projections = []
    vis_count = {}
    for frame in scene["frames"]:
        Kf = np.array(frame["intrinsic"])
        c2w = np.array(frame["extrinsic_c2w"])
        if c2w.shape == (4, 4):
            c2w = c2w[:3, :4]
        c2w_bim = T.apply_pose_c2w(c2w)
        for obj in bim:
            r = project_object_to_frame(obj, Kf, c2w_bim, (H, W), int(frame["frame_number"]))
            projections.append(r)
            if r.visible:
                vis_count[obj.guid] = vis_count.get(obj.guid, 0) + 1
    print(f"[project] visibility_count={vis_count}")
    assert "A" in vis_count and "C" in vis_count, "A/C should be visible in synthetic scene"
    assert "D" not in vis_count, "D is far away and should not be visible"

    # --- evidence + decisions ---
    # Force B to NOT be visible (simulate occlusion) by pretending detector saw A & C strongly only.
    def detector(p):
        # strong signal for A and C, weak/none for B even if visible.
        if p.guid in ("A", "C"):
            return 0.9
        if p.guid == "B":
            return 0.0
        return p.score

    ev = aggregate_evidence(projections, detector=detector)
    decisions = decide_status(bim, ev, visibility_by_guid=vis_count, install_thresh=0.6, min_frames=1)
    print("[decide] before topology:", {g: d.status for g, d in decisions.items()})
    decisions = infer_topology(bim, decisions, max_hops=2)
    print("[topology] after:", {g: d.status for g, d in decisions.items()})

    assert decisions["A"].status == "installed"
    assert decisions["C"].status == "installed"
    assert decisions["B"].status == "inferred", f"B should be inferred, got {decisions['B'].status}"
    assert decisions["D"].status == "unknown"

    report = build_progress_report(bim, decisions)
    print(f"[report] overall={report['overall_install_rate']:.2f} counts={report['status_counts']}")
    assert report["status_counts"].get("installed", 0) == 2
    assert report["status_counts"].get("inferred", 0) == 1
    assert report["status_counts"].get("unknown", 0) == 1

    print("OK")


if __name__ == "__main__":
    main()

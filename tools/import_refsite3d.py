#!/usr/bin/env python3
"""Import RefSite3D phase_01 PM1 (open dataset) into lingbot-map uploads + models.

Downloads only the needed OBJ+PLY from Zenodo via HTTP range (no full 2.2GB zip).
RefSite3D has no raw MP4 in the archive — scan is NeRF/photogrammetry PLY derived
from iPhone X capture. We inject that cloud as LBP2 + a stub walkthrough MP4 so
coverage/coplay can run without re-running GPU reconstruction.

Usage:
    .venv/bin/python tools/import_refsite3d.py
    .venv/bin/python tools/import_refsite3d.py --base-url http://127.0.0.1:8767
"""
from __future__ import annotations

import argparse
import base64
import json
import struct
import time
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pygltflib
import trimesh

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "refsite3d"
MODELS = ROOT / "models"
UPLOADS = ROOT / "realtime" / "_uploads"

ZENODO_URL = "https://zenodo.org/api/records/20285732/files/RefSite3D_v1.0.0.zip/content"
ZENODO_SIZE = 2172793209
OBJ_PATH = "dataset/phases/phase_01/models/full/planned_building_ReStage_Target-X_phase1.obj"
PLY_PATH = (
    "dataset/phases/phase_01/scans/scene_001/"
    "scene_001_pm1_nerf_luma_scene_ReStage_target-x_phase1.ply"
)


class RangeFile:
    def __init__(self, url: str, size: int):
        self.url = url
        self.size = size
        self.pos = 0

    def seek(self, pos: int, whence: int = 0) -> int:
        if whence == 1:
            pos = self.pos + pos
        elif whence == 2:
            pos = self.size + pos
        self.pos = max(0, min(self.size, pos))
        return self.pos

    def tell(self) -> int:
        return self.pos

    def seekable(self) -> bool:
        return True

    def read(self, n: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        if n < 0:
            n = self.size - self.pos
        end = min(self.size - 1, self.pos + n - 1)
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=180) as r:
            data = r.read()
        self.pos += len(data)
        return data


def zenodo_extract(path: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"download {path} -> {out}")
    rf = RangeFile(ZENODO_URL, ZENODO_SIZE)
    with zipfile.ZipFile(rf) as zf:
        out.write_bytes(zf.read(path))


def subsample_points(xyz: np.ndarray, rgb: np.ndarray, n: int, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    if xyz.shape[0] <= n:
        return xyz, rgb
    rng = np.random.default_rng(seed)
    idx = rng.choice(xyz.shape[0], n, replace=False)
    return xyz[idx], rgb[idx]


def look_at_c2w(eye: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    up = np.array([0.0, 1.0, 0.0]) if up is None else up.astype(np.float64)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd) + 1e-12
    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-9:
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, up)
    right /= np.linalg.norm(right) + 1e-12
    up_v = np.cross(right, fwd)
    up_v /= np.linalg.norm(up_v) + 1e-12
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = up_v
    c2w[:3, 2] = -fwd
    c2w[:3, 3] = eye
    return c2w[:3, :4].astype(np.float32)


def orbit_poses(center: np.ndarray, span: float, n: int) -> np.ndarray:
    radius = max(span * 1.2, 2.0)
    height = center[1] + span * 0.15
    poses = []
    for i in range(n):
        th = 2.0 * np.pi * i / n
        eye = center + np.array([radius * np.cos(th), height - center[1], radius * np.sin(th)])
        poses.append(look_at_c2w(eye, center))
    return np.stack(poses, axis=0)


def write_lbp2(path: Path, xyz: np.ndarray, rgb: np.ndarray, poses: np.ndarray, k: np.ndarray) -> None:
    xyz = xyz.astype(np.float32)
    rgb = rgb.astype(np.uint8)
    poses = poses.astype(np.float32)
    k = k.astype(np.float32)
    header = struct.pack("<4sIIII", b"LBP2", 0, xyz.shape[0], poses.shape[0], 1)
    body = xyz.tobytes() + rgb.tobytes() + poses.tobytes() + k.tobytes()
    path.write_bytes(header + body)


def write_stub_mp4(path: Path, n_frames: int = 32, size: tuple[int, int] = (640, 480)) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = size
    vw = cv2.VideoWriter(str(path), fourcc, 8.0, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"VideoWriter failed for {path}")
    for i in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            f"RefSite3D PM1 NeRF scan import frame {i}",
            (20, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        vw.write(frame)
    vw.release()


def patch_glb_guid(glb_path: Path, guid: str = "REFSITE3D-PM1-BUILDING", category: str = "Building") -> None:
    gltf = pygltflib.GLTF2().load(str(glb_path))
    if not gltf.nodes:
        raise RuntimeError(f"GLB has no nodes: {glb_path}")
    for node in gltf.nodes:
        if node.mesh is not None:
            node.extras = {"IfcGUID": guid, "Category": category, "Name": "RefSite3D PM1 planned building"}
            break
    gltf.save(str(glb_path))


def obj_to_glb(obj_path: Path, glb_path: Path) -> dict:
    mesh = trimesh.load(obj_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    bounds = mesh.bounds
    center = mesh.centroid
    span = float(np.linalg.norm(bounds[1] - bounds[0]))
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(glb_path)
    patch_glb_guid(glb_path)
    return {
        "bounds_min": bounds[0].tolist(),
        "bounds_max": bounds[1].tolist(),
        "center": center.tolist(),
        "span": span,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
    }


def write_manifest(model_id: str, glb_name: str, meta: dict) -> None:
    manifest = {
        "id": model_id,
        "model_id": model_id,
        "name": "RefSite3D PM1 (Aachen ReStage)",
        "glb": glb_name,
        "metadata": None,
        "manifest": f"{model_id}.model_manifest.json",
        "units": "m",
        "unit_scale_to_m": 1.0,
        "up_axis": "Z_UP",
        "metadata_kind": "glb_native",
        "glb_axis_transform": "gltf_yup_to_zup",
        "object_count": 1,
        "counts_by_category": {"Building": 1},
        "glb_url": f"/models/{glb_name}",
        "metadata_url": None,
        "manifest_url": f"/models/{model_id}.model_manifest.json",
        "manifest_valid": True,
        "guid_mapping_ratio": 1.0,
        "guid_mapping_status": "direct",
        "guid_mapped_count": 1,
        "guid_total": 1,
        "fallback_geometry": "pag_proxy",
        "fallback_geometry_count": 0,
        "coverage_geometry_source": "glb_guid_mesh",
        "glb_role": "coverage_geometry",
        "proxy_geometry_bounds": {
            "min": meta["bounds_min"],
            "max": meta["bounds_max"],
            "span": (np.array(meta["bounds_max"]) - np.array(meta["bounds_min"])).tolist(),
            "center": meta["center"],
            "diagonal_m": meta["span"],
        },
        "glb_bytes": Path(MODELS / glb_name).stat().st_size,
        "metadata_bytes": 0,
        "source": "RefSite3D phase_01 PM1 open dataset (Zenodo 10.5281/zenodo.20285732)",
    }
    (MODELS / f"{model_id}.model_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def import_upload(model_id: str, n_poses: int = 32, max_points: int = 80000) -> str:
    DATA.mkdir(parents=True, exist_ok=True)
    obj_local = DATA / "planned_building_pm1.obj"
    ply_local = DATA / "scene_001_pm1_nerf.ply"
    if not obj_local.exists():
        zenodo_extract(OBJ_PATH, obj_local)
    if not ply_local.exists():
        zenodo_extract(PLY_PATH, ply_local)

    model_id = model_id
    glb_name = f"{model_id}.glb"
    glb_path = MODELS / glb_name
    print("convert OBJ -> GLB")
    meta = obj_to_glb(obj_local, glb_path)
    write_manifest(model_id, glb_name, meta)

    print("load PLY scan")
    scan = trimesh.load(ply_local, process=False)
    if isinstance(scan, trimesh.PointCloud):
        xyz = np.asarray(scan.vertices, dtype=np.float64)
        rgb = np.asarray(scan.colors[:, :3], dtype=np.uint8) if scan.colors is not None else np.full((len(xyz), 3), 180, np.uint8)
    else:
        xyz = np.asarray(scan.vertices, dtype=np.float64)
        rgb = np.full((len(xyz), 3), 180, np.uint8)
    # RefSite3D PLY covers the whole site; crop to the phase model footprint so
    # orbit poses and auto-placement search stay near the planning GLB.
    bmin = np.asarray(meta["bounds_min"], dtype=np.float64)
    bmax = np.asarray(meta["bounds_max"], dtype=np.float64)
    pad = max(8.0, float(meta["span"]) * 0.5)
    keep = np.all(xyz >= bmin - pad, axis=1) & np.all(xyz <= bmax + pad, axis=1)
    xyz, rgb = xyz[keep], rgb[keep]
    xyz, rgb = subsample_points(xyz, rgb, max_points)
    center = np.asarray(meta["center"], dtype=np.float64)
    span = float(meta["span"])
    poses = orbit_poses(center, span, n_poses)
    k = np.array([600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0], dtype=np.float32)

    ts = int(time.time() * 1000)
    upload_id = f"upload_refsite3d_{ts}"
    UPLOADS.mkdir(parents=True, exist_ok=True)
    mp4 = UPLOADS / f"{upload_id}.mp4"
    lbp2 = UPLOADS / f"{upload_id}.lbp2"
    print("write stub MP4 + LBP2")
    write_stub_mp4(mp4, n_frames=n_poses)
    write_lbp2(lbp2, xyz, rgb, poses, k)

    thumb = cv2.imencode(".jpg", np.zeros((120, 160, 3), dtype=np.uint8))[1]
    (UPLOADS / f"{upload_id}.thumbs.json").write_text(
        json.dumps([base64.b64encode(thumb).decode("ascii")]),
        encoding="utf-8",
    )
    note = {
        "upload_id": upload_id,
        "model_id": model_id,
        "source": "RefSite3D Zenodo 20285732 phase_01 PM1",
        "scan_ply": PLY_PATH,
        "plan_obj": OBJ_PATH,
        "note": "No raw MP4 in dataset; LBP2 from NeRF scene PLY + synthetic orbit poses",
        "points": int(xyz.shape[0]),
        "poses": int(poses.shape[0]),
    }
    (UPLOADS / f"{upload_id}.refsite3d.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    print(json.dumps(note, indent=2))
    return upload_id


def icp_alignment_pairs(xyz: np.ndarray, mesh: trimesh.Trimesh, *, max_pairs: int = 6) -> list[dict]:
    """ICP scan→model then pick spread inlier pairs for Sim(3) save."""
    from scipy.spatial import cKDTree

    mesh_pts, _ = trimesh.sample.sample_surface(mesh, 50_000)
    matrix, _, _ = trimesh.registration.icp(xyz[: min(25_000, len(xyz))], mesh_pts, max_iterations=80)
    scan_t = trimesh.transformations.transform_points(xyz, matrix)
    tree = cKDTree(mesh_pts)
    dist, idx = tree.query(scan_t, k=1)
    order = np.argsort(dist)
    pairs: list[dict] = []
    for i in order:
        if len(pairs) >= max_pairs:
            break
        if float(dist[i]) > 0.35:
            break
        sp = scan_t[i]
        mp = mesh_pts[idx[i]]
        if any(np.linalg.norm(sp - np.asarray(p["scan"])) < 1.0 for p in pairs):
            continue
        pairs.append({"scan": sp.tolist(), "model": mp.tolist()})
    if len(pairs) < 4:
        raise RuntimeError(f"ICP produced only {len(pairs)} spread pairs (need >=4)")
    return pairs


def save_alignment(base_url: str, upload_id: str, model_id: str, pairs: list[dict]) -> dict:
    url = f"{base_url.rstrip('/')}/api/uploads/{upload_id}/alignment"
    payload = json.dumps({"model_id": model_id, "alignment": {"pairs": pairs}}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def analyze_coverage(base_url: str, upload_id: str, model_id: str) -> dict:
    url = f"{base_url.rstrip('/')}/api/uploads/{upload_id}/coverage/analyze"
    payload = json.dumps(
        {
            "model_id": model_id,
            "coverage_options": {
                "distance_threshold_m": 0.15,
                "min_support_frames": 2,
                "require_visibility": False,
            },
        }
    ).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode("utf-8"))


def trigger_auto_align(base_url: str, upload_id: str, model_id: str) -> dict:
    url = f"{base_url.rstrip('/')}/api/uploads/{upload_id}/alignment/candidates"
    payload = json.dumps({"model_id": model_id, "dry_run": False}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="refs_site3d_pm1")
    ap.add_argument("--base-url", default="http://127.0.0.1:8767")
    ap.add_argument("--skip-align", action="store_true")
    ap.add_argument("--skip-coverage", action="store_true")
    args = ap.parse_args()

    upload_id = import_upload(args.model_id)
    print(f"coverage: {args.base_url}/coverage.html?model={args.model_id}&upload={upload_id}")
    if not args.skip_align:
        try:
            resp = trigger_auto_align(args.base_url, upload_id, args.model_id)
            print("auto-align:", json.dumps({k: resp.get(k) for k in ('ok', 'status', 'next_action')}, indent=2))
            if resp.get("status") == "needs_anchor" and not resp.get("candidates"):
                mesh = trimesh.load(MODELS / f"{args.model_id}.glb", force="mesh")
                if isinstance(mesh, trimesh.Scene):
                    mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
                scan = trimesh.load(DATA / "scene_001_pm1_nerf.ply", process=False)
                xyz = np.asarray(scan.vertices, dtype=np.float64)
                bmin = mesh.bounds[0]
                bmax = mesh.bounds[1]
                pad = max(8.0, float(np.linalg.norm(bmax - bmin)) * 0.5)
                keep = np.all(xyz >= bmin - pad, axis=1) & np.all(xyz <= bmax + pad, axis=1)
                xyz = xyz[keep]
                xyz, _ = subsample_points(xyz, np.zeros((len(xyz), 3), np.uint8), 80_000)
                pairs = icp_alignment_pairs(xyz, mesh)
                saved = save_alignment(args.base_url, upload_id, args.model_id, pairs)
                al = saved.get("alignment") or {}
                print(
                    "icp-align:",
                    json.dumps({"quality": al.get("quality"), "rmse_m": al.get("rmse_m")}, indent=2),
                )
        except Exception as exc:
            print("align skipped/failed:", exc)
    if not args.skip_coverage:
        try:
            cov = analyze_coverage(args.base_url, upload_id, args.model_id)
            objs = cov.get("objects") or []
            counts: dict[str, int] = {}
            for o in objs:
                st = o.get("status") or "?"
                counts[st] = counts.get(st, 0) + 1
            print("coverage:", json.dumps({"ok": cov.get("ok"), "objects": len(objs), "status_counts": counts}, indent=2))
        except Exception as exc:
            print("coverage skipped/failed:", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

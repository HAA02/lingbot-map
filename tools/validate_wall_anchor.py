#!/usr/bin/env python3
"""Validate metric-scale reproduction for coplay auto-placement (Phase 1b: wall-scale-x3).

Real data (upload_1781521406685: 144 poses / 35.3s) shows the CURRENT ceiling+camera
2-anchor fusion produces an implausible walking speed (~0.298 m/s vs a real ~0.9 m/s) —
the ~3x monocular scale under-estimate this project is fixing with a wall-width anchor
(scan2bim/wall_anchor.py, Phase 2). This script reproduces that red baseline so the fix
can be verified against a known-bad number, and gives an exit-3 stub for the (not yet
wired) wall-anchor default path.

Usage:
    .venv/bin/python tools/validate_wall_anchor.py <lbp2 path> [--no-wall-anchor] \
        [--duration <s>] [--model models/pipe_duct.glb]

--no-wall-anchor reproduces the CURRENT (pre-wall-anchor) 2-anchor fusion:
    gravity-align (build_coplay.viewer_pose + _rot_a_to_b, read-only reference) ->
    s_vert = model_bbox_height / scan_vertical_extent,
    s_cam  = scan2bim.metric_scale.camera_height_scale(...),
    s_m, info = scan2bim.metric_scale.fuse_scale_estimates([s_vert, s_cam])
This mirrors tools/build_coplay.py::place_pipe_auto()'s scale procedure (not its
pipe-run/waypoint routing, which needs an FXX dtdx file outside this phase's scope).

Default mode (wall anchor) is Phase 2 work and is not wired here; it always exits 3.
"""
from __future__ import annotations

import argparse
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

# Allow running as a script from the repo root without PYTHONPATH=.
_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scan2bim.metric_scale import (
    bbox_height_warning, camera_height_scale, estimate_floor_level,
    fuse_scale_estimates, speed_warning,
)

# Plausible continuous-walk speed band used for this script's PASS/FAIL verdict
# (separate from scan2bim.metric_scale.speed_warning's own internal thresholds,
# which are printed alongside as an additional diagnostic).
SPEED_BAND = (0.6, 1.4)


# ---------------------------------------------------------------------------
# lbp2 (LBP2/LBP3/LBP4) binary reader — mirrors realtime/server.py's
# _parse_scan_payload() layout (read-only reference; not imported directly since
# server.py has FastAPI app-level side effects at import time).
# ---------------------------------------------------------------------------

def parse_lbp2(path: Path) -> dict:
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError(f"payload too small: {path}")
    magic = data[:4]
    if magic not in (b"LBP2", b"LBP3", b"LBP4"):
        raise ValueError(f"unsupported payload magic {magic!r} in {path}")
    _magic, _flags, num_pts, num_poses, _seq = struct.unpack("<4sIIII", data[:20])
    off = 20
    xyz = np.frombuffer(data, dtype="<f4", count=num_pts * 3, offset=off).reshape(-1, 3)
    off += num_pts * 3 * 4
    off += num_pts * 3                       # rgb (uint8 x3) — unused here
    if magic == b"LBP3":
        off += num_pts * 3 * 4               # normals
    if magic == b"LBP4":
        off += num_pts * 3 * 4               # normals
        off += num_pts * 4                   # source ids
        off += num_pts * 4                   # component ids
    poses = np.frombuffer(data, dtype="<f4", count=num_poses * 12, offset=off).reshape(-1, 12)
    return {
        "magic": magic.decode("ascii"),
        "num_points": int(num_pts),
        "num_poses": int(num_poses),
        "points": xyz.astype(np.float64),
        "poses": poses.astype(np.float64),
        # LBP2/LBP3/LBP4 carry no per-pose timestamps in the current format.
        "timestamps": None,
    }


# ---------------------------------------------------------------------------
# duration resolution: --duration > lbp2 timestamps (none exist today) > sibling
# .mp4 via ffprobe, falling back to cv2 if ffprobe isn't on PATH.
# ---------------------------------------------------------------------------

def _ffprobe_duration(mp4_path: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(mp4_path)],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0 and out.stdout.strip():
            return float(out.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return None


def _cv2_duration(mp4_path: Path) -> float | None:
    try:
        import cv2
        cap = cv2.VideoCapture(str(mp4_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps and fps > 0 and frames and frames > 0:
            return float(frames / fps)
    except Exception:
        pass
    return None


def resolve_duration(lbp2_path: Path, parsed: dict, duration_arg: float | None) -> tuple[float | None, list[str]]:
    notes: list[str] = []
    if duration_arg is not None:
        notes.append("duration from --duration")
        return duration_arg, notes
    if parsed.get("timestamps"):
        ts = parsed["timestamps"]
        notes.append("duration from lbp2 timestamps")
        return float(ts[-1] - ts[0]), notes

    mp4_path = lbp2_path.with_suffix(".mp4")
    if mp4_path.exists():
        d = _ffprobe_duration(mp4_path)
        if d is not None:
            notes.append(f"duration from ffprobe: {mp4_path.name}")
            return d, notes
        d = _cv2_duration(mp4_path)
        if d is not None:
            notes.append(f"duration from cv2 (ffprobe unavailable on PATH): {mp4_path.name}")
            return d, notes
        notes.append(f"ffprobe/cv2 both failed to read {mp4_path.name}")

    notes.append("could not determine duration (no --duration, no lbp2 timestamps, "
                 f"no readable {mp4_path.name}) — pass --duration explicitly")
    return None, notes


# ---------------------------------------------------------------------------
# model bbox height (vertical/up-axis extent) — pipe_duct.glb and other
# glb_native models here are internally Y-up, metres, vertex-baked (identity
# node transforms; see CLAUDE.md coordinate conventions), matching the Y-up
# convention build_coplay.py uses for its DTDX-decoded bbox[1] index.
# ---------------------------------------------------------------------------

def model_bbox_height(model_path: Path) -> float:
    import trimesh
    mesh = trimesh.load(str(model_path), process=False)
    if hasattr(mesh, "geometry"):  # trimesh.Scene
        verts = [g.vertices for g in mesh.geometry.values() if len(g.vertices)]
        if not verts:
            raise ValueError(f"no geometry found in {model_path}")
        allv = np.concatenate(verts, axis=0).astype(np.float64)
    else:
        allv = np.asarray(mesh.vertices, dtype=np.float64)
    # drop extreme outliers only, same as tools/build_coplay.py's bbox computation
    med = np.median(allv, axis=0)
    keep = (np.abs(allv - med) < 60).all(axis=1)
    allc = allv[keep] if keep.any() else allv
    return float(allc[:, 1].max() - allc[:, 1].min())


# ---------------------------------------------------------------------------
# metric scale: reproduces tools/build_coplay.py::place_pipe_auto()'s procedure.
# ---------------------------------------------------------------------------

def _import_build_coplay():
    try:
        import tools.build_coplay as bc
    except Exception as e:  # pragma: no cover - environment/dependency issue
        raise RuntimeError(f"failed to import tools/build_coplay.py helpers: {e}") from e
    return bc


def compute_metric_scale(poses: np.ndarray, points: np.ndarray, bbox_height_m: float) -> dict:
    bc = _import_build_coplay()
    vp = [bc.viewer_pose(p) for p in poses]
    centers = np.array([v[0] for v in vp])
    ups = np.array([v[2] for v in vp])
    g = ups.mean(0)
    g /= (np.linalg.norm(g) + 1e-9)
    Rg = bc._rot_a_to_b(g, np.array([0.0, 1.0, 0.0]))
    Cg = centers @ Rg.T

    P = points.copy()
    P[:, 1] *= -1.0
    P[:, 2] *= -1.0
    Pg = P @ Rg.T

    vext = float(np.percentile(Pg[:, 1], 97) - np.percentile(Pg[:, 1], 3))
    s_vert = bbox_height_m / max(vext, 1e-6)
    floor_y = estimate_floor_level(Pg[:, 1], cam_y=float(np.median(Cg[:, 1])))
    s_cam = camera_height_scale(Cg[:, 1], floor_y) if floor_y is not None else None
    s_m, scale_info = fuse_scale_estimates([s_vert, s_cam])

    traj = Cg[:, [0, 2]]
    total_raw = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())
    return {
        "s_vert": s_vert, "s_cam": s_cam, "s_m": s_m, "scale_info": scale_info,
        "vext": vext, "floor_y": floor_y, "total_raw": total_raw,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lbp2", type=Path, help="path to a .lbp2/.lbp3/.lbp4 scan payload")
    ap.add_argument("--no-wall-anchor", action="store_true",
                     help="reproduce the current (pre-wall-anchor) 2-anchor fusion baseline")
    ap.add_argument("--duration", type=float, default=None, help="video duration in seconds")
    ap.add_argument("--model", type=Path, default=Path("models/pipe_duct.glb"))
    args = ap.parse_args()

    if not args.no_wall_anchor:
        # Phase 2 work (connecting scan2bim.wall_anchor as the default anchor) is not
        # part of this phase. Fail clearly rather than a silent placeholder success,
        # whether or not the module happens to exist/import yet.
        try:
            import scan2bim.wall_anchor  # noqa: F401
        except Exception as e:
            print(f"wall anchor not available yet (import failed: {e})")
            return 3
        print("wall anchor not available yet (Phase 2 integration pending — "
              "use --no-wall-anchor to reproduce the current baseline)")
        return 3

    if not args.lbp2.exists():
        print(f"error: lbp2 not found: {args.lbp2}", file=sys.stderr)
        return 2
    if not args.model.exists():
        print(f"error: model not found: {args.model}", file=sys.stderr)
        return 2

    parsed = parse_lbp2(args.lbp2)
    print(f"lbp2: magic={parsed['magic']} points={parsed['num_points']:,} poses={parsed['num_poses']}")

    duration, dur_notes = resolve_duration(args.lbp2, parsed, args.duration)
    for note in dur_notes:
        print(f"  {note}")
    if duration is None:
        print("error: duration could not be determined — pass --duration explicitly", file=sys.stderr)
        return 2

    bbox_h = model_bbox_height(args.model)
    bbox_warn = bbox_height_warning(bbox_h)

    scale = compute_metric_scale(parsed["poses"], parsed["points"], bbox_h)
    path_m = scale["total_raw"] * scale["s_m"]
    speed_ms = path_m / duration
    lo, hi = SPEED_BAND
    verdict = "PASS" if lo <= speed_ms <= hi else "FAIL"

    print(f"scale={scale['s_m']:.4f} path_m={path_m:.2f} duration_s={duration:.2f} "
          f"speed_ms={speed_ms:.3f} band=[{lo},{hi}] verdict={verdict}")
    s_cam_str = f"{scale['s_cam']:.4f}" if scale["s_cam"] is not None else "n/a"
    print(f"  model_bbox_height_m={bbox_h:.3f} scan_vertical_extent={scale['vext']:.3f}")
    print(f"  s_vert={scale['s_vert']:.4f} s_cam={s_cam_str} "
          f"agree={scale['scale_info']['agree']} spread={scale['scale_info']['spread']}")
    if bbox_warn:
        print(f"  WARNING (bbox): {bbox_warn}")
    sw = speed_warning(path_m, duration)
    if sw:
        print(f"  WARNING (speed): {sw}")
    if verdict == "FAIL":
        print(f"  NOTE: speed {speed_ms:.3f} m/s is outside the plausible band {SPEED_BAND} — "
              "this reproduces the KNOWN red baseline (monocular scale under-estimate), "
              "not a script bug. exit 0: reproduction succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

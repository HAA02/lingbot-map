#!/usr/bin/env python3
"""Validate metric-scale reproduction for coplay auto-placement (wall-scale-x3).

Real data (upload_1781521406685: 144 poses / 35.3s) shows the ceiling+camera
2-anchor fusion alone produces an implausible walking speed (~0.3 m/s vs a real
~0.9 m/s) — the ~3x monocular scale under-estimate this project is fixing with a
wall-width anchor (scan2bim/wall_anchor.py). This script reproduces that red
baseline and validates the 3-anchor fusion (tools/build_coplay.py) against it.

Usage:
    .venv/bin/python tools/validate_wall_anchor.py <lbp2 path> [--no-wall-anchor] \
        [--duration <s>] [--model models/pipe_duct.glb] [--dtdx <path/glob>...]

Default mode (AXIS-SPLIT, requires --dtdx <multi-discipline model files/glob>):
    gravity-align (build_coplay.viewer_pose + _rot_a_to_b, read-only reference) ->
    s_v (vertical) = fuse_scale_estimates([s_vert, s_cam]) — ceiling-height +
        camera-height, unchanged 2-anchor logic,
    s_h (horizontal) = tools.build_coplay.compute_wall_anchor(...) — SXX structure
        wall-pair gap (tools.build_coplay.model_corridor_widths, is_triangle_soup=
        True, vertical-span>1.5m + [1.5,6]m width clamp — AXX/architecture was
        tried first but is furniture-dominated, see model_corridor_widths()
        docstring) vs. the recon's own straddle-forced corridor-width anchor
        (scan2bim.wall_anchor.estimate_wall_scale, trajectory_radius=vext). None
        -> s_h falls back to s_v (fallback_reason explains why).
    diag(s_h, s_v, s_h) applied via scan2bim.metric_scale.apply_axis_split_scale
    (monocular recon compresses the two horizontal axes more than the vertical
    one — a single isotropic scale places the path wrong even when "on average"
    close; see tools/build_coplay.py::place_pipe_auto()'s docstring for the
    real-data numbers). Speed is judged on the s_h-scaled path.
Exit codes (default mode):
    3   the wall anchor can't even be attempted (import failure, or no SXX/
        structure discipline in --dtdx).
    2   s_v or s_h itself falls outside its expected physical-plausibility band
        (S_V_BAND/S_H_BAND) — a more specific diagnosis than a bad speed alone;
        checked before the speed verdict.
    1   speed falls outside AXIS_SPEED_BAND.
    0   speed is plausible and both anchors are in-band.
    (diagnostics are always printed regardless of exit code — never silently
    adjusted to fit any band.)

--no-wall-anchor reproduces the PRIOR (pre-wall-anchor) 2-anchor fusion, UNCHANGED:
    wall_points is always None here, so s_h falls back to s_v — mathematically
    identical to the pre-axis-split isotropic 2-anchor value (apply_axis_split_
    scale(s_h=s_v,s_v=s_v) degenerates to a uniform scale). If --dtdx is given,
    s_vert uses the dtdx interior bbox height (real red-baseline reproduction);
    otherwise falls back to --model (default models/pipe_duct.glb, Phase-1
    behavior). Same SPEED_BAND/summary format as Phase 1, always exits 0
    (baseline reproduction "succeeding" is independent of whether the resulting
    speed is plausible).
"""
from __future__ import annotations

import argparse
import glob
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
    apply_axis_split_scale, bbox_height_warning, camera_height_scale, estimate_floor_level,
    fuse_scale_estimates, speed_warning,
)

# --no-wall-anchor: plausible continuous-walk speed band for the PASS/FAIL verdict
# (separate from scan2bim.metric_scale.speed_warning's own internal thresholds,
# which are printed alongside as an additional diagnostic). Unchanged from Phase 1.
SPEED_BAND = (0.6, 1.4)

# Default (axis-split) mode: speed is judged on the s_h-scaled path (wider band —
# a wall-anchor-driven horizontal scale is expected to land closer to the true
# speed than the old omni-fused one). s_v/s_h sanity bands catch an anchor value
# that is itself implausible (e.g. the wrong SXX width candidate matched) even
# before it shows up as a bad speed — see main()'s exit-2 tagging.
AXIS_SPEED_BAND = (0.4, 1.4)
S_V_BAND = (1.6, 1.9)
S_H_BAND = (3.2, 4.1)


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
# model bbox height (vertical/up-axis extent) — GLB path (Phase-1 fallback when
# --dtdx is not given). glb_native models here are internally Y-up, metres,
# vertex-baked (identity node transforms; see CLAUDE.md coordinate conventions),
# matching the Y-up convention build_coplay.py uses for its DTDX-decoded bbox[1].
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
    med = np.median(allv, axis=0)
    keep = (np.abs(allv - med) < 60).all(axis=1)
    allc = allv[keep] if keep.any() else allv
    return float(allc[:, 1].max() - allc[:, 1].min())


# ---------------------------------------------------------------------------
# dtdx model loading — mirrors tools/build_coplay.py::main()'s geometry loading
# and interior (non-SXX union) bbox-height computation exactly.
# ---------------------------------------------------------------------------

def _expand_dtdx(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if matches:
            paths.extend(Path(m) for m in matches)
        elif Path(pat).exists():
            paths.append(Path(pat))
    return paths


def load_dtdx_vertices_by_code(dtdx_paths: list[Path]) -> dict[str, np.ndarray]:
    from scan2bim.dtdx_geometry import decode_geometry
    out: dict[str, list[np.ndarray]] = {}
    for p in dtdx_paths:
        g = decode_geometry(str(p))
        code = g["discipline_code"]
        for m in g["meshes"]:
            pos = np.asarray(m["positions"], dtype=np.float64)
            if len(pos):
                out.setdefault(code, []).append(pos)
    return {code: np.concatenate(v, axis=0) for code, v in out.items()}


def dtdx_bbox_height(by_code: dict[str, np.ndarray]) -> float:
    """Interior (non-SXX) union bbox height, exactly mirroring
    tools/build_coplay.py::main()'s `interior`/`bbox` computation."""
    interior = [v for c, v in by_code.items() if c != "SXX"] or list(by_code.values())
    if not interior:
        raise ValueError("no dtdx geometry loaded")
    allp = np.concatenate(interior, axis=0)
    med = np.median(allp, axis=0)
    keep = (np.abs(allp - med) < 60).all(axis=1)
    allc = allp[keep] if keep.any() else allp
    return float(allc[:, 1].max() - allc[:, 1].min())


# ---------------------------------------------------------------------------
# metric scale: reproduces tools/build_coplay.py::place_pipe_auto()'s procedure,
# optionally extended with the corridor-width (wall) anchor.
# ---------------------------------------------------------------------------

def _import_build_coplay():
    try:
        import tools.build_coplay as bc
    except Exception as e:  # pragma: no cover - environment/dependency issue
        raise RuntimeError(f"failed to import tools/build_coplay.py helpers: {e}") from e
    return bc


def compute_metric_scale(poses: np.ndarray, points: np.ndarray, bbox_height_m: float,
                         wall_points: np.ndarray | None = None,
                         corridor_width_hint: float | None = None) -> dict:
    """AXIS-SPLIT metric scale (mirrors tools/build_coplay.py::place_pipe_auto()):
    s_v = fuse_scale_estimates([s_vert, s_cam]) (vertical: ceiling-height + camera-
    height, unchanged 2-anchor logic); s_h = the wall (corridor-width) anchor if
    available, else falls back to s_v (fallback_reason explains why). diag(s_h,
    s_v,s_h) is applied via scan2bim.metric_scale.apply_axis_split_scale, so the
    returned path_m is already metric (no separate scalar multiply needed).
    When wall_points=None, s_h==s_v always (isotropic), so path_m is
    mathematically identical to the pre-axis-split 2-anchor value — the
    --no-wall-anchor reproduction path is unaffected.
    corridor_width_hint: see tools.build_coplay.compute_wall_anchor() — bypasses
    automatic corridor-width detection with a single user-verified value."""
    bc = _import_build_coplay()
    centers, fwd, up = [], [], []
    for p in poses:
        c, f, u = bc.viewer_pose(p)
        centers.append(c); fwd.append(f); up.append(u)
    centers = np.array(centers); fwd = np.array(fwd); up = np.array(up)
    g = up.mean(0)
    g /= (np.linalg.norm(g) + 1e-9)
    Rg = bc._rot_a_to_b(g, np.array([0.0, 1.0, 0.0]))
    Cg = centers @ Rg.T; Fg = fwd @ Rg.T; Ug = up @ Rg.T

    P = points.copy()
    P[:, 1] *= -1.0
    P[:, 2] *= -1.0
    Pg = P @ Rg.T

    vext = float(np.percentile(Pg[:, 1], 97) - np.percentile(Pg[:, 1], 3))
    s_vert = bbox_height_m / max(vext, 1e-6)
    floor_y = estimate_floor_level(Pg[:, 1], cam_y=float(np.median(Cg[:, 1])))
    s_cam = camera_height_scale(Cg[:, 1], floor_y) if floor_y is not None else None
    s_v, s_v_info = fuse_scale_estimates([s_vert, s_cam])

    s_wall, wall_info = bc.compute_wall_anchor(Pg, Cg[:, [0, 2]], wall_points, vext,
                                               corridor_width_hint=corridor_width_hint)
    fallback_reason = None
    if s_wall is not None:
        s_h = s_wall
    else:
        s_h = s_v
        fallback_reason = wall_info.get("fail", "wall anchor unavailable")

    poses_s, _Pg_s = apply_axis_split_scale({"c": Cg, "f": Fg, "u": Ug}, Pg, s_h=s_h, s_v=s_v)
    traj = poses_s["c"][:, [0, 2]]                                    # already metric
    path_m = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())

    scale_info = {"s_v": round(s_v, 4), "s_h": round(s_h, 4), "s_v_anchors": s_v_info,
                  "wall_anchor": wall_info, "fallback_reason": fallback_reason}
    return {
        "s_vert": s_vert, "s_cam": s_cam, "s_wall": s_wall,
        "s_v": s_v, "s_v_info": s_v_info, "s_h": s_h, "fallback_reason": fallback_reason,
        "scale_info": scale_info, "vext": vext, "floor_y": floor_y, "path_m": path_m,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lbp2", type=Path, help="path to a .lbp2/.lbp3/.lbp4 scan payload")
    ap.add_argument("--no-wall-anchor", action="store_true",
                     help="reproduce the prior (pre-wall-anchor) 2-anchor fusion baseline")
    ap.add_argument("--duration", type=float, default=None, help="video duration in seconds")
    ap.add_argument("--model", type=Path, default=Path("models/pipe_duct.glb"),
                     help="GLB model bbox for --no-wall-anchor when --dtdx is not given")
    ap.add_argument("--dtdx", nargs="+", default=None,
                     help="dtdx path(s) or glob pattern(s), e.g. 'models/Gasan_7F/*.dtdx' "
                          "(multi-discipline: interior bbox height + SXX corridor widths)")
    ap.add_argument("--corridor-width-hint", type=float, default=None,
                     help="single user-verified corridor width (m) — bypasses automatic SXX "
                          "candidate detection entirely (len==1 candidate list, no ambiguity). "
                          "Not auto-detected; a scalar hint the caller has separately verified.")
    args = ap.parse_args()

    if not args.lbp2.exists():
        print(f"error: lbp2 not found: {args.lbp2}", file=sys.stderr)
        return 2

    parsed = parse_lbp2(args.lbp2)
    print(f"lbp2: magic={parsed['magic']} points={parsed['num_points']:,} poses={parsed['num_poses']}")

    duration, dur_notes = resolve_duration(args.lbp2, parsed, args.duration)
    for note in dur_notes:
        print(f"  {note}")
    if duration is None:
        print("error: duration could not be determined — pass --duration explicitly", file=sys.stderr)
        return 2

    dtdx_paths = _expand_dtdx(args.dtdx) if args.dtdx else []
    if args.dtdx and not dtdx_paths:
        print(f"error: no dtdx files matched: {args.dtdx}", file=sys.stderr)
        return 2
    by_code = load_dtdx_vertices_by_code(dtdx_paths) if dtdx_paths else {}
    if dtdx_paths:
        print(f"  dtdx: {len(dtdx_paths)} file(s), disciplines={sorted(by_code)}")

    if not args.no_wall_anchor:
        # Default mode needs a real multi-discipline model: the interior bbox
        # height for s_vert, and SXX structure geometry for the wall anchor.
        # (AXX/architecture was tried first but is furniture-dominated on real
        # data — zero triangles with vertical span > 1.5m; SXX carries the real
        # walls/glass partitions. See model_corridor_widths()'s docstring.)
        if not dtdx_paths:
            print("error: default (wall-anchor) mode needs --dtdx <multi-discipline model "
                  "files/glob> — corridor widths come from the SXX structure geometry",
                  file=sys.stderr)
            return 2
        try:
            import scan2bim.wall_anchor  # noqa: F401
        except Exception as e:
            print(f"wall anchor not available (import failed: {e})")
            return 3
        bbox_h = dtdx_bbox_height(by_code)
        wall_points = by_code.get("SXX")
        if wall_points is None and args.corridor_width_hint is None:
            print("wall anchor not available: no SXX (structure) discipline in --dtdx "
                  "— corridor widths need real walls, not MEP-only geometry")
            return 3
    else:
        if dtdx_paths:
            bbox_h = dtdx_bbox_height(by_code)
        else:
            if not args.model.exists():
                print(f"error: model not found: {args.model}", file=sys.stderr)
                return 2
            bbox_h = model_bbox_height(args.model)
        wall_points = None  # --no-wall-anchor: reproduce the 2-anchor baseline only

    if args.corridor_width_hint is not None:
        print(f"  corridor_width_hint: {args.corridor_width_hint} (manual, verified) "
              "— bypassing automatic SXX corridor-width detection")

    bbox_warn = bbox_height_warning(bbox_h)
    scale = compute_metric_scale(parsed["poses"], parsed["points"], bbox_h, wall_points=wall_points,
                                 corridor_width_hint=args.corridor_width_hint)
    path_m = scale["path_m"]                      # already axis-split metric
    speed_ms = path_m / duration
    s_cam_str = f"{scale['s_cam']:.4f}" if scale["s_cam"] is not None else "n/a"
    s_wall_str = f"{scale['s_wall']:.4f}" if scale["s_wall"] is not None else "n/a"

    if args.no_wall_anchor:
        # --no-wall-anchor: wall_points is always None here, so s_h==s_v always
        # (isotropic) — mathematically identical to the pre-axis-split 2-anchor
        # value. Same summary line / band / exit-0-always contract as Phase 1.
        lo, hi = SPEED_BAND
        verdict = "PASS" if lo <= speed_ms <= hi else "FAIL"
        print(f"scale={scale['s_v']:.4f} path_m={path_m:.2f} duration_s={duration:.2f} "
              f"speed_ms={speed_ms:.3f} band=[{lo},{hi}] verdict={verdict}")
        print(f"  model_bbox_height_m={bbox_h:.3f} scan_vertical_extent={scale['vext']:.3f}")
        print(f"  s_vert={scale['s_vert']:.4f} s_cam={s_cam_str} s_wall={s_wall_str} "
              f"n_anchors={scale['s_v_info']['n']} agree={scale['s_v_info']['agree']} "
              f"spread={scale['s_v_info']['spread']}")
        wa = scale["scale_info"].get("wall_anchor") or {}
        if wa:
            print(f"  wall_anchor: {wa}")
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

    # Default (wall-anchor) mode: speed judged on the s_h-scaled path.
    lo, hi = AXIS_SPEED_BAND
    verdict = "PASS" if lo <= speed_ms <= hi else "FAIL"
    s_v_ok = S_V_BAND[0] <= scale["s_v"] <= S_V_BAND[1]
    s_h_ok = S_H_BAND[0] <= scale["s_h"] <= S_H_BAND[1]

    print(f"scale={scale['s_h']:.4f} path_m={path_m:.2f} duration_s={duration:.2f} "
          f"speed_ms={speed_ms:.3f} band=[{lo},{hi}] verdict={verdict}")
    print(f"  model_bbox_height_m={bbox_h:.3f} scan_vertical_extent={scale['vext']:.3f}")
    print(f"  s_vert={scale['s_vert']:.4f} s_cam={s_cam_str} s_wall={s_wall_str}")
    print(f"  s_v={scale['s_v']:.4f} (band={S_V_BAND} ok={s_v_ok}) "
          f"s_h={scale['s_h']:.4f} (band={S_H_BAND} ok={s_h_ok}) "
          f"fallback_reason={scale['fallback_reason']!r}")
    wa = scale["scale_info"].get("wall_anchor") or {}
    if wa:
        print(f"  wall_anchor: {wa}")
    if bbox_warn:
        print(f"  WARNING (bbox): {bbox_warn}")
    sw = speed_warning(path_m, duration)
    if sw:
        print(f"  WARNING (speed): {sw}")

    # Anchor-value sanity tagging takes precedence over the speed verdict: if the
    # anchor itself looks implausible, that's the more specific diagnosis. Never
    # adjusted to force a pass — reported exactly as measured either way.
    if not (s_v_ok and s_h_ok):
        print(f"  NOTE: anchor value outside its expected physical-plausibility band "
              f"(s_v ok={s_v_ok}, s_h ok={s_h_ok}) — reported as measured, not adjusted. exit 2.")
        return 2
    if verdict == "FAIL":
        print(f"  NOTE: speed {speed_ms:.3f} m/s is outside the plausible band {AXIS_SPEED_BAND} — "
              "reported as measured, not adjusted.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

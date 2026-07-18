#!/usr/bin/env python3
"""Geometric sanity checker for a built coplay HTML (RAWP embedded pose array).

Parses the embedded RAWP (placed camera poses, model-space) + META.duration from
a tools/build_coplay.py-generated HTML, and checks two "does this look like a
real building" gates, independent of any particular scale-anchor fusion method:
  (a) turn point: the trajectory's main corner (scan2bim.pipe_path.
      trajectory_turn_fraction — the SAME corner-finder tools/build_coplay.py::
      place_pipe_auto() already uses to route the corridor's L-turn) must sit at
      Z <= --turn-z-max. A corner placed deep inside an unrelated room/wing is a
      placement bug, not a scale artifact.
  (b) endpoint: the walk's last position must land at X in --end-x (lo,hi) and
      Z <= --end-z-max — a plausible open/walkable zone, not embedded in a wall.
Also recomputes path length / average speed (XZ-only) from RAWP as a
cross-check, purely informational (no pass/fail threshold here — that's
tools/validate_wall_anchor.py's job; this script checks WHERE the path bends
and ends, not how fast it appears to move).

Usage:
    .venv/bin/python tools/check_coplay_geometry.py <coplay.html> \
        [--turn-z-max 4] [--end-x -8,4] [--end-z-max 2.5]

Exit 0 if both (a) and (b) pass, exit 1 otherwise (diagnostics always printed).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# Allow running as a script from the repo root without PYTHONPATH=.
_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scan2bim.pipe_path import trajectory_turn_fraction


def parse_coplay_html(path: Path) -> dict:
    """Extract RAWP (placed pose list) + META (duration) from a coplay HTML.
    The build_coplay.py TEMPLATE embeds these as a single JS statement:
    `const MESHES=..., MODELS=..., RAWP=[...], META={...};` — RAWP is a JSON
    array of {"c":[x,y,z],"f":[...],"u":[...]} (model-space, metres)."""
    html = path.read_text(encoding="utf-8")
    m = re.search(r"RAWP=(\[.*\]),\s*META=(\{.*?\});", html, re.DOTALL)
    if not m:
        raise ValueError(f"could not find RAWP/META in {path} (not a build_coplay.py output?)")
    rawp = json.loads(m.group(1))
    meta = json.loads(m.group(2))
    if not rawp:
        raise ValueError(f"RAWP is empty in {path}")
    return {"poses": rawp, "duration": meta.get("duration")}


def turn_point(xz: np.ndarray, smooth_window: int = 1) -> dict:
    """Main corner via scan2bim.pipe_path.trajectory_turn_fraction — the same
    corner-finder tools/build_coplay.py::place_pipe_auto() already uses to route
    the corridor L-turn (best two-straight-segment line-fit split over a
    resampled curve, robust to end-hooks/jitter by design — see its docstring).

    smooth_window>1 pre-smooths XZ with a moving average before the split
    search. Empirically (upload_1781521406685's served HTML) this BLURS the
    real ~90deg corridor corner (found at window=1) into a much weaker ~15deg
    bend at a different location — trajectory_turn_fraction's own resampling
    already provides adequate noise robustness, so the default is off (1)."""
    sm = xz
    if smooth_window > 1 and len(xz) > smooth_window:
        ker = np.ones(smooth_window) / smooth_window
        sm = np.column_stack([
            np.convolve(xz[:, 0], ker, mode="same"),
            np.convolve(xz[:, 1], ker, mode="same"),
        ])
    tf, angle_deg = trajectory_turn_fraction(sm)
    arclen = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(sm, axis=0), axis=1))])
    target = tf * arclen[-1]
    return {
        "x": float(np.interp(target, arclen, sm[:, 0])),
        "z": float(np.interp(target, arclen, sm[:, 1])),
        "fraction": float(tf), "angle_deg": float(angle_deg),
    }


def _parse_range(spec: str) -> tuple[float, float]:
    lo_s, hi_s = spec.split(",")
    lo, hi = float(lo_s), float(hi_s)
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("html", type=Path, help="path to a tools/build_coplay.py-generated HTML")
    ap.add_argument("--turn-z-max", type=float, default=4.0)
    ap.add_argument("--turn-z-min", type=float, default=None,
                     help="counterpart of --turn-z-max (lower bound) — the turn point Z must be >= this. "
                          "Default off: not checked, so existing invocations behave exactly as before.")
    ap.add_argument("--turn-fraction-range", type=str, default=None,
                     help="lo,hi — the detected corner's trajectory_turn_fraction must land in this "
                          "range (e.g. near the physical-turn-time fraction). Default off: not checked, "
                          "so existing invocations behave exactly as before.")
    ap.add_argument("--end-x", type=str, default="-8,4", help="lo,hi (model X range)")
    ap.add_argument("--end-z-max", type=float, default=2.5)
    ap.add_argument("--pre-turn-x-range", type=str, default=None,
                     help="lo,hi — every pose up to and including the turn must have X in this band "
                          "(the straight leg stays inside the corridor). Default off: not checked, so "
                          "existing invocations behave exactly as before.")
    ap.add_argument("--smooth-window", type=int, default=1,
                     help="moving-average window before turn detection (default 1 = off; see turn_point() docstring)")
    ap.add_argument("--duration", type=float, default=None, help="override the HTML's META.duration")
    ap.add_argument("--path-min", type=float, default=None,
                     help="minimum total path length (path_m) required to pass. Default off: not "
                          "checked, so existing invocations behave exactly as before.")
    ap.add_argument("--post-turn-heading-span-min", type=float, default=None,
                     help="minimum total-variation of heading (degrees, 360-wrap-safe) among poses AFTER "
                          "the detected turn point (tp['fraction']) required to pass — guards against a "
                          "naturally winding post-turn segment (e.g. a lounge look-around) being forced "
                          "straight. Default off: not checked, so existing invocations behave exactly "
                          "as before.")
    args = ap.parse_args()

    if not args.html.exists():
        print(f"error: html not found: {args.html}", file=sys.stderr)
        return 2

    data = parse_coplay_html(args.html)
    poses = data["poses"]
    duration = args.duration if args.duration is not None else data["duration"]
    xz = np.array([[p["c"][0], p["c"][2]] for p in poses], dtype=np.float64)
    print(f"coplay html: {args.html.name} poses={len(poses)} duration_s={duration}")

    tp = turn_point(xz, smooth_window=args.smooth_window)
    end_x, end_z = float(xz[-1, 0]), float(xz[-1, 1])
    path_m = float(np.linalg.norm(np.diff(xz, axis=0), axis=1).sum())
    speed_ms = path_m / duration if duration else None

    lo_x, hi_x = _parse_range(args.end_x)
    turn_ok = tp["z"] <= args.turn_z_max
    end_ok = (lo_x <= end_x <= hi_x) and (end_z <= args.end_z_max)

    # turn Z lower bound (optional): pairs with --turn-z-max to band the turn point's Z.
    turn_z_min_ok = True
    turn_z_min_line = None
    if args.turn_z_min is not None:
        turn_z_min_ok = tp["z"] >= args.turn_z_min
        turn_z_min_line = (f"turn_z_min z={tp['z']:.2f} turn_z_min={args.turn_z_min} -> "
                            f"{'OK' if turn_z_min_ok else 'FAIL'}")

    # turn fraction band (optional): the corner's arc-length fraction must land near
    # where the physical turn is expected (e.g. from --turn-time-s in place_rigid).
    frac_ok = True
    frac_line = None
    if args.turn_fraction_range is not None:
        flo, fhi = _parse_range(args.turn_fraction_range)
        frac_ok = flo <= tp["fraction"] <= fhi
        frac_line = (f"turn_fraction={tp['fraction']:.2f} range=[{flo},{fhi}] -> "
                      f"{'OK' if frac_ok else 'FAIL'}")

    # pre-turn straight-leg X band (optional): every pose up to the turn must stay in
    # the corridor. Catches the diagonal-drift failure the turn/end gates alone miss —
    # a leg that drifts sideways can still end in a valid spot (cycle-1 defect).
    pre_ok = True
    pre_line = None
    if args.pre_turn_x_range is not None:
        plo, phi = _parse_range(args.pre_turn_x_range)
        arclen = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xz, axis=0), axis=1))])
        turn_arc = tp["fraction"] * arclen[-1]
        pre_x = xz[arclen <= turn_arc + 1e-9, 0]
        pmin, pmax = float(pre_x.min()), float(pre_x.max())
        pre_ok = (pmin >= plo) and (pmax <= phi)
        pre_line = (f"pre_turn_x min={pmin:.2f} max={pmax:.2f} drift={pmax - pmin:.2f} "
                    f"band=[{plo},{phi}] (n={len(pre_x)}) -> {'OK' if pre_ok else 'FAIL'}")

    # post-turn heading total-variation (optional): poses AFTER the turn point must still
    # show enough heading change (360-wrap-safe) — catches a real look-around (e.g. a lounge
    # entry) being force-straightened by a placement bug, distinct from a legitimate turn.
    post_turn_ok = True
    post_turn_line = None
    if args.post_turn_heading_span_min is not None:
        arclen_pt = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xz, axis=0), axis=1))])
        turn_arc = tp["fraction"] * arclen_pt[-1]
        post_xz = xz[arclen_pt >= turn_arc - 1e-9]
        steps = np.diff(post_xz, axis=0)
        step_len = np.linalg.norm(steps, axis=1)
        valid = step_len >= 0.01  # skip near-zero steps: heading is noise, not signal
        headings = np.degrees(np.arctan2(steps[valid, 1], steps[valid, 0]))
        if len(headings) >= 2:
            wrapped = (np.diff(headings) + 180.0) % 360.0 - 180.0
            tv_deg = float(np.abs(wrapped).sum())
        else:
            tv_deg = 0.0
        post_turn_ok = tv_deg >= args.post_turn_heading_span_min
        post_turn_line = (f"post_turn_heading_span={tv_deg:.1f} deg min={args.post_turn_heading_span_min} -> "
                           f"{'OK' if post_turn_ok else 'FAIL'}")

    # total path length lower bound (optional): the walk's full path_m (already computed
    # below as an informational cross-check) must reach at least this length.
    path_min_ok = True
    path_min_line = None
    if args.path_min is not None:
        path_min_ok = path_m >= args.path_min
        path_min_line = f"path_m={path_m:.2f} path_min={args.path_min} -> {'OK' if path_min_ok else 'FAIL'}"

    verdict = "PASS" if (turn_ok and turn_z_min_ok and end_ok and pre_ok and frac_ok
                          and post_turn_ok and path_min_ok) else "FAIL"

    print(f"turn_point x={tp['x']:.2f} z={tp['z']:.2f} (fraction={tp['fraction']:.2f} "
          f"angle={tp['angle_deg']:.1f}deg) turn_z_max={args.turn_z_max} -> {'OK' if turn_ok else 'FAIL'}")
    if turn_z_min_line is not None:
        print(turn_z_min_line)
    if frac_line is not None:
        print(frac_line)
    print(f"endpoint x={end_x:.2f} z={end_z:.2f} end_x_range=[{lo_x},{hi_x}] "
          f"end_z_max={args.end_z_max} -> {'OK' if end_ok else 'FAIL'}")
    if pre_line is not None:
        print(pre_line)
    if post_turn_line is not None:
        print(post_turn_line)
    speed_str = f"{speed_ms:.3f}" if speed_ms is not None else "n/a (no duration)"
    print(f"path_m={path_m:.2f} duration_s={duration} avg_speed_ms={speed_str} (informational, no threshold here)")
    if path_min_line is not None:
        print(path_min_line)
    print(f"verdict={verdict}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

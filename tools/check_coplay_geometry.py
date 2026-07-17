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
    ap.add_argument("--end-x", type=str, default="-8,4", help="lo,hi (model X range)")
    ap.add_argument("--end-z-max", type=float, default=2.5)
    ap.add_argument("--pre-turn-x-range", type=str, default=None,
                     help="lo,hi — every pose up to and including the turn must have X in this band "
                          "(the straight leg stays inside the corridor). Default off: not checked, so "
                          "existing invocations behave exactly as before.")
    ap.add_argument("--smooth-window", type=int, default=1,
                     help="moving-average window before turn detection (default 1 = off; see turn_point() docstring)")
    ap.add_argument("--duration", type=float, default=None, help="override the HTML's META.duration")
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

    verdict = "PASS" if (turn_ok and end_ok and pre_ok) else "FAIL"

    print(f"turn_point x={tp['x']:.2f} z={tp['z']:.2f} (fraction={tp['fraction']:.2f} "
          f"angle={tp['angle_deg']:.1f}deg) turn_z_max={args.turn_z_max} -> {'OK' if turn_ok else 'FAIL'}")
    print(f"endpoint x={end_x:.2f} z={end_z:.2f} end_x_range=[{lo_x},{hi_x}] "
          f"end_z_max={args.end_z_max} -> {'OK' if end_ok else 'FAIL'}")
    if pre_line is not None:
        print(pre_line)
    speed_str = f"{speed_ms:.3f}" if speed_ms is not None else "n/a (no duration)"
    print(f"path_m={path_m:.2f} duration_s={duration} avg_speed_ms={speed_str} (informational, no threshold here)")
    print(f"verdict={verdict}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""D2 robustness gate for coplay-planmatch-01: seeded plan-perturbation generator
+ success-rate / outlier-recall / ambiguous-HOLD verdict, wired against the real
matcher (`scan2bim/coarse_match.py`, dev-core P1).

OWNERSHIP: this file, and the perturbation parameters / judgement thresholds it
defines, are QA-OWNED (coplay-planmatch-01-qa). Dev requests to relax them go to
PM, not into an edit here.

===========================================================================================
CYCLE 7 -- TWO DEFINITION CORRECTIONS (thresholds 90%/80% UNCHANGED; what is measured
changed). Both are documented in full where they are implemented (Section 1b /
Section 3); this is the short version for anyone diffing the module docstring.

(A) HARNESS DENSITY. Measured (main session, before this cycle): the D2 harness fed the
    matcher a plan where ONE WALL == ONE SEGMENT (`tests/test_coarse_match._corridor_walls`
    emits exactly 10 segments for the 4-leg STAIR fixture). Removing ONE segment therefore
    removed an ENTIRE wall face and collapsed a whole leg: legs 4 -> 3 on one removal,
    -> 2 on two, -> 1 on three (reproduced and confirmed again this cycle, see
    `assert_single_fragment_removal_is_graceful`'s pre-fix numbers in the cycle report).
    That is a HARNESS artefact (no real DXF wall is one uncut polyline edge from corner to
    corner -- `dxf_plan.load_wall_segments` turns every LWPOLYLINE VERTEX PAIR into its own
    segment already), not a measurement of matcher robustness. Section 1b
    (`densify_wall_segments`) now splits every wall into many collinear, TOUCHING
    (zero-gap) fragments BEFORE perturbing, so "remove 10-30%" removes PARTS of walls, not
    whole walls. The fragment size is NOT the team-doc's illustrative "3-8 pieces" -- that
    was MEASURED to reproduce the exact same collapse (a piece bigger than dev-core's own
    `wall_join` bridging tolerance still severs the wall line on a single hit) -- see
    Section 1b's docstring for the measured numbers behind the size actually used.

(B) RECALL DENOMINATOR. Measured (main session): outlier-recall case#17's "missed" item
    was a noise wall 21.8 m from the walk -- undetectable by a walk-vs-plan mismatch
    mechanism BY CONSTRUCTION, no matter how good the matcher gets. Folding it into the
    denominator makes the 80% target structurally unreachable. Section 3's
    `classify_changed_segments` now splits every changed (shifted/noise) segment into
    observable / not_observable (further than `OUTLIER_RECALL_RADIUS_M` from the TRUE walk
    path -- a matcher can only report a mismatch near where it actually walked) and, among
    the observable ones, scoreable / not_scoreable (a SHIFT whose LATERAL component is
    already inside the matcher's OWN reported association tolerance, `info.tol.leg_lat`,
    is physically indistinguishable from noise even to a perfect matcher -- case#8's
    inlier_ratio==1.0 with a 0.3-1.0 m shift band that can dip under a ~0.55 m tolerance).
    recall is now scored ONLY over the scoreable-and-observable set; the excluded counts
    are ALWAYS printed, never silently dropped from the denominator -- see Section 3.
===========================================================================================

===========================================================================================
CYCLE 11 -- PERTURBATION MODEL SPLIT: STRUCTURED (new default) vs FRAGMENT (Cycle-7's
original model, kept, not deleted, selectable). Measured (main session, cycle 11 order):
Cycle 7's harness-density fix (item (A) above) fed the matcher REMOVE/SHIFT draws that
scatter across ALL ~232 post-densify fragments independently -- 10-30% of 232 = 23-70
fragments, picked without any notion of which of the STAIR fixture's 10 original walls each
one came from. A 30% removal at that granularity puts a hole in EVERY wall of the fixture at
once. That is a model of drawing/scan INCOMPLETENESS (missing or noisy fragments scattered
everywhere), not the team's actual DoD target -- "현장이 변경됐는데 BIM에 반영 안 된 상태"
(the site changed but the drawing was never updated). A real as-built change is a WHOLE
torn-down partition, a WHOLE relocated wall, or a WHOLE new partition -- structurally
CONTIGUOUS, never a scatter of unrelated fragments across every wall simultaneously.

`perturb_wall_segments` (Section 2) is UNCHANGED this cycle and NOT deleted (per
instruction) -- it still produces exactly that fragment-scatter model, now reachable as
`--perturb-mode fragment`. Section 2b (`perturb_wall_segments_structured`) is the NEW
DEFAULT (`--perturb-mode structured`): it draws from the EXACT SAME QA-owned budget ranges
as Section 2 (`REMOVE_FRAC_RANGE` / `SHIFT_FRAC_RANGE` / `SHIFT_DIST_RANGE` /
`NOISE_FRAC_RANGE` -- none redefined, none relaxed; only the EDIT UNIT changed, per
instruction "바꾸는 것은 어디에 분포하는가 이지 얼마나가 아니다"). The edit unit becomes a
WHOLE wall: `densify_wall_segments_with_ids` (Section 1c, new) tags every post-densify
fragment with which of the ORIGINAL (pre-densify) wall rows it was split from -- fragments
of one original wall are CONTIGUOUS in the output array by construction (the densifier
iterates input rows in order and emits all of one row's pieces before the next), so "a
contiguous run within a wall" is just a contiguous slice of that row's own index range, no
extra bookkeeping needed. With that tag:
  REMOVE -- whole walls, removed entirely, in a random order, until the drawn 10-30% budget
            is reached; only if the NEXT whole wall would overshoot it is that ONE wall cut
            down to a single contiguous bite sized to land EXACTLY on the budget (so the
            total removed COUNT matches `perturb_wall_segments`'s draw exactly -- same total
            change quantity, per instruction -- while every removed unit is a whole wall or
            one contiguous bite of one wall, never scattered fragments).
  SHIFT  -- whole (surviving) walls, translated together by ONE shared vector per wall
            (still 0.3-1.0 m, QA-owned, unchanged) -- a shifted wall stays a straight wall
            at its new location, never an independently-drifted fragment cloud. Candidates
            accumulate in random order until the shift budget is met or exceeded (a wall is
            never split to land on it exactly -- an as-built wall does not partially drift),
            so the achieved shift fraction is approximate, unlike remove's exact match;
            always reported alongside the drawn target, never silently substituted.
  ADD    -- exactly ONE new wall (one straight run of touching fragments, densified at the
            SAME piece length as every real wall), sized so its fragment count matches the
            drawn noise-fraction budget -- "새 벽 하나를 통째로 추가" (a room subdivision),
            not many independent furniture-scale blips (Section 2's noise model, which stays
            exactly as it was for `fragment` mode).

`--perturb-mode both` runs BOTH suites (same seed, same n_perturb, each mode's OWN
`derive_seeds` stream so the two suites are genuinely different draws, not the same draws
reinterpreted) and reports both sets of ①②③ numbers SIDE BY SIDE, never averaged or merged
-- they measure two different things (site-change robustness vs. drawing-noise robustness)
and merging them would hide which one the matcher is actually failing. Thresholds (90%/80%)
and every QA-owned budget range are UNCHANGED this cycle in BOTH modes -- this is a
model/harness correction (Cycle 7 asked "what does a fragment-scatter model measure";
Cycle 11 asks "what does it FAIL to measure, which is the team's actual DoD target"), not a
threshold relaxation. Per repeated instruction across cycles 7 and 11: if `structured`
STILL misses a gate, that is reported as a genuine matcher limitation (P2-Robust input,
including the new failure-reason breakdown, `failure_reason_breakdown`), never papered over
by loosening a threshold or a budget range.
===========================================================================================

PERTURBATION MODEL (operates on `scan2bim.dxf_plan.load_wall_segments`'s (N,2,2)
metre format, whether the array came from a real DXF or a synthetic fixture, AFTER
Section 1b's density pass has split it into many fragments per wall):
  (a) REMOVE   10-30% of wall segments (torn-down / undrawn walls)
  (b) SHIFT    some of the remaining segments 0.3-1.0 m (as-built drift / stale DXF)
  (c) ADD      noise segments the plan never had (furniture / temporary partitions)
All draws come from a single `numpy.random.default_rng(seed)` stream, so the same
seed reproduces byte-identical output (segments array AND ground_truth dict) --
see `selftest_reproducibility()` / the `--selftest-only` CLI flag. Doors are NOT
perturbed by this generator (only wall segments) -- the team DoD text this file's
Section 2 constants quote verbatim says "벽" (walls), not doors; the D2 gate's plan
fixture keeps its doors fixed across every perturbation for that reason (see
`run_d2_gate`). Section 2 itself is UNCHANGED this cycle (still operates on whatever
flat (N,2,2) array it is handed, fragment-by-fragment, with no notion of "which
fragments belong to the same original wall" -- it stays generic over any DXF-shaped
input, per this module's own design philosophy); only WHAT it is handed (Section 1b's
denser base) changed. Cycle 11 (see block above) adds a SECOND perturbation model,
Section 2b, whose edit unit IS "which fragments belong to the same original wall" --
selectable via `--perturb-mode {structured,fragment,both}` (default `structured`); the
description above still applies verbatim to `fragment` mode / Section 2.

BASE SYNTHETIC PLAN (Section 1, `synth_l_corridor_segments`): no real SXX/Gasan DXF
exists in this repo. Confirmed by `find / -iname '*.dxf' -not -path '*/.git/*'` under
the repo root (only unrelated files under /usr/share and the user's ~/다운로드 turned
up). This fixture stays INDEPENDENT of tests/ on purpose (re-derived, not imported) --
it is a 2-leg L (1 corner), which per `coarse_match.py`'s own documented load-bearing
claim is AMBIGUOUS by construction without a door, so it is used for the CLI's
generator self-test/demo, not for the D2 accuracy gate (see next paragraph for why the
gate uses a different, imported fixture).

D2 GATE FIXTURE (Section 3, `run_d2_gate` / `_run_d2_gate_single`): DELIBERATELY
DIFFERENT from Section 1's fixture, and DELIBERATELY imported from
`tests/test_coarse_match.py` rather than re-derived, per the cycle-6 instruction to
reuse that harness (still honoured this cycle). The STAIR corridor (4 legs / 3 corners
/ 5 doors) is UNAMBIGUOUS, which the 2-leg L is not -- the D2 success-rate metric needs
an unambiguous plan to be measuring ACCURACY under perturbation, not tie-breaking (the
tie-breaking case, item (3) of the gate, uses a SEPARATE fixed fixture reused from
`tests/test_plan_skeleton.py`'s 2-leg L instead, see `build_ambiguous_fixture`). This
cycle, `_run_d2_gate_single` runs `densify_wall_segments_with_ids` on that harness's raw
10-segment `_corridor_walls(STAIR)` output BEFORE perturbing it -- see Section 1b/1c /
Section 3 -- so BOTH perturbation models (structured and fragment) share the identical
densified base plan; only the perturbation step differs.

CLI:
    .venv/bin/python tools/check_plan_match_robust.py [--upload PATH] \\
        [--plan-dxf PLAN.dxf] [--n-perturb 20] [--seed 0] \\
        [--perturb-mode {structured,fragment,both}] [--selftest-only] [--json OUT.json]

`--upload` is accepted but UNUSED by the D2 gate: no real recon-walk-from-upload
extraction exists in this repo as of this cycle (that is separate P3 scope), and the
D2 gate needs a walk with a CONSTRUCTIVELY known correct answer to measure accuracy
against, which only the synthetic harness provides (dev-core's own test suite is
synthetic-only for the same reason, see its module docstring).

`--perturb-mode` (Cycle 11, default `structured`) selects the perturbation model: see
the CYCLE 11 block above. `both` runs and reports both models independently -- exit 0
only if EVERY item of BOTH suites passes; the printed report always separates them.

Exit codes: 0 = D2 gate PASS (success-rate AND outlier-recall AND ambiguous-HOLD all
met, for every mode run -- BOTH modes if `--perturb-mode both`), or `--selftest-only`
PASS for every mode run; 1 = a generator self-test failed, OR the D2 gate ran but one or
more of its three items missed the threshold in ANY mode run (stated on stderr with the
measured number -- thresholds are never relaxed to make a number pass); 2 = the D2
gate could not be evaluated at all (matcher or tests/ harness import failed, the
harness's own zero-perturbation baseline was not recovered exactly, the density pass
changed the skeleton it was supposed to leave untouched, or the single-fragment-removal
regression probe found a topology collapse) -- NEVER a fabricated pass.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow running as a script from the repo root without PYTHONPATH=.
_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ==========================================================================================
# SECTION 1 -- base synthetic corridor plan (QA fixture: width ~=1.82 m, L corner)
# ==========================================================================================

DEFAULT_CORRIDOR_WIDTH = 1.82   # m -- Gasan_7F measured corridor clear width (fixture basis)


def synth_l_corridor_segments(width: float = DEFAULT_CORRIDOR_WIDTH,
                               leg_a_len: float = 12.0, leg_b_len: float = 10.0,
                               cap: bool = True, room: bool = True) -> np.ndarray:
    """Base synthetic L-corridor wall segments, (N,2,2) metres -- the SAME shape as
    `scan2bim.dxf_plan.load_wall_segments`'s output, so the perturbation generator
    (and, later, plan_skeleton.corridor_skeleton) can consume it unmodified.

    Corridor A runs along +y, x in [0, width]; corridor B runs along +x off the
    inner corner at (width, leg_a_len - width), same clear width. `room` hangs an
    out-of-width-band room off corridor B (must NOT be picked up as a corridor leg
    by anything consuming this fixture -- same negative-control intent as
    tests/test_plan_skeleton.py's room fixture, independently re-derived here)."""
    if width <= 0 or leg_a_len <= width or leg_b_len <= 0:
        raise ValueError(f"degenerate corridor geometry: width={width} leg_a_len={leg_a_len} "
                          f"leg_b_len={leg_b_len}")
    y_inner = leg_a_len - width
    segs = [((0.0, 0.0), (0.0, leg_a_len)),                  # outer wall, corridor A
            ((width, 0.0), (width, y_inner)),                # inner wall, corridor A
            ((width, y_inner), (leg_b_len, y_inner)),         # inner wall, corridor B
            ((0.0, leg_a_len), (leg_b_len, leg_a_len))]       # outer wall, corridor B
    if cap:
        segs.append(((0.0, 0.0), (width, 0.0)))               # dead end at y=0
    if room:                                                   # negative control: out-of-band room
        rx0, rx1 = leg_b_len * 0.3, leg_b_len * 0.7
        segs += [((rx0, leg_a_len), (rx0, leg_a_len + 3.0)),
                 ((rx1, leg_a_len), (rx1, leg_a_len + 3.0)),
                 ((rx0, leg_a_len + 3.0), (rx1, leg_a_len + 3.0))]
    return np.asarray(segs, dtype=np.float64).reshape(-1, 2, 2)


def _ezdxf():
    try:
        import ezdxf
        return ezdxf
    except ImportError as e:                                  # pragma: no cover - env guard
        raise ImportError("tools.check_plan_match_robust needs `ezdxf` to write a synthetic "
                           "DXF (pip install ezdxf)") from e


def write_synth_l_corridor_dxf(path, width: float = DEFAULT_CORRIDOR_WIDTH,
                                leg_a_len: float = 12.0, leg_b_len: float = 10.0,
                                doors=((DEFAULT_CORRIDOR_WIDTH, 4.0),)) -> Path:
    """Write `synth_l_corridor_segments()` out as a REAL ezdxf DXF (mm units,
    A-WALL/A-DOOR layers matching `scan2bim.dxf_plan._WALL_LAYERS`/`_DOOR_LAYERS`),
    so the generator can be exercised through the real
    `scan2bim.dxf_plan.load_wall_segments` file-read path, not only the in-memory
    array -- proves format compatibility end to end."""
    ezdxf = _ezdxf()
    doc = ezdxf.new(setup=True)
    doc.header["$INSUNITS"] = 4                                # mm, matches dxf_plan._UNIT_TO_M[4]
    doc.blocks.new(name="DOOR")
    msp = doc.modelspace()
    mm = 1000.0
    for (p, q) in synth_l_corridor_segments(width, leg_a_len, leg_b_len):
        msp.add_line((float(p[0]) * mm, float(p[1]) * mm), (float(q[0]) * mm, float(q[1]) * mm),
                     dxfattribs={"layer": "A-WALL-____-OTLN"})
    for (dx, dy) in doors:
        msp.add_blockref("DOOR", (float(dx) * mm, float(dy) * mm),
                         dxfattribs={"layer": "A-DOOR-____-OTLN"})
    out = Path(path)
    doc.saveas(out)
    return out


# ==========================================================================================
# SECTION 1b -- CYCLE 7 harness-density fix: split each wall into collinear fragments
# ==========================================================================================
#
# WHY A FIXED FRAGMENT LENGTH, AND WHY THIS VALUE (measured this cycle, do not re-derive
# to a different number without re-running the sweep below):
#
# The team doc's illustrative "벽 길이에 따라 3~8조각" was tried FIRST and MEASURED to fail:
# splitting the STAIR harness's 10 walls into 3-8 pieces each (`target_piece_len=3.0 m`,
# N=39 fragments) and running the UNCHANGED Section-2 perturbation suite (seed=7, n=20)
# gave success 0/20 -- WORSE than the pre-fix 2/20 baseline. The reason is mechanical, not
# a matcher weakness: `plan_skeleton.corridor_skeleton`'s own `wall_join=1.0 m` default
# bridges a gap between two collinear same-offset fragments ONLY if the gap is <= 1.0 m
# (`_wall_lines`, "gaps up to wall_join are bridged, so a door opening does not cut a
# wall"). A fragment of length > wall_join, if it is the ONLY thing removed, already opens
# a gap bigger than the bridge tolerance -- i.e. removing ONE 3 m fragment from a 6-fragment
# wall reproduces EXACTLY the whole-wall-vanishing collapse this cycle was opened to fix,
# just with a smaller, still-catastrophic piece. (Swept 2.0/1.5/1.0/0.7/0.5/0.3 m too --
# same failure mode down to ~1.0 m; below 1.0 m the failure rate drops sharply because a
# single fragment's gap finally stays inside `wall_join`.)
#
# So the fragment length is anchored to `wall_join` (the ONE dev-core constant that
# actually determines whether a fragment loss is bridgeable), not to an illustrative
# piece-count range: DENSIFY_TARGET_PIECE_LEN_M = 0.5 * wall_join, i.e. HALF of dev-core's
# own bridging tolerance, so a SINGLE fragment loss anywhere leaves a <= 0.5 m gap (well
# inside the 1.0 m bridge) with a 2x safety margin against also losing an immediate
# neighbour by chance. Measured regression probe for the ORIGINAL bug this produces (see
# `assert_single_fragment_removal_is_graceful`): 30/30 random single-fragment removals at
# this length leave n_legs/n_corners unchanged (was reproducibly 4->3 legs at the old
# one-segment-per-wall density). This does NOT mean the full 10-30% DoD perturbation band
# always succeeds -- see Section 3 / the cycle report for that number and why the
# remaining gap is a genuine (not harness-induced) P2-Robust input: removing AND shifting
# 10-30% each of ~230 fragments still occasionally strings together adjacent losses (a
# shift also evicts a fragment from its wall-line cluster, see Section 2's docstring)
# whose COMBINED gap exceeds wall_join, which is a cumulative-tolerance question for
# dev-core's matcher, not a single-fragment harness artefact.
#
# wall_join is NOT re-exported as a top-level constant by scan2bim/plan_skeleton.py (it is
# a keyword default on `corridor_skeleton`, read-only reference here, never overridden by
# this file's calls into it) -- the value below is a literal copy of that default for our
# OWN sizing decision, not a claim of ownership over it.
_DEV_CORE_WALL_JOIN_M = 1.0     # scan2bim/plan_skeleton.py corridor_skeleton(..., wall_join=1.0)

DENSIFY_TARGET_PIECE_LEN_M = 0.5 * _DEV_CORE_WALL_JOIN_M   # 0.5 m -- see rationale above
DENSIFY_MIN_PIECES_PER_WALL = 1
DENSIFY_MAX_PIECES_PER_WALL = 200   # defensive ceiling only; never the active constraint
                                    # at this fixture's 1.8-17.2 m wall lengths


def _piece_count(length: float, target: float = DENSIFY_TARGET_PIECE_LEN_M,
                 min_pieces: int = DENSIFY_MIN_PIECES_PER_WALL,
                 max_pieces: int = DENSIFY_MAX_PIECES_PER_WALL) -> int:
    """Fragments for one wall of `length` metres -- `round(length/target)`, clamped to
    [min_pieces, max_pieces]. A degenerate (<=0) length always returns 1 (nothing to
    split)."""
    if length <= 0:
        return 1
    return int(np.clip(round(length / target), min_pieces, max_pieces))


def densify_wall_segments_with_ids(segments,
                                   target_piece_len: float = DENSIFY_TARGET_PIECE_LEN_M,
                                   min_pieces: int = DENSIFY_MIN_PIECES_PER_WALL,
                                   max_pieces: int = DENSIFY_MAX_PIECES_PER_WALL) -> tuple:
    """CYCLE 11 addition. Identical splitting logic to `densify_wall_segments` (same
    output geometry, byte-for-byte -- `densify_wall_segments` now just delegates here and
    drops the second return value), but ALSO returns a parallel `wall_ids` int64 array
    (len == M, the output fragment count) giving, for every output fragment, the index
    (0..len(segments)-1) into the INPUT `segments` array of the original wall row it was
    split from. Fragments of one input row are CONTIGUOUS in the output array (this
    function iterates input rows in order and emits all of one row's pieces before moving
    to the next) -- Section 2b's `perturb_wall_segments_structured` (its STRUCTURED
    perturbation, the new default) relies on that contiguity to edit "a whole wall" or "a
    contiguous run within one wall" as a single index-range slice, with no separate
    bookkeeping.

    Returns (segments (M,2,2) float64, wall_ids (M,) int64)."""
    seg = np.asarray(segments, dtype=np.float64).reshape(-1, 2, 2)
    out = []
    wall_ids = []
    for wi, (p, q) in enumerate(seg):
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        d = q - p
        L = float(np.linalg.norm(d))
        n = _piece_count(L, target_piece_len, min_pieces, max_pieces)
        if L < 1e-9 or n <= 1:
            out.append((p, q))
            wall_ids.append(wi)
            continue
        u = d / L
        breaks = np.linspace(0.0, L, n + 1)
        for s0, s1 in zip(breaks, breaks[1:]):
            out.append((p + u * s0, p + u * s1))
            wall_ids.append(wi)
    return (np.asarray(out, dtype=np.float64).reshape(-1, 2, 2),
            np.asarray(wall_ids, dtype=np.int64))


def densify_wall_segments(segments,
                          target_piece_len: float = DENSIFY_TARGET_PIECE_LEN_M,
                          min_pieces: int = DENSIFY_MIN_PIECES_PER_WALL,
                          max_pieces: int = DENSIFY_MAX_PIECES_PER_WALL) -> np.ndarray:
    """Split every (p, q) wall segment into `_piece_count` COLLINEAR, TOUCHING
    (zero-gap) pieces of ~`target_piece_len` metres. Deterministic (no RNG) -- density
    is a property of the BASE plan, not of a perturbation seed; the SAME input always
    returns the SAME output.

    Zero-gap by construction (each piece's end IS the next piece's start, to float
    precision) means this is, on its own, a NO-OP on `plan_skeleton.corridor_skeleton`'s
    output: `_wall_lines` bridges any gap <= `wall_join` when merging same-offset
    segments into one wall LINE, and 0.0 <= wall_join trivially. See
    `assert_densify_preserves_skeleton`, which locks this as an assertion rather than
    trusting it -- run BEFORE any perturbation in `run_d2_gate`.

    CYCLE 11: delegates to `densify_wall_segments_with_ids` and drops the `wall_ids`
    column -- SAME geometry as before this cycle (verified by
    `assert_densify_preserves_skeleton`, which still runs on this function's output
    unchanged).

    Returns (M,2,2) float64, M >= len(segments) (M == len(segments) only for
    already-short walls where `_piece_count` returns 1, e.g. none at this fixture's
    wall lengths with the default target)."""
    segs, _wall_ids = densify_wall_segments_with_ids(segments, target_piece_len, min_pieces, max_pieces)
    return segs


def assert_densify_preserves_skeleton(corridor_skeleton_fn, raw_segments, densified_segments,
                                      doors=None) -> dict:
    """THE invariant Part (A) requires locked BEFORE any perturbation is applied:
    subdividing every wall into touching (zero-gap) collinear pieces must not change
    what `corridor_skeleton` sees -- differing ONLY in `info['n_segments']`. Compares
    n_legs, n_corners, n_doors, and the SORTED leg lengths (rounded to mm) between the
    raw and densified inputs. Returns the full comparison dict; `['ok']` is False on
    any mismatch -- `run_d2_gate` raises rather than proceeding on an unverified
    harness (never a silently-trusted split)."""
    raw_skel = corridor_skeleton_fn(raw_segments, doors=doors)
    dense_skel = corridor_skeleton_fn(densified_segments, doors=doors)
    raw_lens = sorted(round(g["length"], 3) for g in raw_skel["legs"])
    dense_lens = sorted(round(g["length"], 3) for g in dense_skel["legs"])
    n_legs_ok = len(raw_skel["legs"]) == len(dense_skel["legs"])
    n_corners_ok = raw_skel["info"].get("n_corners") == dense_skel["info"].get("n_corners")
    n_doors_ok = len(raw_skel["doors"]) == len(dense_skel["doors"])
    lens_ok = raw_lens == dense_lens
    ok = bool(n_legs_ok and n_corners_ok and n_doors_ok and lens_ok)
    return {"ok": ok,
            "raw_n_segments": int(len(np.asarray(raw_segments).reshape(-1, 2, 2))),
            "densified_n_segments": int(len(np.asarray(densified_segments).reshape(-1, 2, 2))),
            "raw_n_legs": len(raw_skel["legs"]), "densified_n_legs": len(dense_skel["legs"]),
            "raw_n_corners": raw_skel["info"].get("n_corners"),
            "densified_n_corners": dense_skel["info"].get("n_corners"),
            "raw_n_doors": len(raw_skel["doors"]), "densified_n_doors": len(dense_skel["doors"]),
            "raw_leg_lengths": raw_lens, "densified_leg_lengths": dense_lens}


def assert_single_fragment_removal_is_graceful(corridor_skeleton_fn, densified_segments, doors,
                                               true_n_legs: int, true_n_corners: int,
                                               n_trials: int = 30, seed: int = 0) -> dict:
    """Regression probe for the EXACT bug measured (main session) before this cycle:
    with the OLD one-segment-per-wall harness, removing ONE wall segment removed an
    ENTIRE wall face and collapsed a whole leg (4 legs -> 3 on one removal, -> 2 on
    two, -> 1 on three). Removes ONE fragment at a time from the DENSIFIED base
    (`n_trials` random single fragments -- NOT the 10-30% suite), and checks
    `corridor_skeleton` still reports the SAME n_legs/n_corners every time: a single
    fragment of a many-fragment wall going missing should now be a graceful,
    bridgeable gap (`wall_join`), never a topology collapse. `['ok']` is False if ANY
    trial fails -- `run_d2_gate` raises rather than reporting numbers measured against
    a harness that has not actually fixed the bug it was built to fix."""
    rng = np.random.default_rng(seed)
    seg = np.asarray(densified_segments, dtype=np.float64).reshape(-1, 2, 2)
    n = len(seg)
    trials = []
    for _ in range(min(int(n_trials), n)):
        i = int(rng.integers(0, n))
        segs = np.delete(seg, i, axis=0)
        skel = corridor_skeleton_fn(segs, doors=doors)
        ok = (len(skel["legs"]) == true_n_legs
              and skel["info"].get("n_corners") == true_n_corners)
        trials.append({"removed_index": i, "ok": bool(ok), "n_legs": len(skel["legs"]),
                       "n_corners": skel["info"].get("n_corners")})
    n_ok = sum(1 for t in trials if t["ok"])
    return {"ok": bool(trials) and n_ok == len(trials), "n_trials": len(trials),
            "n_graceful": n_ok, "detail": [t for t in trials if not t["ok"]]}


# ==========================================================================================
# SECTION 2 -- seeded FRAGMENT-scatter perturbation generator (QA-owned; `fragment` mode)
# ==========================================================================================

REMOVE_FRAC_RANGE = (0.10, 0.30)      # 벽 10~30% 제거 (팀 DoD 문구 그대로)
SHIFT_DIST_RANGE = (0.3, 1.0)         # 이동 세그먼트 변위 0.3~1.0 m (팀 DoD 문구 그대로)
SHIFT_FRAC_RANGE = (0.10, 0.30)       # 제거 후 남은 벽 중 이동 대상 비율 -- QA 판단치
NOISE_FRAC_RANGE = (0.05, 0.20)       # 잡음 세그먼트 개수 = 원본 개수 대비 비율 -- QA 판단치
NOISE_LENGTH_RANGE = (0.5, 3.0)       # 잡음 세그먼트 길이(가구/파티션 규모) -- QA 판단치


def perturb_wall_segments(segments, seed: int, *,
                          remove_frac_range=REMOVE_FRAC_RANGE,
                          shift_frac_range=SHIFT_FRAC_RANGE,
                          shift_dist_range=SHIFT_DIST_RANGE,
                          noise_frac_range=NOISE_FRAC_RANGE,
                          noise_length_range=NOISE_LENGTH_RANGE,
                          bbox_pad: float = 2.0) -> tuple:
    """Seeded perturbation of (N,2,2) wall segments simulating on-site plan drift.
    Deterministic: the SAME seed (and same other kwargs) reproduces a
    byte-identical `(segments, ground_truth)` pair every call -- every draw comes
    from one `numpy.random.default_rng(seed)` stream in a fixed order (remove ->
    shift -> noise), and the code path taken for a given seed is itself
    seed-determined, so re-running never desyncs the stream.

    UNCHANGED this cycle (Cycle 11 added Section 2b as a SEPARATE mode instead of
    editing this function -- see the module docstring's CYCLE 11 block for why this
    function's fragment-scatter model is a DIFFERENT measurement, not a superseded one).
    Still operates fragment-by-fragment on whatever flat (N,2,2) array it is given, with
    no notion of "which fragments came from the same original wall" -- this stays
    generic over any DXF-shaped input (real or synthetic), per this module's design.
    Fed Section 1b's denser base, SHIFT now moves an individual FRAGMENT rather than
    an entire wall face; since its lateral distance (0.3-1.0 m) is >> `offset_tol`
    (0.10 m, `plan_skeleton._wall_lines`), a shifted fragment detaches from its
    original wall-line cluster the same way a removed one does -- i.e. shift and
    remove now have a SIMILAR mechanical effect on wall-line coverage (both can evict
    a fragment), which the cycle report's residual-gap discussion (Section 1b) is
    about. Measured, not silently assumed.

    Returns (perturbed_segments (M,2,2) float64, ground_truth: dict) where
    ground_truth is the ANSWER KEY for the later outlier-recall check:
      seed, n_original, n_output, params (actual sampled fractions/counts)
      removed_original_indices        -- indices dropped from the input array
      shifted_original_indices        -- indices (in the INPUT array) that moved
      shift_vectors                   -- {str(index): [dx, dy]} metres
      kept_unchanged_original_indices -- input indices carried through untouched
      noise_output_indices            -- indices (in the OUTPUT array) that are
                                          pure noise, not derived from any input row
      output_to_original_index        -- len == n_output; output row -> input
                                          index, or None for a noise row
      changed_output_indices          -- shifted-output-rows UNION noise-output-rows;
                                          the set a matcher's outlier list should
                                          recall >= GATE_OUTLIER_RECALL_MIN of
                                          (Cycle 7: over the honest, observable/
                                          scoreable subset -- see Section 3)
    """
    seg0 = np.asarray(segments, dtype=np.float64).reshape(-1, 2, 2)
    n = len(seg0)
    if n == 0:
        raise ValueError("perturb_wall_segments: input segments array is empty")
    rng = np.random.default_rng(seed)

    # (a) remove 10-30%
    remove_frac = float(rng.uniform(*remove_frac_range))
    n_remove = int(round(n * remove_frac))
    n_remove = min(max(n_remove, 0), n - 1)                    # never remove every wall
    removed = sorted(int(i) for i in rng.choice(n, size=n_remove, replace=False)) if n_remove else []
    removed_set = set(removed)
    kept = [i for i in range(n) if i not in removed_set]

    # (b) shift some of the remainder 0.3-1.0 m
    shift_frac = float(rng.uniform(*shift_frac_range))
    n_shift = int(round(len(kept) * shift_frac))
    n_shift = min(max(n_shift, 0), len(kept))
    shift_idx = (sorted(int(i) for i in rng.choice(kept, size=n_shift, replace=False))
                 if n_shift else [])
    shift_set = set(shift_idx)
    shift_vectors: dict = {}
    for i in shift_idx:
        dist = float(rng.uniform(*shift_dist_range))
        angle = float(rng.uniform(0.0, 2 * np.pi))
        shift_vectors[i] = [round(dist * float(np.cos(angle)), 5),
                            round(dist * float(np.sin(angle)), 5)]

    # (c) add noise segments the plan never had
    noise_frac = float(rng.uniform(*noise_frac_range))
    n_noise = max(1, int(round(n * noise_frac)))
    flat = seg0.reshape(-1, 2)
    lo = flat.min(axis=0) - bbox_pad
    hi = flat.max(axis=0) + bbox_pad

    out_rows = []
    output_to_original = []
    for i in kept:
        p, q = seg0[i, 0].copy(), seg0[i, 1].copy()
        if i in shift_set:
            dv = np.asarray(shift_vectors[i], dtype=np.float64)
            p = p + dv
            q = q + dv
        out_rows.append((p, q))
        output_to_original.append(i)

    noise_output_indices = []
    for _ in range(n_noise):
        length = float(rng.uniform(*noise_length_range))
        angle = float(rng.uniform(0.0, 2 * np.pi))
        cx = float(rng.uniform(lo[0], hi[0]))
        cy = float(rng.uniform(lo[1], hi[1]))
        dx, dy = 0.5 * length * float(np.cos(angle)), 0.5 * length * float(np.sin(angle))
        p = np.array([cx - dx, cy - dy])
        q = np.array([cx + dx, cy + dy])
        noise_output_indices.append(len(out_rows))
        out_rows.append((p, q))
        output_to_original.append(None)

    out = np.asarray([[p, q] for p, q in out_rows], dtype=np.float64).reshape(-1, 2, 2)
    changed_output_indices = sorted(
        [oi for oi, orig in enumerate(output_to_original) if orig in shift_set] + noise_output_indices)

    ground_truth = {
        "seed": int(seed),
        "n_original": int(n),
        "n_output": int(len(out)),
        "params": {
            "remove_frac": round(remove_frac, 5), "n_removed": len(removed),
            "shift_frac": round(shift_frac, 5), "n_shifted": len(shift_idx),
            "noise_frac": round(noise_frac, 5), "n_noise": n_noise,
        },
        "removed_original_indices": removed,
        "shifted_original_indices": shift_idx,
        "shift_vectors": {str(k): v for k, v in shift_vectors.items()},
        "kept_unchanged_original_indices": sorted(i for i in kept if i not in shift_set),
        "noise_output_indices": noise_output_indices,
        "output_to_original_index": output_to_original,
        "changed_output_indices": changed_output_indices,
    }
    return out, ground_truth


# ==========================================================================================
# SECTION 2b -- CYCLE 11: STRUCTURED perturbation (new default; `structured` mode)
# ==========================================================================================
#
# See the module docstring's CYCLE 11 block for the full measured rationale. Short version:
# Section 2 treats every post-densify fragment as independently removable/shiftable, with no
# notion of which fragments came from the same original wall -- that models drawing/scan
# NOISE, not the team's DoD target of a real as-built CHANGE (a torn-down partition, a
# relocated wall, a new partition -- all structurally contiguous). `perturb_wall_segments_
# structured` draws from the EXACT SAME QA-owned budget constants as Section 2 (none
# redefined here) but edits whole walls (or, only where needed to hit the removal budget
# exactly, one contiguous bite out of one wall) instead of a scatter of unrelated fragments.


def _wall_groups(wall_ids) -> dict:
    """`wall_ids` (len N, from `densify_wall_segments_with_ids`) -> {wall_id: [global
    output indices]}, ascending. Fragments of one original wall are CONTIGUOUS in the
    output array by construction of `densify_wall_segments_with_ids` (it iterates input
    rows in order and emits all of one row's pieces before the next), so this is a
    straight grouping pass, not a claim that needs separate verification."""
    groups: dict = {}
    for gi, wi in enumerate(wall_ids):
        groups.setdefault(int(wi), []).append(int(gi))
    for wi in groups:
        groups[wi].sort()
    return groups


def perturb_wall_segments_structured(segments, wall_ids, seed: int, *,
                                     remove_frac_range=REMOVE_FRAC_RANGE,
                                     shift_frac_range=SHIFT_FRAC_RANGE,
                                     shift_dist_range=SHIFT_DIST_RANGE,
                                     noise_frac_range=NOISE_FRAC_RANGE,
                                     target_piece_len: float = DENSIFY_TARGET_PIECE_LEN_M,
                                     bbox_pad: float = 2.0) -> tuple:
    """CYCLE 11 STRUCTURED perturbation of (N,2,2) wall segments, simulating on-site
    CONSTRUCTION change (not drawing noise -- see Section 2 for that model).
    Deterministic: one `numpy.random.default_rng(seed)` stream, fixed draw order (remove
    -> shift -> add) -- same reproducibility contract as `perturb_wall_segments` (see
    `selftest_reproducibility`).

      remove -- whole walls (`wall_ids` groups), in a random order, removed entirely one
                at a time until the (10-30%, QA-owned) removal budget is reached; if the
                next whole wall would overshoot it, that ONE wall is cut down to a single
                CONTIGUOUS run (random start offset within it) sized to land EXACTLY on
                the budget instead -- so the total removed COUNT matches
                `perturb_wall_segments`'s draw exactly (same total-change quantity, per
                instruction), while every removed unit is a whole wall or one contiguous
                bite of one wall, never scattered fragments.
      shift   -- whole (surviving) walls translated together by ONE shared vector per
                wall (magnitude 0.3-1.0 m, QA-owned range, unchanged), so a shifted wall
                stays a straight wall at its new location, never an independently-drifted
                fragment cloud. Candidates accumulate in random order until the shift
                budget is MET OR EXCEEDED (a wall is never split to land on the budget
                exactly -- an as-built wall does not partially drift); the achieved
                fraction is therefore approximate, unlike remove's exact match -- always
                reported in `params` alongside the drawn target, never silently
                substituted.
      add     -- exactly ONE new wall (one straight run of touching fragments, densified
                at the SAME `target_piece_len` as every real wall passed to this
                function), its length sized so its fragment count matches the
                noise-fraction budget -- "새 벽 하나를 통째로 추가" (a room subdivision),
                not many independent furniture-scale blips (Section 2's noise model).

    `wall_ids` must be the SAME length as `segments` (see `densify_wall_segments_with_
    ids`) -- every output row of `segments` tagged with which ORIGINAL (pre-densify) wall
    row it came from. Returns (perturbed_segments (M,2,2) float64, ground_truth: dict) in
    the EXACT SAME schema as `perturb_wall_segments` (`removed_original_indices`,
    `shifted_original_indices`, `shift_vectors`, `output_to_original_index`,
    `changed_output_indices`, ...) -- every downstream Section 3 consumer
    (`classify_changed_segments`, `compute_outlier_recall`, ...) is mode-agnostic by
    design and needed NO changes for this mode. Adds one extra key,
    `ground_truth['structured']`, with the per-wall audit trail (which wall ids were
    removed whole / shifted whole, the one partial-removal wall if any, and the new
    wall's geometry) for reporting -- never read by the Section 3 scoring functions,
    informational only."""
    seg0 = np.asarray(segments, dtype=np.float64).reshape(-1, 2, 2)
    n = len(seg0)
    if n == 0:
        raise ValueError("perturb_wall_segments_structured: input segments array is empty")
    wids = np.asarray(wall_ids, dtype=np.int64).reshape(-1)
    if len(wids) != n:
        raise ValueError(f"perturb_wall_segments_structured: wall_ids length {len(wids)} "
                         f"!= segments length {n}")
    rng = np.random.default_rng(seed)
    groups = _wall_groups(wids)

    # ---- (a) remove: whole walls, greedy random order, exact budget via one partial bite.
    remove_frac = float(rng.uniform(*remove_frac_range))
    n_remove_target = int(round(n * remove_frac))
    n_remove_target = min(max(n_remove_target, 0), n - 1)      # never remove every fragment

    order = list(groups.keys())
    rng.shuffle(order)
    removed_indices: list = []
    removed_whole_wall_ids: list = []
    partial_remove = None
    budget = n_remove_target
    for wi in order:
        if budget <= 0:
            break
        gidx = groups[wi]
        m = len(gidx)
        if m <= budget:
            removed_indices.extend(gidx)
            removed_whole_wall_ids.append(int(wi))
            budget -= m
        else:
            run_len = budget
            start = int(rng.integers(0, m - run_len + 1))
            removed_indices.extend(gidx[start:start + run_len])
            partial_remove = {"wall_id": int(wi), "start": start, "len": run_len}
            budget = 0
            break
    removed_set = set(removed_indices)
    kept = [i for i in range(n) if i not in removed_set]

    # ---- (b) shift: whole (surviving) walls, greedy random order, budget met-or-exceeded.
    kept_groups: dict = {}
    for i in kept:
        kept_groups.setdefault(int(wids[i]), []).append(i)

    shift_frac = float(rng.uniform(*shift_frac_range))
    n_shift_target = int(round(len(kept) * shift_frac))
    n_shift_target = min(max(n_shift_target, 0), len(kept))

    shift_order = list(kept_groups.keys())
    rng.shuffle(shift_order)
    shift_set: set = set()
    shift_vectors: dict = {}
    shifted_whole_wall_ids: list = []
    sbudget = n_shift_target
    for wi in shift_order:
        if sbudget <= 0:
            break
        gidx = kept_groups[wi]
        dist = float(rng.uniform(*shift_dist_range))
        angle = float(rng.uniform(0.0, 2 * np.pi))
        dv = [round(dist * float(np.cos(angle)), 5), round(dist * float(np.sin(angle)), 5)]
        for i in gidx:
            shift_set.add(i)
            shift_vectors[i] = dv
        shifted_whole_wall_ids.append(int(wi))
        sbudget -= len(gidx)

    # ---- (c) add: exactly one new wall, densified at the same piece length.
    noise_frac = float(rng.uniform(*noise_frac_range))
    n_noise_target = max(1, int(round(n * noise_frac)))
    new_wall_len = n_noise_target * target_piece_len
    flat = seg0.reshape(-1, 2)
    lo = flat.min(axis=0) - bbox_pad
    hi = flat.max(axis=0) + bbox_pad
    angle = float(rng.uniform(0.0, 2 * np.pi))
    cx = float(rng.uniform(lo[0], hi[0]))
    cy = float(rng.uniform(lo[1], hi[1]))
    ux, uy = float(np.cos(angle)), float(np.sin(angle))
    p0 = np.array([cx - 0.5 * new_wall_len * ux, cy - 0.5 * new_wall_len * uy])
    q0 = np.array([cx + 0.5 * new_wall_len * ux, cy + 0.5 * new_wall_len * uy])
    new_wall_pieces, _new_wall_ids = densify_wall_segments_with_ids(
        np.asarray([[p0, q0]]), target_piece_len=target_piece_len)

    # ---- assemble output rows: kept (with shift applied) + new-wall pieces.
    out_rows = []
    output_to_original = []
    for i in kept:
        p, q = seg0[i, 0].copy(), seg0[i, 1].copy()
        if i in shift_set:
            dv = np.asarray(shift_vectors[i], dtype=np.float64)
            p = p + dv
            q = q + dv
        out_rows.append((p, q))
        output_to_original.append(i)

    noise_output_indices = []
    for p, q in new_wall_pieces:
        noise_output_indices.append(len(out_rows))
        out_rows.append((p, q))
        output_to_original.append(None)

    out = np.asarray([[p, q] for p, q in out_rows], dtype=np.float64).reshape(-1, 2, 2)
    changed_output_indices = sorted(
        [oi for oi, orig in enumerate(output_to_original) if orig in shift_set] + noise_output_indices)

    ground_truth = {
        "seed": int(seed), "n_original": int(n), "n_output": int(len(out)),
        "params": {
            "remove_frac": round(remove_frac, 5), "n_removed": len(removed_indices),
            "shift_frac": round(shift_frac, 5), "n_shifted": len(shift_set),
            "noise_frac": round(noise_frac, 5), "n_noise": len(noise_output_indices),
        },
        "removed_original_indices": sorted(removed_indices),
        "shifted_original_indices": sorted(shift_set),
        "shift_vectors": {str(k): v for k, v in shift_vectors.items()},
        "kept_unchanged_original_indices": sorted(i for i in kept if i not in shift_set),
        "noise_output_indices": noise_output_indices,
        "output_to_original_index": output_to_original,
        "changed_output_indices": changed_output_indices,
        "structured": {
            "removed_whole_wall_ids": sorted(removed_whole_wall_ids),
            "partial_remove": partial_remove,
            "shifted_whole_wall_ids": sorted(shifted_whole_wall_ids),
            "new_wall": {"p0": p0.tolist(), "q0": q0.tolist(),
                        "length_m": round(new_wall_len, 4), "n_pieces": len(noise_output_indices)},
        },
    }
    return out, ground_truth


# ==========================================================================================
# SECTION 2c -- CYCLE 12: TOTAL-CHANGE-RATIO sweep generator (`--sweep`)
# ==========================================================================================
#
# WHY THIS EXISTS, AND WHAT IT DOES NOT DO (read before touching): Section 2b draws
# remove_frac / shift_frac / noise_frac INDEPENDENTLY, each from its OWN QA-owned range
# (REMOVE_FRAC_RANGE / SHIFT_FRAC_RANGE / NOISE_FRAC_RANGE -- UNCHANGED here, still the
# default D2 gate's budget). Measured (cycle-11 session, seed=7, n_perturb=20,
# structured): stacking three INDEPENDENT 10-30%/10-30%/5-20% draws means every case
# changes roughly 30-75% of the base plan's 232 fragments (removed 24-67 + shifted
# 26-62 + noise 14-46), because the three draws are never small TOGETHER by
# construction -- there is no way, using ONLY Section 2b's own knobs, to ask "what if
# the total site change were 5%". This section does NOT touch, replace, or relax
# REMOVE_FRAC_RANGE / SHIFT_FRAC_RANGE / NOISE_FRAC_RANGE, and it is NEVER called by
# `run_d2_gate` / `_run_d2_gate_single` (the default gate path, still exactly Section
# 2b's independent-range draw, unchanged, see the `--n-perturb 20 --seed 7` no-regression
# check in the cycle report) -- it is an ADDITIONAL, separate axis of measurement
# (`--sweep`), per instruction: "게이트를 통과시키려 정의를 또 고치는 것은 금지... 그러나
# '몇 % 변경까지 강건한가'는 정의 변경이 아니라 추가 측정이다".
#
# ALLOCATION RULE (how a single TOTAL ratio T -- e.g. 0.05 for "5% of the 232 base
# fragments changed" -- is split into a remove/shift/noise triple), stated explicitly per
# instruction ("배분 규칙을 명시하라"): reuse the RELATIVE WEIGHT the three QA-owned
# ranges' OWN midpoints already imply, and split T in those SAME proportions -- no new
# weighting is invented:
#   mid_remove          = mean(REMOVE_FRAC_RANGE)                = 0.20   (of N)
#   mid_shift_of_kept    = mean(SHIFT_FRAC_RANGE)                 = 0.20   (of KEPT,
#                          i.e. N - removed -- Section 2b's own parameterisation)
#   mid_noise            = mean(NOISE_FRAC_RANGE)                 = 0.125  (of N)
#   mid_shift_of_total   = mid_shift_of_kept * (1 - mid_remove)   = 0.16   (of N --
#                          converted to the SAME "of N" basis as the other two so the
#                          three CAN be added; shift's own knob stays "of kept" in the
#                          generator itself, unchanged -- this conversion is only for
#                          computing the WEIGHT)
#   (w_remove, w_shift, w_noise) = the three "of N" midpoints above, normalised to sum to
#                          1: (0.20, 0.16, 0.125) / 0.485 =~ (0.412, 0.330, 0.258), see
#                          `_sweep_allocation_weights` / `SWEEP_ALLOC_WEIGHTS` (computed,
#                          not hand-typed, from the three live QA constants).
# Given a target T, `perturb_wall_segments_structured_at_total_ratio` sets:
#   remove_frac        = T * w_remove   (EXACT -- Section 2b's own remove-to-budget
#                        mechanism, the "one partial bite" trick, lands on this count
#                        exactly, same guarantee as the default path)
#   noise_frac          = T * w_noise    (EXACT, ditto for the one-new-wall mechanism)
#   shift_frac_of_kept  = (T * w_shift * N) / max(1, N - round(N*remove_frac))
#                        (Section 2b's shift mechanism only guarantees MET-OR-EXCEEDED,
#                        same caveat as the default path -- see its own docstring; the
#                        ACHIEVED total ratio is therefore reported per-case, never
#                        assumed to equal T exactly)
# All three are passed to `perturb_wall_segments_structured` (Section 2b, UNCHANGED) as
# DEGENERATE ranges `(x, x)` (`numpy.random.Generator.uniform(x, x) == x`) -- the EXACT
# SAME function the default gate already uses, called with a single point instead of a
# band, so every byte of the remove/shift/noise MECHANISM (whole-wall edit unit, one
# contiguous partial bite, one new wall) is shared, unchanged, with the default path;
# only the SAMPLING WIDTH of the three fractions differs (a point, not a 10-30%-style
# band, because the sweep's own independent axis IS the total ratio -- widening each of
# the three around it as well would re-introduce Section 2b's stacking effect this
# section exists to avoid).
#
# `shift_dist_range` (0.3-1.0 m, the DISPLACEMENT magnitude, not a fraction of anything)
# is UNCHANGED and NOT part of "total change ratio" -- QA-owned, still the DoD text's
# literal band, passed through unmodified.


def _sweep_allocation_weights() -> tuple:
    """(w_remove, w_shift, w_noise), summing to 1.0 -- see the block comment above this
    function for the derivation. Computed FROM REMOVE_FRAC_RANGE / SHIFT_FRAC_RANGE /
    NOISE_FRAC_RANGE (QA-owned, unchanged) rather than hardcoded, so if those ranges are
    ever legitimately revised the sweep's allocation stays consistent with them
    automatically -- this function does NOT redefine or relax any of the three ranges."""
    mid_remove = 0.5 * (REMOVE_FRAC_RANGE[0] + REMOVE_FRAC_RANGE[1])
    mid_shift_of_kept = 0.5 * (SHIFT_FRAC_RANGE[0] + SHIFT_FRAC_RANGE[1])
    mid_noise = 0.5 * (NOISE_FRAC_RANGE[0] + NOISE_FRAC_RANGE[1])
    mid_shift_of_total = mid_shift_of_kept * (1.0 - mid_remove)
    total = mid_remove + mid_shift_of_total + mid_noise
    return (mid_remove / total, mid_shift_of_total / total, mid_noise / total)


#: (w_remove, w_shift, w_noise) -- see `_sweep_allocation_weights` / the block comment
#: above it. NOT used by the default D2 gate path (`run_d2_gate` / Section 2b), only by
#: `--sweep` (this section).
SWEEP_ALLOC_WEIGHTS = _sweep_allocation_weights()

#: `--sweep`'s default total-change-ratio axis points (fraction of the 232-fragment
#: densified STAIR base) -- 5/10/15/20/30/40/50%, per this cycle's instruction.
SWEEP_RATIOS_DEFAULT = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)

#: `--sweep`'s default seeds -- at least 3, per this cycle's instruction, so every ratio
#: point is reported as a mean AND a range, not a single (possibly lucky/unlucky) draw.
SWEEP_SEEDS_DEFAULT = (7, 42, 123)


def perturb_wall_segments_structured_at_total_ratio(segments, wall_ids, seed: int,
                                                     total_frac: float, *,
                                                     alloc_weights=SWEEP_ALLOC_WEIGHTS,
                                                     shift_dist_range=SHIFT_DIST_RANGE,
                                                     target_piece_len: float = DENSIFY_TARGET_PIECE_LEN_M,
                                                     bbox_pad: float = 2.0) -> tuple:
    """`--sweep` generator: a SINGLE-POINT (not a range) structured perturbation whose
    remove+shift+noise fragment count sums to (approximately -- see the module-level
    block comment's shift caveat) `total_frac` of `len(segments)`. Delegates entirely to
    `perturb_wall_segments_structured` (Section 2b, UNCHANGED) with degenerate `(x, x)`
    ranges for remove_frac/shift_frac/noise_frac -- same mechanism, same ground_truth
    schema (plus the same `ground_truth['structured']` audit trail), just a point
    instead of a band. Returns (perturbed_segments, ground_truth), same shape as every
    other perturbation function in this file."""
    seg0 = np.asarray(segments, dtype=np.float64).reshape(-1, 2, 2)
    n = len(seg0)
    w_r, w_s, w_n = alloc_weights
    t = float(total_frac)
    remove_frac = t * w_r
    noise_frac = t * w_n
    n_remove_est = int(round(n * remove_frac))
    n_remove_est = min(max(n_remove_est, 0), max(n - 1, 0))
    kept_est = max(1, n - n_remove_est)
    shift_frac_of_kept = (t * w_s * n) / kept_est
    return perturb_wall_segments_structured(
        segments, wall_ids, seed,
        remove_frac_range=(remove_frac, remove_frac),
        shift_frac_range=(shift_frac_of_kept, shift_frac_of_kept),
        shift_dist_range=shift_dist_range,
        noise_frac_range=(noise_frac, noise_frac),
        target_piece_len=target_piece_len, bbox_pad=bbox_pad)


def generate_perturbation_suite_at_total_ratio(base_segments, wall_ids, seed: int,
                                               n_perturb: int, total_frac: float,
                                               **kwargs) -> list:
    """`--sweep`'s per-ratio-point suite: `n_perturb` perturbations at a FIXED
    `total_frac` (WHICH walls/where noise lands still varies per child seed --
    `derive_seeds`, same reproducibility contract as `generate_perturbation_suite`).
    Item shape matches `generate_perturbation_suite`'s (`mode` is always `'structured'`
    here -- the sweep is deliberately structured-only, the site-change model, see the
    module docstring's CYCLE 11 block for why fragment-scatter is a different
    measurement), plus `total_frac_target` for the report."""
    child_seeds = derive_seeds(seed, n_perturb)
    suite = []
    for i, cs in enumerate(child_seeds):
        segs, gt = perturb_wall_segments_structured_at_total_ratio(
            base_segments, wall_ids, cs, total_frac, **kwargs)
        suite.append({"index": i, "seed": int(seed), "child_seed": int(cs), "segments": segs,
                      "ground_truth": gt, "mode": "structured", "total_frac_target": float(total_frac)})
    return suite


def derive_seeds(seed: int, n: int) -> list:
    """n independent child seeds from one base seed via numpy's SeedSequence.spawn
    (reproducible AND statistically independent across suite members -- unlike
    `seed + i`, which would correlate adjacent perturbations)."""
    ss = np.random.SeedSequence(int(seed))
    children = ss.spawn(int(n))
    return [int(c.generate_state(1)[0]) for c in children]


def generate_perturbation_suite(base_segments, seed: int, n_perturb: int, *,
                                mode: str = "structured", wall_ids=None,
                                **perturb_kwargs) -> list:
    """n_perturb perturbations of `base_segments`, deterministic in (seed, n_perturb,
    mode, **perturb_kwargs). CYCLE 11: `mode` selects the edit-unit model -- 'structured'
    (default, Section 2b, `perturb_wall_segments_structured` -- REQUIRES `wall_ids`, see
    `densify_wall_segments_with_ids`) or 'fragment' (Cycle-7's original model, Section 2,
    `perturb_wall_segments`, unchanged, no `wall_ids` needed). Each item: {"index",
    "seed", "child_seed", "segments", "ground_truth", "mode"}."""
    if mode not in ("structured", "fragment"):
        raise ValueError(f"generate_perturbation_suite: mode must be 'structured' or "
                         f"'fragment', got {mode!r}")
    if mode == "structured" and wall_ids is None:
        raise ValueError("generate_perturbation_suite: mode='structured' requires "
                         "wall_ids (see densify_wall_segments_with_ids)")
    child_seeds = derive_seeds(seed, n_perturb)
    suite = []
    for i, cs in enumerate(child_seeds):
        if mode == "fragment":
            segs, gt = perturb_wall_segments(base_segments, cs, **perturb_kwargs)
        else:
            segs, gt = perturb_wall_segments_structured(base_segments, wall_ids, cs, **perturb_kwargs)
        suite.append({"index": i, "seed": int(seed), "child_seed": int(cs), "segments": segs,
                      "ground_truth": gt, "mode": mode})
    return suite


def selftest_reproducibility(base_segments, seed: int = 0, n_perturb: int = 5, *,
                             mode: str = "structured", wall_ids=None) -> dict:
    """Runs `generate_perturbation_suite` TWICE with identical arguments (INCLUDING
    `mode`/`wall_ids`, Cycle 11) and checks byte-identical output (segments array
    equality + ground_truth JSON equality) -- the reproducibility requirement the CLI
    must demonstrate every invocation, for whichever perturbation model is selected."""
    run1 = generate_perturbation_suite(base_segments, seed, n_perturb, mode=mode, wall_ids=wall_ids)
    run2 = generate_perturbation_suite(base_segments, seed, n_perturb, mode=mode, wall_ids=wall_ids)
    mismatches = []
    for i, (a, b) in enumerate(zip(run1, run2)):
        same_array = a["segments"].shape == b["segments"].shape and bool(
            np.array_equal(a["segments"], b["segments"]))
        same_gt = json.dumps(a["ground_truth"], sort_keys=True) == json.dumps(b["ground_truth"], sort_keys=True)
        same_child_seed = a["child_seed"] == b["child_seed"]
        if not (same_array and same_gt and same_child_seed):
            mismatches.append({"index": i, "same_array": same_array, "same_gt": same_gt,
                               "same_child_seed": same_child_seed})
    return {"ok": not mismatches, "seed": int(seed), "n_perturb": int(n_perturb), "mode": mode,
            "mismatches": mismatches}


# ==========================================================================================
# SECTION 3 -- D2 gate: real wiring against scan2bim/coarse_match.py
# ==========================================================================================

GATE_N_PERTURB_DEFAULT = 20        # D2: "섭동 20개"
GATE_SUCCESS_RATE_MIN = 0.90       # D2: 성공률 >= 90%  -- UNCHANGED this cycle
GATE_OUTLIER_RECALL_MIN = 0.80     # D2: outlier recall >= 80%  -- UNCHANGED this cycle

#: D2 accuracy band ("이 파이프라인이 주장하는 정확도" -- 팀 DoD 문구 그대로), applied to a
#: perturbed-plan candidate's transform against the harness's CONSTRUCTIVELY known true
#: transform. Independently defined here (NOT imported from scan2bim/coarse_match.py or
#: tests/test_coarse_match.py's D2_* constants, even though the numbers coincide -- this
#: team's design puts judgement thresholds under QA ownership, not dev's).
GATE_YAW_DEG_MAX = 5.0             # D2: yaw <= 5 deg
GATE_OFFSET_WIDTH_FRAC = 0.5       # D2: translation <= 0.5 x corridor clear width
GATE_SCALE_REL_MAX = 0.10          # D2: per-axis (s_f, s_h) relative error <= 10%

#: Outlier-recall "same regional change" / "observable" radius (metres, QA judgement
#: call, plan frame). Cycle 7 gives this constant a SECOND job (see
#: `classify_changed_segments`): a changed wall segment is OBSERVABLE at all only if its
#: TRUE location is within this radius of the walk's own plan-frame path -- a walk-vs-plan
#: mismatch mechanism cannot, by construction, see a change it never walked near (measured
#: cycle-7 case: a noise wall 21.8 m from the walk). The SAME radius also still gates
#: whether a detected outlier "counts" as the SAME regional change once observability has
#: already been established (its original job, unchanged): a changed wall segment counts
#: as RECALLED if some outlier's position (its recon-side event, PROJECTED into plan
#: metres through the case's own winning candidate transform --
#: `plan_skeleton.apply_candidate_transform`, the only transform available operationally)
#: lands within this radius of the changed segment's TRUE location. Set to 3x the corridor
#: clear width (~5.46 m on the 1.82 m Gasan_7F value): wide enough that projecting through
#: a transform that is only D2-ACCURATE (yaw off by up to 5 deg, offset off by up to 0.5x
#: width -- not exact) does not itself manufacture a miss; tight enough that it cannot span
#: from one 11-17 m STAIR-fixture leg to the next, i.e. this tests "was the RIGHT REGION
#: flagged" / "was the region even walked", not sub-metre agreement.
OUTLIER_RECALL_RADIUS_M = 3.0 * DEFAULT_CORRIDOR_WIDTH

# ---- CYCLE 19: precision-paired gate items (dev-core self-reported flood loophole) ----
#
# dev-core's own commit (8d7fcbf) that raised recall 38.8% -> 91.2% flagged the exact gap
# this section closes, VERBATIM: "FLOOD 상한: 84개 span 을 전부 깃발 꽂으면 recall 100%. ②는
# 커버리지로 포화되므로 recall 수치만으로는 검출과 범람을 구별할 수 없다. 구별하는 숫자는
# 정밀도다." Measured (this cycle, `selftest_flood_detection`): patching
# `scan2bim.coarse_match._span_outliers` to unconditionally flag EVERY span (no geometry
# check at all) pushes item② recall toward its ceiling too -- recall alone, unmodified,
# does not by itself catch this, exactly as reported. Two NEW gate items close the
# loophole, both scored on the SAME span-outlier mechanism recall already reads
# (`candidate['outliers']`, `kind=='span'`), never on a second copy of the matcher:
#
#   ④ CLEAN-PLAN FALSE-ALARM RATE. The unperturbed base plan (STAIR, densified, the SAME
#      "clean baseline" `_build_d2_harness_context` already builds and sanity-checks) has
#      ZERO real changes by construction -- there is nothing ambiguous to weigh here, so
#      the target is exactly 0%, not a tolerance band: `GATE_SPAN_CLEAN_FPR_MAX = 0.0`.
#      A NONZERO value here means the R1/R2 rules' own built-in tolerances
#      (`SPAN_CROSS_MARGIN`, `SPAN_WIDTH_TOL`, scan2bim/coarse_match.py) are mis-set for
#      THIS plan, independent of any perturbation -- exactly the kind of defect a flood
#      implementation (or a tolerance regression) would introduce, caught with zero
#      ground-truth machinery at all.
#
#   ⑤ PRECISION-OVER-PREVALENCE. `GATE_SPAN_PRECISION_PREVALENCE_FACTOR = 1.5`. Physical
#      basis (NOT fit to any observed number): an "always fire" (flood) detector's
#      precision is, BY CONSTRUCTION, exactly equal to the base rate of true positives in
#      whatever it is applied to (TP = every real positive, FP = every real negative,
#      precision = positives / total = prevalence) -- flooding can therefore NEVER exceed
#      prevalence, no matter how severe a given perturbation suite's actual changes are.
#      Requiring precision to clear prevalence by a real margin (1.5x, not the fragile
#      exact-1.0x boundary a single lucky/unlucky case could cross either way by chance)
#      is a DATA-RELATIVE test: it self-scales to however much of the plan a given suite
#      actually changed (`compute_span_precision`'s own `prevalence` field, computed
#      fresh from ground truth every run) -- this IS the "실제 변경분이 차지하는 span 비율의
#      기댓값" this cycle's instruction asked the ceiling be grounded in. The measured
#      24-25/84 span-firing rate dev-core's commit cited is NEVER read by this threshold
#      (it is only ever printed as a data point) -- a threshold copied from one run's
#      observed number would catch nothing (per instruction: "현재 값에 맞춘 임계를 만들지
#      마라").
#
# Both items are scored ALONGSIDE ①②③ (same `gate*_ok` pattern, same
# `gate_ok = all(...)` composite) -- NEVER folded silently into ②'s own number, so a FAIL
# is always attributable to a SPECIFIC item (see `_print_single_gate_report`'s ④/⑤ lines
# and `main`'s FAIL-reason decomposition). `selftest_flood_detection` is the regression
# test that proves ④/⑤ actually reject the exact failure mode dev-core reported -- run it
# via `--selftest-flood`.
GATE_SPAN_CLEAN_FPR_MAX = 0.0
GATE_SPAN_PRECISION_PREVALENCE_FACTOR = 1.5

#: Defensive fallback only -- every schema-valid result carries its OWN 'gates' dict
#: (RESULT_FIELDS), which `verify_hold_on_ambiguous` reads instead of this constant.
_FALLBACK_MARGIN_MIN = 0.08


def try_import_matcher():
    """Returns the scan2bim.coarse_match module, or None if it does not exist yet."""
    try:
        from scan2bim import coarse_match
        return coarse_match
    except ImportError:
        return None


def _wrap180(a: float) -> float:
    return float((float(a) + 180.0) % 360.0 - 180.0)


def _transform_within_d2(got: dict, want: dict, *, yaw_max: float = GATE_YAW_DEG_MAX,
                         scale_rel_max: float = GATE_SCALE_REL_MAX, offset_max: float) -> tuple:
    """Independent QA check of the D2 accuracy band (handedness, yaw, per-axis scale,
    translation) between a candidate's transform (`got`) and the harness's
    CONSTRUCTIVELY KNOWN true transform (`want`). Deliberately NOT a call into
    tests/test_coarse_match.py's `_MatchAssertions.assert_transform_close` (that is a
    unittest assertion method, this returns a verdict this file's own gate consumes) --
    but the same five checks, because the D2 numbers are the team's DoD, not this
    file's invention. Returns (within_d2: bool, detail: dict) so a miss states WHICH
    axis missed by how much."""
    d = {}
    d["chi_ok"] = int(got["chi"]) == int(want["chi"])
    yaw_err = abs(_wrap180(float(got["yaw_deg"]) - float(want["yaw_deg"])))
    d["yaw_err_deg"] = round(yaw_err, 3)
    d["yaw_ok"] = yaw_err <= yaw_max
    for k in ("s_f", "s_h"):
        rel = abs(float(got[k]) - float(want[k])) / float(want[k])
        d[f"{k}_rel_err"] = round(rel, 5)
        d[f"{k}_ok"] = rel <= scale_rel_max
    off = float(np.linalg.norm(np.asarray(got["translation"], dtype=np.float64)
                               - np.asarray(want["translation"], dtype=np.float64)))
    d["offset_m"] = round(off, 4)
    d["offset_max_m"] = round(float(offset_max), 4)
    d["offset_ok"] = off <= offset_max
    within = d["chi_ok"] and d["yaw_ok"] and d["s_f_ok"] and d["s_h_ok"] and d["offset_ok"]
    return within, d


def compute_success_rate(cases: list, true_transform: dict, offset_max: float) -> dict:
    """D2 item (1) over `cases` (see `run_d2_gate` for their shape). Deliberately NOT
    `status=='ok'` alone -- that only proves the matcher's OWN internal gates were
    satisfied, and this team's QA mandate is to verify the NUMBER against the harness's
    constructively-known true transform, not trust the matcher's self-report. A case is
    a SUCCESS iff status=='ok' AND its `best` transform is inside the D2 band
    (`_transform_within_d2`) of `true_transform`. hold/reject are NOT successes for
    this metric -- see `verify_hold_gate` / the cycle report for why a HOLD is scored
    separately as a SAFE non-answer, never folded into "success" to inflate this rate."""
    detail = []
    n_ok_within = n_ok_outside = n_hold = n_reject = n_error = 0
    for c in cases:
        res = c.get("result")
        if c.get("error") is not None or res is None:
            n_error += 1
            detail.append({"index": c["index"], "outcome": "error", "error": c.get("error")})
            continue
        status = res.get("status")
        if status == "ok":
            within, d = _transform_within_d2(res["best"]["transform"], true_transform,
                                             offset_max=offset_max)
            if within:
                n_ok_within += 1
                detail.append({"index": c["index"], "outcome": "ok_within_d2", **d})
            else:
                n_ok_outside += 1                 # dangerous: a CONFIDENT wrong answer
                detail.append({"index": c["index"], "outcome": "ok_OUTSIDE_d2", **d})
        elif status == "hold":
            n_hold += 1
            detail.append({"index": c["index"], "outcome": "hold",
                           "hold_reason": res.get("hold_reason"), "margin": res.get("margin")})
        else:
            n_reject += 1
            detail.append({"index": c["index"], "outcome": "reject",
                           "hold_reason": res.get("hold_reason")})
    n = len(cases)
    rate = (n_ok_within / n) if n else 0.0
    return {"rate": rate, "n": n, "n_ok_within_d2": n_ok_within, "n_ok_outside_d2": n_ok_outside,
            "n_hold": n_hold, "n_reject": n_reject, "n_error": n_error, "detail": detail}


def failure_reason_breakdown(succ: dict) -> dict:
    """CYCLE 11 P2 input: reason distribution of every NON-success case in
    `compute_success_rate`'s `detail` -- the question PM asked, "is inlier_ratio_below_min
    DOMINANT", which if true means the matcher is REJECTING THE WHOLE CANDIDATE instead of
    carving the changed wall out as an outlier (the trimmed/RANSAC design's central
    promise -- see the module docstring). Counts every HOLD_REASONS/REJECT_REASONS string
    seen among hold/reject outcomes, PLUS two buckets that are NOT a `hold_reason` by
    schema: 'ok_OUTSIDE_d2' (status=='ok', so `hold_reason` is forced None by
    `plan_skeleton`'s own contract -- this is a CONFIRMED but WRONG transform, the
    dangerous case) and 'error' (the matcher/skeleton build raised). `ok_within_d2` cases
    (successes) are excluded -- this is a FAILURE breakdown only. Returns
    {reason_or_outcome: count}, insertion-ordered by first occurrence (NOT sorted --
    callers wanting a ranked view should sort by count themselves, e.g. via
    `max(d.items(), key=...)`, done in the CLI report)."""
    counts: dict = {}
    for d in succ["detail"]:
        if d["outcome"] == "ok_within_d2":
            continue
        key = d.get("hold_reason") if d["outcome"] in ("hold", "reject") else d["outcome"]
        key = key or d["outcome"]
        counts[key] = counts.get(key, 0) + 1
    return counts


def _changed_segment_truth_points(base_segments, perturbed_segments, ground_truth: dict) -> dict:
    """output_index -> (2,) plan-frame reference point for every entry of
    `ground_truth['changed_output_indices']` (see `perturb_wall_segments`'s docstring):
      * a SHIFTED original segment -> its ORIGINAL (pre-shift) midpoint. The recon walk
        is generated from the TRUE, unperturbed plan (see `run_d2_gate`), so that
        original location -- not the drawing's new, wrong one -- is where a mismatch
        would actually be OBSERVED by the walk.
      * a NOISE segment (no original; `output_to_original_index[oi] is None`) -> its
        own (perturbed) midpoint -- the injected clutter wall has no other position to
        be judged against.
    """
    base = np.asarray(base_segments, dtype=np.float64).reshape(-1, 2, 2)
    out = {}
    o2o = ground_truth["output_to_original_index"]
    for oi in ground_truth["changed_output_indices"]:
        orig = o2o[oi]
        seg = base[orig] if orig is not None else np.asarray(perturbed_segments[oi], dtype=np.float64)
        out[oi] = 0.5 * (seg[0] + seg[1])
    return out


def _outlier_plan_positions(candidate: dict, rev_by_index: dict, apply_transform) -> list:
    """Every OUTLIER of one candidate, projected into plan metres through that SAME
    candidate's OWN transform (`plan_skeleton.apply_candidate_transform`, passed in as
    `apply_transform` so this function stays import-free). This is the ONLY transform
    ever available for a non-'ok' result (`best` is None on hold/reject by design) --
    which is why recall (below) is restricted to cases whose transform was ALREADY
    confirmed correct: projecting through a WRONG (e.g. mirrored) transform would place
    the outlier at a meaningless point."""
    tf = candidate["transform"]
    out = []
    for o in candidate["outliers"]:
        e = rev_by_index.get(int(o["event_index"]))
        if e is None:
            continue
        q = 0.5 * (np.asarray(e["a"]) + np.asarray(e["b"])) if e["kind"] == "leg" else np.asarray(e["xy"])
        out.append(apply_transform(tf, [q])[0])
    return out


def _walk_distance(point, plan_walk) -> float:
    """Distance (metres, plan frame) from `point` to the nearest sampled pose of the
    TRUE walk polyline -- the OBSERVABILITY test for D2 item (2), Cycle 7: a
    walk-vs-plan mismatch mechanism can only report a change near where it actually
    walked. `plan_walk` is the densely (0.15 m step) sampled TRUE plan-frame walk from
    `tests/test_coarse_match._make_walk` -- known exactly in this synthetic harness, so
    this needs no candidate transform (unlike `_outlier_plan_positions`)."""
    d = np.asarray(plan_walk, dtype=np.float64) - np.asarray(point, dtype=np.float64)
    return float(np.min(np.linalg.norm(d, axis=1)))


def _lateral_shift_component(base_segments, orig_index: int, shift_vector) -> float:
    """The component of one SHIFT vector PERPENDICULAR to its wall's own direction,
    metres -- the part of a shift an association tolerance (`leg_lat`,
    `scan2bim/coarse_match.py` TOL_FLOOR/TOL_PER_WIDTH) can actually see. Motion
    PARALLEL to the wall does not move its offset line at all
    (`plan_skeleton._wall_lines` clusters by perpendicular offset) and is invisible to
    `leg_lat` by construction; measuring the RAW shift magnitude against `leg_lat`
    would over-count what a perfect matcher could detect. This is the basis for
    `classify_changed_segments`'s `not_scoreable` bucket (Cycle 7 item (B), case#8)."""
    seg = np.asarray(base_segments[orig_index], dtype=np.float64)
    d = seg[1] - seg[0]
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return float(np.linalg.norm(np.asarray(shift_vector, dtype=np.float64)))
    u = d / L
    nrm = np.array([-u[1], u[0]])
    return float(abs(np.dot(np.asarray(shift_vector, dtype=np.float64), nrm)))


def classify_changed_segments(cases: list, base_segments, plan_walk,
                              radius: float = OUTLIER_RECALL_RADIUS_M,
                              fallback_leg_lat_tol=None) -> dict:
    """Cycle 7 item (B): per-changed-segment classification, the HONEST denominator for
    D2 item (2), computed over EVERY case (regardless of match status -- so the counts
    below are never silently case-gated away):

      not_observable -- the changed segment's TRUE location (`_changed_segment_truth_points`)
                        is further than `radius` from the TRUE walk path
                        (`_walk_distance`). A walk-vs-plan mismatch mechanism cannot see
                        this BY CONSTRUCTION, no matter how good the matcher is (measured
                        cycle-6 case: a noise wall 21.8 m from the walk). Applies to both
                        shifted and noise items.
      not_scoreable  -- observable, but a SHIFT (not noise -- see
                        `_lateral_shift_component`'s docstring for why noise has no
                        equivalent test) whose LATERAL component is <= the matcher's OWN
                        reported association tolerance for THAT case
                        (`result['info']['tol']['leg_lat']`, falling back to
                        `fallback_leg_lat_tol` if that case's result never got far enough
                        to report one, e.g. 'no_plan_skeleton'). Physically
                        indistinguishable from noise even to a perfect matcher (measured
                        cycle-6 case#8: inlier_ratio==1.0, a 0.3-1.0 m shift band dipping
                        under leg_lat~0.55 m). NEVER applied just because a case happened
                        to score well this run -- the test is the absolute lateral
                        magnitude vs. the matcher's declared tolerance, independent of
                        this run's outcome, so it cannot be used to paper over a genuine
                        miss.
      scoreable      -- everything else: observable, and (noise, or a shift whose
                        lateral component EXCEEDS the tolerance) -- a real,
                        in-principle-detectable mismatch. This is D2 item (2)'s
                        denominator (further restricted by `compute_outlier_recall` to
                        cases whose transform was independently confirmed correct).

    Returns {"items": [...], "n_total", "n_not_observable", "n_not_scoreable",
    "n_scoreable"}; every item also carries enough detail (case_index, output_index,
    dist_to_walk_m, lateral_shift_m, leg_lat_tol_m) to audit the classification, never
    just a bucket name."""
    items = []
    for c in cases:
        gt = c["ground_truth"]
        changed = gt["changed_output_indices"]
        if not changed:
            continue
        res = c.get("result") or {}
        leg_lat_tol = ((res.get("info", {}) or {}).get("tol", {}) or {}).get("leg_lat")
        if leg_lat_tol is None:
            leg_lat_tol = fallback_leg_lat_tol
        truth_pts = _changed_segment_truth_points(base_segments, c["perturbed_segments"], gt)
        o2o = gt["output_to_original_index"]
        shift_vectors = gt["shift_vectors"]
        for oi, p in truth_pts.items():
            dist_walk = _walk_distance(p, plan_walk)
            observable = dist_walk <= radius
            orig = o2o[oi]
            entry = {"case_index": c["index"], "output_index": int(oi),
                     "kind": "noise" if orig is None else "shift",
                     "dist_to_walk_m": round(dist_walk, 3), "observable": bool(observable)}
            if not observable:
                entry["category"] = "not_observable"
            elif orig is not None:
                lateral = _lateral_shift_component(base_segments, orig, shift_vectors[str(orig)])
                entry["lateral_shift_m"] = round(lateral, 4)
                entry["leg_lat_tol_m"] = None if leg_lat_tol is None else round(float(leg_lat_tol), 4)
                entry["category"] = ("not_scoreable" if (leg_lat_tol is not None and lateral <= leg_lat_tol)
                                     else "scoreable")
            else:
                entry["category"] = "scoreable"
            items.append(entry)
    n_not_observable = sum(1 for e in items if e["category"] == "not_observable")
    n_not_scoreable = sum(1 for e in items if e["category"] == "not_scoreable")
    n_scoreable = sum(1 for e in items if e["category"] == "scoreable")
    return {"items": items, "n_total": len(items), "n_not_observable": n_not_observable,
            "n_not_scoreable": n_not_scoreable, "n_scoreable": n_scoreable}


def compute_outlier_recall(cases: list, base_segments, classification: dict, rev_by_index: dict,
                           apply_transform, transform_confirmed_indices,
                           radius: float = OUTLIER_RECALL_RADIUS_M) -> dict:
    """D2 item (2), Cycle 7 definition: recall over `classification`'s `scoreable`
    items (see `classify_changed_segments`) that ALSO belong to a case whose transform
    `compute_success_rate` independently confirmed correct (`transform_confirmed_indices`
    -- projecting an outlier through an unverified/wrong transform would be meaningless,
    unchanged reasoning from cycle 6). A scoreable item counts as DETECTED if ANY
    outlier of that case's TOP-RANKED candidate (`candidates[0]`) projects within
    `radius` metres of the item's truth point.

    Every excluded item is counted, never silently dropped:
      n_not_observable / n_not_scoreable       -- from `classification`, denominator-level
                                                   exclusions (Cycle 7 items (B))
      n_excluded_case_no_confirmed_transform   -- scoreable items whose CASE's transform
                                                   was not confirmed (hold/reject/
                                                   ok-but-outside-D2) -- unchanged cycle-6
                                                   reasoning, renamed from the old
                                                   'n_excluded_not_scoreable' (that name is
                                                   now Cycle 7's PER-ITEM bucket above;
                                                   keep them distinct)
      n_case_no_candidate                      -- defensive: a confirmed-'ok' case with an
                                                   empty candidate list is a contract
                                                   violation (status=='ok' implies
                                                   candidates[0]==best), expected to stay 0
    """
    by_case: dict = {}
    for e in classification["items"]:
        if e["category"] != "scoreable":
            continue
        by_case.setdefault(e["case_index"], []).append(e)

    cases_by_index = {c["index"]: c for c in cases}
    detected = total = 0
    n_excluded_case_no_confirmed_transform = 0
    n_case_no_candidate = 0
    per_case = []
    for ci, entries in sorted(by_case.items()):
        if ci not in transform_confirmed_indices:
            n_excluded_case_no_confirmed_transform += len(entries)
            continue
        c = cases_by_index[ci]
        gt = c["ground_truth"]
        res = c.get("result") or {}
        cands = res.get("candidates") or []
        if not cands:
            n_case_no_candidate += len(entries)
            continue
        truth_pts = _changed_segment_truth_points(base_segments, c["perturbed_segments"], gt)
        outlier_pts = _outlier_plan_positions(cands[0], rev_by_index, apply_transform)
        n_det = 0
        misses = []
        for e in entries:
            oi = e["output_index"]
            p = truth_pts[oi]
            dmin = (min(float(np.linalg.norm(np.asarray(p) - op)) for op in outlier_pts)
                    if outlier_pts else float("inf"))
            hit = dmin <= radius
            n_det += int(hit)
            if not hit:
                misses.append({"output_index": int(oi), "kind": e["kind"],
                              "nearest_outlier_dist_m": None if not np.isfinite(dmin) else round(dmin, 3)})
        detected += n_det
        total += len(entries)
        per_case.append({"index": ci, "n_scoreable": len(entries), "n_detected": n_det, "misses": misses})

    recall = (detected / total) if total else None
    return {"recall": recall, "n_detected": detected, "n_total": total,
            "n_not_observable": classification["n_not_observable"],
            "n_not_scoreable": classification["n_not_scoreable"],
            "n_excluded_case_no_confirmed_transform": n_excluded_case_no_confirmed_transform,
            "n_case_no_candidate": n_case_no_candidate,
            "n_all_changed_segments": classification["n_total"],
            "radius_m": radius, "per_case": per_case}


def _span_ground_truth_positive_set(case: dict, base_segments, spans: list, apply_transform,
                                    radius: float = OUTLIER_RECALL_RADIUS_M) -> set:
    """CYCLE 19. Span `event_index` set that is a GROUND-TRUTH positive for ONE case: a
    span whose position (projected into plan metres through THIS case's OWN confirmed
    `best` transform) lands within `radius` of ANY truly-changed segment's truth point
    (`_changed_segment_truth_points` -- every `changed_output_indices` entry, shift OR
    noise, regardless of the leg-association `not_observable`/`not_scoreable` buckets --
    those are LEG-tolerance-specific classifications (Cycle 7), not applicable to the
    span scan's own, different detection mechanism (`corridor_width_mismatch` sees the
    FULL shift magnitude, not just the component lateral to the wall's own line -- see
    scan2bim/coarse_match.py's span-change-scan block comment). Reuses
    `OUTLIER_RECALL_RADIUS_M` (already the file's justified "same regional change"
    tolerance, Cycle 7) rather than inventing a second radius constant.

    `spans` is the FIXED span-event list (same across every case -- they are generated
    from the one TRUE walk, see `_build_d2_harness_context`'s `spans_all`). A case with
    no confirmed `best` transform, or no changed segments at all, returns the empty set
    (nothing to be positive about / no transform to project through)."""
    res = case.get("result") or {}
    best = res.get("best")
    if not best:
        return set()
    gt = case["ground_truth"]
    if not gt["changed_output_indices"]:
        return set()
    tf = best["transform"]
    truth_pts = list(_changed_segment_truth_points(base_segments, case["perturbed_segments"], gt).values())
    if not truth_pts:
        return set()
    positive = set()
    for e in spans:
        p = apply_transform(tf, [e["xy"]])[0]
        if any(float(np.linalg.norm(np.asarray(p) - np.asarray(t))) <= radius for t in truth_pts):
            positive.add(int(e["index"]))
    return positive


def compute_span_precision(cases: list, base_segments, spans: list, transform_confirmed_indices,
                           apply_transform, radius: float = OUTLIER_RECALL_RADIUS_M) -> dict:
    """CYCLE 19, the metric that PAIRS with item② (recall) to close the flood loophole
    dev-core self-reported (commit 8d7fcbf): recall alone cannot distinguish "detected the
    real change" from "flagged everything", because flagging every span trivially recalls
    100% of it. This computes a per-SPAN (not per-truth-item, unlike `compute_outlier_
    recall`) confusion matrix, aggregated over every case in `transform_confirmed_indices`
    (SAME restriction `compute_outlier_recall` uses -- a case whose transform was never
    confirmed correct has no meaningful projection to test against):

      TP -- span WAS flagged (`kind=='span'` in the top candidate's outliers) AND is a
            ground-truth positive (`_span_ground_truth_positive_set`)
      FP -- flagged but NOT a ground-truth positive (a genuine false alarm)
      FN -- a ground-truth positive that was NOT flagged
      TN -- neither

    `precision = TP/(TP+FP)`, `recall_span = TP/(TP+FN)` (a SEPARATE number from item②'s
    recall -- that one is per-truth-changed-segment across every outlier kind; this one is
    per-span, span-kind only; do not conflate the two when reading the CLI output),
    `f1` their harmonic mean, `prevalence = (TP+FN)/n` (the ground-truth positive
    fraction -- "실제 변경분이 차지하는 span 비율", CYCLE 19's physical reference for the
    flood gate, computed FRESH from this run's own ground truth, never hardcoded),
    `fired_frac = (TP+FP)/n` (how much of the walk the matcher actually flagged, for the
    report). `n == 0` (no confirmed case, or `spans` empty) leaves every ratio `None` --
    never a fabricated number."""
    cases_by_index = {c["index"]: c for c in cases}
    tp = fp = fn = tn = 0
    n_cases_included = 0
    per_case = []
    for ci in sorted(transform_confirmed_indices):
        c = cases_by_index.get(ci)
        if c is None:
            continue
        res = c.get("result") or {}
        cands = res.get("candidates") or []
        if not cands:
            continue
        n_cases_included += 1
        fired = {int(o["event_index"]) for o in cands[0].get("outliers", []) if o.get("kind") == "span"}
        positive = _span_ground_truth_positive_set(c, base_segments, spans, apply_transform, radius)
        c_tp = c_fp = c_fn = c_tn = 0
        for e in spans:
            idx = int(e["index"])
            is_fired, is_pos = idx in fired, idx in positive
            if is_fired and is_pos:
                c_tp += 1
            elif is_fired:
                c_fp += 1
            elif is_pos:
                c_fn += 1
            else:
                c_tn += 1
        tp += c_tp; fp += c_fp; fn += c_fn; tn += c_tn
        per_case.append({"index": ci, "tp": c_tp, "fp": c_fp, "fn": c_fn, "tn": c_tn})
    n = tp + fp + fn + tn
    prevalence = ((tp + fn) / n) if n else None
    precision = (tp / (tp + fp)) if (tp + fp) else None
    recall_span = (tp / (tp + fn)) if (tp + fn) else None
    f1 = (2 * precision * recall_span / (precision + recall_span)
          if (precision is not None and recall_span is not None and (precision + recall_span) > 0) else None)
    fired_frac = ((tp + fp) / n) if n else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": n, "n_cases_included": n_cases_included,
            "prevalence": prevalence, "precision": precision, "recall_span": recall_span, "f1": f1,
            "fired_frac": fired_frac, "radius_m": radius, "per_case": per_case}


def span_precision_gate_ok(span_stats: dict,
                           factor: float = GATE_SPAN_PRECISION_PREVALENCE_FACTOR) -> bool:
    """CYCLE 19 item⑤ verdict from `compute_span_precision`'s output. Edge cases, all
    documented rather than silently defaulted:
      n==0            -- no confirmed-transform case had any span to score (typically
                         because item① already has 0 confirmed cases) -- NOT a flood
                         signal either way; item① already fails that scenario on its own,
                         so this returns True (non-blocking) rather than double-counting.
      tp+fp==0        -- nothing was EVER flagged across the whole suite -- trivially
                         cannot be flooding (a flood fires on everything); True.
      prevalence<=0   -- no real change was ever a ground-truth positive across the whole
                         suite (rare -- e.g. every changed segment landed not_observable);
                         with nothing to be right about, ANY firing is a straight false
                         positive, so the bar becomes fp==0.
      else            -- the CYCLE 19 physical test: precision must clear `factor` times
                         the suite's own measured prevalence (see the CYCLE 19 constants
                         block for why flooding is mathematically capped AT prevalence)."""
    if span_stats["n"] == 0:
        return True
    if span_stats["tp"] + span_stats["fp"] == 0:
        return True
    prevalence = span_stats["prevalence"] or 0.0
    if prevalence <= 0.0:
        return span_stats["fp"] == 0
    return span_stats["precision"] >= factor * prevalence


def verify_hold_on_ambiguous(result: dict) -> bool:
    """CLIENT-side check that a TIE never gets auto-confirmed. If the top-2 candidate
    scores were within `result['gates']['margin_min']` (the threshold `finalize_match`
    ACTUALLY applied to THIS result -- read from the result itself, so a
    `coarse_match(gates=...)` override is still checked correctly, never re-derived
    from a module-level default that might not match), the result must NOT be
    status=='ok' (best must stay None).

    NOTE, measured cycle 6: the invariant is "!= 'ok'", NOT "== 'hold'". Per
    `plan_skeleton.finalize_match`'s OWN documented gate order ("hard gates before the
    tie test"), a case whose best candidate ALSO fails inlier_ratio/residual is
    reported REJECT even when its margin is simultaneously < margin_min -- REJECT is
    just as much a refusal-to-auto-confirm as HOLD is. This function/invariant is
    UNCHANGED this cycle (item (3) of the gate is out of scope for this cycle's
    changes, per instruction). A single-candidate result (margin is None) is
    unambiguous by construction and always passes."""
    margin = result.get("margin")
    margin_min = float(result.get("gates", {}).get("margin_min", _FALLBACK_MARGIN_MIN))
    if margin is not None and margin < margin_min:
        return result.get("status") != "ok" and result.get("best") is None
    return True


def _load_d2_harness():
    """Import the dev-core synthetic spine->wall->skeleton harness from
    tests/test_coarse_match.py (+ tests/test_plan_skeleton.py, for the 2-leg L
    ambiguous fixture), reused rather than built new (cycle-6 instruction, still
    honoured). Returns (coarse_match, plan_skeleton, test_coarse_match,
    test_plan_skeleton) modules. Raises ImportError (propagated, never swallowed) if
    the matcher or the tests/ harness is unavailable -- the caller turns that into
    exit 2, never a fabricated pass."""
    from scan2bim import coarse_match as cm
    from scan2bim import plan_skeleton as ps
    tests_dir = _REPO / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    import test_coarse_match as tcm
    import test_plan_skeleton as tps
    return cm, ps, tcm, tps


def build_ambiguous_fixture(cm, ps, tcm, tps) -> dict:
    """The DETERMINISTIC known-tie case for D2 item (3), reused (not re-derived) from
    tests/test_coarse_match.py's TestAmbiguityHolds.
    test_two_leg_L_is_ambiguous_by_construction: a 2-leg L corridor with NO door,
    walked corner to corner -- coarse_match.py's own documented load-bearing claim is
    that with (s_f, s_h) both free this ties EXACTLY (margin 0.0) between the forward
    fit and the chi-flipped mirror fit. Independent of the seeded perturbation suite
    (n_perturb, seed): a FIXED regression probe that the HOLD mechanism actually FIRES,
    not merely that it was never exercised by the (generally unambiguous, see the
    cycle report) 20 perturbation draws. UNCHANGED this cycle."""
    skel = ps.corridor_skeleton(tps._l_corridor_segments())
    walk = [(tps.CENTER_A, 0.0), (tps.CENTER_A, tps.CENTER_B), (10.0, tps.CENTER_B)]
    _, traj, _, _ = tcm._make_walk(walk)
    res = cm.coarse_match(skel, traj)
    return {"skel_n_legs": len(skel["legs"]), "traj_n": len(traj), "result": res}


def verify_hold_gate(cases: list, ambiguous_fixture: dict) -> dict:
    """D2 item (3): (a) the deterministic known-tie fixture (`build_ambiguous_fixture`)
    must actually come back hold/'ambiguous_margin'/best=None -- proof the mechanism
    fires, not just that it was never exercised; (b) EVERY one of the N perturbation
    results independently satisfies `verify_hold_on_ambiguous` -- a REGRESSION check,
    NOT proof the 20-draw suite itself ever produces a genuine tie. UNCHANGED this
    cycle (out of scope per instruction)."""
    res = ambiguous_fixture["result"]
    gates = res.get("gates", {})
    margin_min = float(gates.get("margin_min", _FALLBACK_MARGIN_MIN))
    fixture_ok = (res.get("status") == "hold" and res.get("hold_reason") == "ambiguous_margin"
                  and res.get("best") is None and res.get("margin") is not None
                  and float(res["margin"]) < margin_min)
    violations = []
    for c in cases:
        r = c.get("result")
        if r is None:
            continue
        if not verify_hold_on_ambiguous(r):
            violations.append(c["index"])
    return {"fixture_ok": bool(fixture_ok), "fixture_status": res.get("status"),
            "fixture_hold_reason": res.get("hold_reason"), "fixture_margin": res.get("margin"),
            "perturbation_invariant_violations": violations,
            "ok": bool(fixture_ok and not violations)}


def _build_d2_harness_context() -> dict:
    """CYCLE 12: harness build factored out of `_run_d2_gate_single` so `--sweep` (many
    (ratio, seed) points) can reuse ONE build instead of repeating it per point -- pure
    performance refactor, ZERO behaviour change: this is EXACTLY the sequence
    `_run_d2_gate_single` ran inline before this cycle (still runs, unchanged, for every
    call the default D2 gate path makes -- see `_run_d2_gate_single` below, now a thin
    wrapper around this + `_evaluate_suite`). Builds the dev-core STAIR harness (4 legs /
    3 corners / 5 doors, unambiguous), DENSIFIES its wall segments with wall-id tracking
    (Cycle 7 item (A) + Cycle 11's `wall_ids` -- verified as a no-op on the clean
    skeleton and on single-fragment removal BEFORE anything is perturbed), and confirms
    the UNPERTURBED plan recovers its own known-true transform exactly. Deterministic --
    no seed argument; none of this depends on which perturbation seed will later be
    evaluated against it (`assert_single_fragment_removal_is_graceful`'s OWN internal
    probe seed defaults to 0, unrelated to any outer `--seed`, unchanged from before this
    cycle).

    Raises ImportError (matcher/tests-harness missing) or RuntimeError (the density pass
    changed the skeleton it should have left untouched, the single-fragment-removal
    regression probe found a topology collapse, or the harness's OWN zero-perturbation
    baseline was not recovered exactly) -- same conditions, same exceptions, as
    `_run_d2_gate_single` raised inline before this cycle; the caller (`main` /
    `run_sweep`) turns either into exit 2, never a fabricated verdict."""
    cm, ps, tcm, tps = _load_d2_harness()

    raw_segments = tcm._corridor_walls(tcm.STAIR)                        # (10,2,2) m -- dev-core's own fixture
    plan_doors = list(tcm.STAIR_PLAN_DOORS)
    base_segments, wall_ids = densify_wall_segments_with_ids(raw_segments)   # Cycle 7 item (A) + Cycle 11 tags

    density_check = assert_densify_preserves_skeleton(ps.corridor_skeleton, raw_segments,
                                                      base_segments, plan_doors)
    if not density_check["ok"]:
        raise RuntimeError(
            "P0-Perturb 밀도 보정 실패: densify_wall_segments 가 무섭동 상태에서 스켈레톤을 "
            f"바꿔서는 안 되는데 바꿨습니다 -- {density_check}")

    single_removal_check = assert_single_fragment_removal_is_graceful(
        ps.corridor_skeleton, base_segments, plan_doors,
        true_n_legs=density_check["raw_n_legs"], true_n_corners=density_check["raw_n_corners"])
    if not single_removal_check["ok"]:
        raise RuntimeError(
            "P0-Perturb 단일 조각 제거 회귀 테스트 실패: 조각 1개 제거만으로 legs/corners 위상이 "
            f"무너집니다(이 사이클이 고치려던 바로 그 버그) -- {single_removal_check}")

    tf_true, traj, plan_walk, door_s = tcm._make_walk(tcm.STAIR, tcm.STAIR_WALK_DOORS)
    offset_max = GATE_OFFSET_WIDTH_FRAC * float(tcm.W)

    clean_skel = ps.corridor_skeleton(base_segments, doors=plan_doors)
    clean_res = cm.coarse_match(clean_skel, traj, door_s=door_s)
    if clean_res.get("status") == "ok":
        clean_ok, clean_detail = _transform_within_d2(clean_res["best"]["transform"], tf_true,
                                                       offset_max=offset_max)
    else:
        clean_ok, clean_detail = False, {}
    if not clean_ok:
        raise RuntimeError(
            "D2 harness sanity check failed: the UNPERTURBED (densified) STAIR plan did not "
            f"recover the known true transform exactly (status={clean_res.get('status')!r} "
            f"hold_reason={clean_res.get('hold_reason')!r} detail={clean_detail}) -- "
            "refusing to evaluate perturbed cases against a harness that does not even "
            "pass its own zero-perturbation baseline")

    rev_list = cm.recon_events(traj, door_s=door_s)
    rev_by_index = {e["index"]: e for e in rev_list}
    # CYCLE 19 item④: the SAME clean/unperturbed match already computed above
    # (`clean_res`) carries its OWN span outliers -- the span scan runs on every match,
    # perturbed or not (scan2bim/coarse_match.py's `_cand` always appends
    # `_span_outliers(...)`). Zero real changes exist on this plan by construction, so
    # ANY span outlier on this candidate is a genuine false alarm -- see the CYCLE 19
    # constants block (above `GATE_SPAN_CLEAN_FPR_MAX`) for why the gate target is
    # exactly 0, not a tolerance band.
    spans_all = [e for e in rev_list if e["kind"] == "span"]
    n_spans_total = len(spans_all)
    clean_cands = clean_res.get("candidates") or []
    clean_span_fired = (sum(1 for o in clean_cands[0].get("outliers", []) if o.get("kind") == "span")
                        if clean_cands else 0)
    clean_span_fpr = (clean_span_fired / n_spans_total) if n_spans_total else None
    fallback_leg_lat_tol = max(float(cm.TOL_FLOOR["leg_lat"]),
                               float(cm.TOL_PER_WIDTH["leg_lat"]) * float(tcm.W))

    return {"cm": cm, "ps": ps, "tcm": tcm, "tps": tps,
            "raw_segments": raw_segments, "plan_doors": plan_doors,
            "base_segments": base_segments, "wall_ids": wall_ids,
            "density_check": density_check, "single_removal_check": single_removal_check,
            "tf_true": tf_true, "traj": traj, "plan_walk": plan_walk, "door_s": door_s,
            "offset_max": offset_max,
            "clean_baseline": {"status": clean_res.get("status"), "within_d2": clean_ok,
                               "detail": clean_detail},
            "rev_by_index": rev_by_index, "fallback_leg_lat_tol": fallback_leg_lat_tol,
            "spans_all": spans_all, "n_spans_total": n_spans_total,
            "clean_span_fired": clean_span_fired, "clean_span_fpr": clean_span_fpr}


def _evaluate_suite(ctx: dict, suite: list) -> dict:
    """CYCLE 12: the per-suite matching + scoring tail of `_run_d2_gate_single`, factored
    out so `--sweep` can call it once per (ratio, seed) point against the ONE shared
    `ctx` (`_build_d2_harness_context`). UNCHANGED math/logic from what
    `_run_d2_gate_single` ran inline before this cycle -- matches every case in `suite`
    against `ctx`'s fixed walk, then runs the SAME `compute_success_rate` /
    `failure_reason_breakdown` / `classify_changed_segments` / `compute_outlier_recall` /
    `build_ambiguous_fixture` / `verify_hold_gate` sequence. Returns the tail half of
    `_run_d2_gate_single`'s result dict (`case_summaries`, `contract_violations`,
    `success`, `failure_reasons`, `outlier_recall`, `hold`,
    `gate1_ok`/`gate2_ok`/`gate3_ok`/`gate_ok`) -- `_run_d2_gate_single` merges this with
    `ctx`'s harness-level fields to reproduce its EXACT pre-cycle-12 return shape."""
    ps, cm = ctx["ps"], ctx["cm"]
    plan_doors, traj, door_s = ctx["plan_doors"], ctx["traj"], ctx["door_s"]

    contract_violations = []
    cases = []
    for item in suite:
        segs = item["segments"]
        res, err = None, None
        try:
            skel = ps.corridor_skeleton(segs, doors=plan_doors)
            res = cm.coarse_match(skel, traj, door_s=door_s)
            ok, errs = ps.validate_match_result(res)
            if not ok:
                contract_violations.append({"index": item["index"], "errors": errs})
        except Exception as e:                        # a perturbation crashing the matcher is DATA
            err = f"{type(e).__name__}: {e}"
        cases.append({"index": item["index"], "child_seed": item["child_seed"],
                      "perturbed_segments": segs, "ground_truth": item["ground_truth"],
                      "result": res, "error": err})

    succ = compute_success_rate(cases, ctx["tf_true"], ctx["offset_max"])
    transform_confirmed_indices = {d["index"] for d in succ["detail"] if d["outcome"] == "ok_within_d2"}
    failure_reasons = failure_reason_breakdown(succ)

    classification = classify_changed_segments(cases, ctx["base_segments"], ctx["plan_walk"],
                                               fallback_leg_lat_tol=ctx["fallback_leg_lat_tol"])
    recall = compute_outlier_recall(cases, ctx["base_segments"], classification, ctx["rev_by_index"],
                                    ps.apply_candidate_transform, transform_confirmed_indices)
    span_stats = compute_span_precision(cases, ctx["base_segments"], ctx["spans_all"],
                                        transform_confirmed_indices, ps.apply_candidate_transform)

    ambiguous = build_ambiguous_fixture(cm, ps, ctx["tcm"], ctx["tps"])
    hold = verify_hold_gate(cases, ambiguous)

    case_summaries = []
    for c in cases:
        r = c.get("result")
        p = c["ground_truth"]["params"]
        cands_ = ((r or {}).get("candidates") or [])
        best = cands_[0] if cands_ else None
        case_summaries.append({
            "index": c["index"], "n_removed": p["n_removed"], "n_shifted": p["n_shifted"],
            "n_noise": p["n_noise"], "error": c.get("error"),
            "status": (r or {}).get("status"), "hold_reason": (r or {}).get("hold_reason"),
            "margin": (r or {}).get("margin"),
            "n_candidates": len(((r or {}).get("candidates")) or []),
            "top_score": (best or {}).get("score"), "top_inlier_ratio": (best or {}).get("inlier_ratio"),
        })

    gate1_ok = succ["rate"] >= GATE_SUCCESS_RATE_MIN
    gate2_ok = recall["recall"] is not None and recall["recall"] >= GATE_OUTLIER_RECALL_MIN
    gate3_ok = hold["ok"]
    # CYCLE 19 items ④/⑤ -- see the constants block above `GATE_SPAN_CLEAN_FPR_MAX`. ④
    # reads the ctx-level (suite-independent) clean-plan measurement straight through;
    # ⑤ is THIS suite's own span_stats (computed just above, from THIS suite's cases).
    gate4_ok = ctx["clean_span_fpr"] is None or ctx["clean_span_fpr"] <= GATE_SPAN_CLEAN_FPR_MAX
    gate5_ok = span_precision_gate_ok(span_stats)
    return {"cases": cases, "case_summaries": case_summaries, "contract_violations": contract_violations,
            "success": succ, "failure_reasons": failure_reasons,
            "outlier_recall": recall, "hold": hold,
            "span_precision": span_stats,
            "clean_span_fired": ctx["clean_span_fired"], "clean_span_fpr": ctx["clean_span_fpr"],
            "n_spans_total": ctx["n_spans_total"],
            "gate1_ok": gate1_ok, "gate2_ok": gate2_ok, "gate3_ok": gate3_ok,
            "gate4_ok": gate4_ok, "gate5_ok": gate5_ok,
            "gate_ok": bool(gate1_ok and gate2_ok and gate3_ok and gate4_ok and gate5_ok)}


def _run_d2_gate_single(seed: int, n_perturb: int, perturb_mode: str) -> dict:
    """Runs the full D2 gate for ONE perturbation model: builds the dev-core STAIR
    harness (`_build_d2_harness_context`), perturbs the densified WALL SEGMENTS ONLY
    `n_perturb` times using `perturb_mode` (`'structured'` or `'fragment'`, QA's seeded
    generators, Section 2 / 2b -- doors are held fixed, see the module docstring),
    matches each perturbed plan against the SAME walk, and evaluates all three D2 items
    (`_evaluate_suite`). CYCLE 12: now a thin wrapper -- harness build and per-case
    scoring both moved to `_build_d2_harness_context` / `_evaluate_suite` so `--sweep`
    can reuse them; this function's OWN return shape and every value in it are BYTE-
    IDENTICAL to before this cycle for the same (seed, n_perturb, perturb_mode) -- see
    the `--n-perturb 20 --seed 7` no-regression check, cycle report.

    Raises ImportError / RuntimeError exactly as `_build_d2_harness_context` does -- the
    caller (`main`) turns either into exit 2, never a fabricated verdict."""
    if perturb_mode not in ("structured", "fragment"):
        raise ValueError(f"_run_d2_gate_single: perturb_mode must be 'structured' or "
                         f"'fragment', got {perturb_mode!r}")
    ctx = _build_d2_harness_context()

    if perturb_mode == "fragment":
        suite = generate_perturbation_suite(ctx["base_segments"], seed, n_perturb, mode="fragment")
    else:
        suite = generate_perturbation_suite(ctx["base_segments"], seed, n_perturb, mode="structured",
                                            wall_ids=ctx["wall_ids"])
    ev = _evaluate_suite(ctx, suite)

    return {"n_perturb": n_perturb, "seed": seed, "perturb_mode": perturb_mode,
            "density_check": ctx["density_check"],
            "single_removal_check": ctx["single_removal_check"],
            "clean_baseline": ctx["clean_baseline"],
            "case_summaries": ev["case_summaries"],
            "contract_violations": ev["contract_violations"],
            "success": ev["success"], "failure_reasons": ev["failure_reasons"],
            "outlier_recall": ev["outlier_recall"], "hold": ev["hold"],
            "span_precision": ev["span_precision"],
            "clean_span_fired": ev["clean_span_fired"], "clean_span_fpr": ev["clean_span_fpr"],
            "n_spans_total": ev["n_spans_total"],
            "gate1_ok": ev["gate1_ok"], "gate2_ok": ev["gate2_ok"], "gate3_ok": ev["gate3_ok"],
            "gate4_ok": ev["gate4_ok"], "gate5_ok": ev["gate5_ok"],
            "gate_ok": ev["gate_ok"]}


def run_d2_gate(seed: int, n_perturb: int, perturb_mode: str = "structured") -> dict:
    """CYCLE 11 dispatcher. `perturb_mode in {'structured', 'fragment'}` runs ONE suite
    (`_run_d2_gate_single`) and returns its result dict directly (unchanged shape from
    before this cycle, plus the new `perturb_mode`/`failure_reasons` keys).
    `perturb_mode == 'both'` runs STRUCTURED AND FRAGMENT independently -- same seed,
    same n_perturb, but each mode's OWN `derive_seeds` stream (so the two suites are
    genuinely different draws under each model, not the same draws reinterpreted) --
    and returns `{"perturb_mode": "both", "structured": ..., "fragment": ...,
    "gate_ok": both}`, side by side, never merged into one set of numbers (see the
    module docstring's Cycle 11 section for why merging would hide which model the
    matcher is actually failing)."""
    if perturb_mode == "both":
        g_structured = _run_d2_gate_single(seed, n_perturb, "structured")
        g_fragment = _run_d2_gate_single(seed, n_perturb, "fragment")
        return {"perturb_mode": "both", "n_perturb": n_perturb, "seed": seed,
                "structured": g_structured, "fragment": g_fragment,
                "gate_ok": bool(g_structured["gate_ok"] and g_fragment["gate_ok"])}
    return _run_d2_gate_single(seed, n_perturb, perturb_mode)


def selftest_flood_detection(seed: int = 7, n_perturb: int = 20) -> dict:
    """CYCLE 19 -- THE regression test this cycle exists to add. Proves items ④/⑤
    (`GATE_SPAN_CLEAN_FPR_MAX` / `GATE_SPAN_PRECISION_PREVALENCE_FACTOR`) actually reject
    the exact failure mode dev-core self-reported in commit 8d7fcbf: "flag every span,
    recall goes to 100%". Monkeypatches `scan2bim.coarse_match._span_outliers` (in
    memory, restored in a `finally` -- scan2bim/** is never written to disk, this team's
    read-only rule on it is honoured) to a fake that unconditionally returns EVERY span
    of the walk as an outlier, no geometry check at all -- the simplest possible flood
    implementation -- then runs the SAME harness build + suite evaluation the real D2
    gate runs, and returns whether ④ or ⑤ (or both) caught it.

    Two things are measured, both under the SAME patch:
      1. The clean (UNPERTURBED) baseline -- rebuilt fresh under the patch, so its own
         span scan is flooded too. Zero real changes exist on this plan, so a flood
         reports EVERY span as a false alarm: this alone should fail ④ outright.
      2. The `seed`/`n_perturb` perturbed suite (default: the same seed=7, n=20 the D2
         gate's own DoD command uses) -- item⑤'s precision collapses toward the suite's
         own prevalence (a flood cannot discriminate real change from unchanged plan by
         construction, see the CYCLE 19 constants block), which the 1.5x-prevalence bar
         is designed to catch regardless of how much of the suite is genuinely changed.

    Returns a dict with both measurements plus `caught` (bool: gate4_ok is False OR
    gate5_ok is False under the flood patch -- `False` here would mean the flood slipped
    through undetected, the exact defect this cycle's instruction asked to be tested
    for). Raises ImportError/RuntimeError exactly as `_build_d2_harness_context` does
    (propagated, not swallowed -- the CLI turns either into exit 2)."""
    cm, ps, tcm, tps = _load_d2_harness()
    original_span_outliers = cm._span_outliers

    def _flood_all_spans(tf, R, P, walls, width_ref):
        return [{"kind": "span", "event_index": int(e["index"]), "plan_event_id": None,
                "residual": 0.0, "reason": "FAKE_FLOOD_SELFTEST_ALL_SPANS_UNCONDITIONALLY"}
               for e in (R.get("span") or [])]

    try:
        cm._span_outliers = _flood_all_spans
        # Rebuilt UNDER the patch (not reusing any pre-patch ctx) so the clean baseline's
        # OWN span scan is flooded too -- proves the flood is caught even before any
        # perturbation exists, the sharpest possible demonstration.
        flooded_ctx = _build_d2_harness_context()
        suite = generate_perturbation_suite(flooded_ctx["base_segments"], seed, n_perturb,
                                            mode="structured", wall_ids=flooded_ctx["wall_ids"])
        ev = _evaluate_suite(flooded_ctx, suite)
    finally:
        cm._span_outliers = original_span_outliers

    return {
        "seed": int(seed), "n_perturb": int(n_perturb),
        "clean_span_fired": flooded_ctx["clean_span_fired"],
        "n_spans_total": flooded_ctx["n_spans_total"],
        "clean_span_fpr": flooded_ctx["clean_span_fpr"],
        "gate4_ok": ev["gate4_ok"],
        "span_precision": ev["span_precision"],
        "gate5_ok": ev["gate5_ok"],
        "gate1_ok": ev["gate1_ok"], "gate2_ok": ev["gate2_ok"], "gate3_ok": ev["gate3_ok"],
        "gate_ok": ev["gate_ok"],
        "caught": bool((not ev["gate4_ok"]) or (not ev["gate5_ok"])),
    }


def run_sweep(ratios=SWEEP_RATIOS_DEFAULT, seeds=SWEEP_SEEDS_DEFAULT,
             n_perturb: int = GATE_N_PERTURB_DEFAULT) -> dict:
    """CYCLE 12 (`--sweep`): the DEGRADATION CURVE -- success-rate / outlier-recall /
    HOLD-violation-count measured across the TOTAL-change-ratio axis (Section 2c),
    `structured` mode only (the sweep is about the site-change model this team's DoD
    targets, not drawing-noise -- see Section 2c / the module docstring's CYCLE 11
    block), at each of `ratios` x each of `seeds`. Reuses ONE `_build_d2_harness_context`
    across every point (deterministic, seed-independent -- see that function's own
    docstring) instead of rebuilding it `len(ratios)*len(seeds)` times.

    THIS DOES NOT REDEFINE OR RELAX THE D2 GATE. `GATE_SUCCESS_RATE_MIN` /
    `GATE_OUTLIER_RECALL_MIN` (90%/80%) are read UNCHANGED for the per-point
    gate1_ok/gate2_ok/gate3_ok flags reported alongside each point (so a reader can see
    exactly where the curve crosses the SAME thresholds `run_d2_gate` uses) -- but
    `run_sweep`'s OWN CLI exit-code contract (see `_run_sweep_cli`) is a MEASUREMENT, not
    a pass/fail gate: a low-ratio point failing is exactly as reportable a result as a
    high-ratio one failing, per this cycle's instruction to measure the whole curve,
    never to declare only one point in isolation.

    Returns {"ratios", "seeds", "n_perturb", "n_base_fragments", "alloc_weights",
    "clean_baseline", "points": [...]}; each point carries `per_seed` (one dict per seed
    -- achieved ratio, success rate, recall, ok_outside_d2 count, HOLD-violation count,
    failure-reason breakdown, gate1/2/3_ok) PLUS the seed-aggregated mean/min/max the
    report table reads (`success_rate_mean` etc.) -- both levels always present,
    per-seed detail never collapsed away."""
    ctx = _build_d2_harness_context()
    base_segments, wall_ids = ctx["base_segments"], ctx["wall_ids"]
    n_base = len(base_segments)

    points = []
    for ratio in ratios:
        per_seed = []
        for seed in seeds:
            suite = generate_perturbation_suite_at_total_ratio(base_segments, wall_ids, seed,
                                                                n_perturb, ratio)
            ev = _evaluate_suite(ctx, suite)
            achieved = []
            for item in suite:
                p = item["ground_truth"]["params"]
                achieved.append((p["n_removed"] + p["n_shifted"] + p["n_noise"]) / n_base)
            succ = ev["success"]
            recall = ev["outlier_recall"]
            per_seed.append({
                "seed": int(seed), "ratio_target": float(ratio),
                "achieved_ratio_mean": float(np.mean(achieved)),
                "achieved_ratio_min": float(np.min(achieved)),
                "achieved_ratio_max": float(np.max(achieved)),
                "n": succ["n"], "success_rate": succ["rate"],
                "n_ok_within_d2": succ["n_ok_within_d2"], "n_ok_outside_d2": succ["n_ok_outside_d2"],
                "n_hold": succ["n_hold"], "n_reject": succ["n_reject"], "n_error": succ["n_error"],
                "recall": recall["recall"], "recall_n_detected": recall["n_detected"],
                "recall_n_total": recall["n_total"],
                "hold_violations": len(ev["hold"]["perturbation_invariant_violations"]),
                "failure_reasons": ev["failure_reasons"],
                "span_precision": ev["span_precision"],
                "clean_span_fpr": ev["clean_span_fpr"],
                "gate1_ok": ev["gate1_ok"], "gate2_ok": ev["gate2_ok"], "gate3_ok": ev["gate3_ok"],
                "gate4_ok": ev["gate4_ok"], "gate5_ok": ev["gate5_ok"],
                "gate_ok": ev["gate_ok"],
            })

        rates = [s["success_rate"] for s in per_seed]
        recalls = [s["recall"] for s in per_seed if s["recall"] is not None]
        merged_failure_reasons: dict = {}
        for s in per_seed:
            for k, v in s["failure_reasons"].items():
                merged_failure_reasons[k] = merged_failure_reasons.get(k, 0) + v
        precisions = [s["span_precision"]["precision"] for s in per_seed
                     if s["span_precision"]["precision"] is not None]
        points.append({
            "ratio_target": float(ratio), "per_seed": per_seed,
            "achieved_ratio_mean": float(np.mean([s["achieved_ratio_mean"] for s in per_seed])),
            "success_rate_mean": float(np.mean(rates)), "success_rate_min": float(np.min(rates)),
            "success_rate_max": float(np.max(rates)),
            "recall_mean": (float(np.mean(recalls)) if recalls else None),
            "recall_min": (float(np.min(recalls)) if recalls else None),
            "recall_max": (float(np.max(recalls)) if recalls else None),
            "recall_n_seeds_scoreable": len(recalls), "recall_n_seeds_total": len(per_seed),
            "precision_mean": (float(np.mean(precisions)) if precisions else None),
            "precision_n_seeds_scoreable": len(precisions),
            "n_ok_outside_d2_total": sum(s["n_ok_outside_d2"] for s in per_seed),
            "hold_violations_total": sum(s["hold_violations"] for s in per_seed),
            "failure_reasons_merged": merged_failure_reasons,
            "all_seeds_gate1_ok": all(s["gate1_ok"] for s in per_seed),
            "all_seeds_gate2_ok": all(s["gate2_ok"] for s in per_seed),
            "all_seeds_gate3_ok": all(s["gate3_ok"] for s in per_seed),
            "all_seeds_gate4_ok": all(s["gate4_ok"] for s in per_seed),
            "all_seeds_gate5_ok": all(s["gate5_ok"] for s in per_seed),
        })

    return {"ratios": [float(r) for r in ratios], "seeds": [int(s) for s in seeds],
            "n_perturb": int(n_perturb), "n_base_fragments": n_base,
            "alloc_weights": {"w_remove": SWEEP_ALLOC_WEIGHTS[0], "w_shift": SWEEP_ALLOC_WEIGHTS[1],
                             "w_noise": SWEEP_ALLOC_WEIGHTS[2]},
            "clean_baseline": ctx["clean_baseline"], "points": points}


# ==========================================================================================
# SECTION 4 -- CLI
# ==========================================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upload", type=str, default=None,
                    help="path to a real recon upload/session (e.g. realtime/_uploads/upload_XXXX). "
                         "Accepted but UNUSED by the D2 gate this cycle: no real recon-walk-from-upload "
                         "extraction exists yet, and the gate needs a walk with a constructively known "
                         "correct answer (the synthetic harness), see the module docstring.")
    ap.add_argument("--plan-dxf", type=Path, default=None,
                    help="real DXF plan (A-WALL/A-GLAZ layers) to perturb for the GENERATOR self-test/demo "
                         "only (not the D2 gate, which always uses the tests/ STAIR harness). Default: none "
                         f"exists in this repo (confirmed) -- falls back to the QA synthetic L-corridor "
                         f"(width={DEFAULT_CORRIDOR_WIDTH} m).")
    ap.add_argument("--n-perturb", type=int, default=GATE_N_PERTURB_DEFAULT,
                    help=f"perturbations in the D2 suite (default {GATE_N_PERTURB_DEFAULT}, the D2 gate size)")
    ap.add_argument("--seed", type=int, default=0,
                    help="base seed; same seed (+ same --n-perturb) -> same suite, reproducibly")
    ap.add_argument("--perturb-mode", choices=["structured", "fragment", "both"], default="structured",
                    help="CYCLE 11: 'structured' (default) edits whole walls (site-change model, see "
                         "module docstring CYCLE 11 block); 'fragment' is Cycle-7's original scattered-"
                         "fragment model (drawing-noise model), kept selectable, not deleted; 'both' runs "
                         "and reports both independently, side by side, never merged")
    ap.add_argument("--selftest-only", action="store_true",
                    help="run ONLY the reproducibility self-test (2x generation, compare) and exit; "
                         "skips suite generation and the D2 gate")
    ap.add_argument("--selftest-flood", action="store_true",
                    help="CYCLE 19: run ONLY the flood-detection regression self-test "
                         "(selftest_flood_detection -- monkeypatch "
                         "scan2bim.coarse_match._span_outliers, in memory only, to flag EVERY "
                         "span unconditionally, verify gate items ④/⑤ reject it) and exit. Uses "
                         "--seed/--n-perturb for the perturbed half of the check.")
    ap.add_argument("--json", type=Path, default=None,
                    help="write the generated suite (segments + ground_truth per perturbation) plus the "
                         "D2 gate result (if it ran) as JSON (or, with --sweep, the sweep result)")
    ap.add_argument("--sweep", action="store_true",
                    help="CYCLE 12: run the DEGRADATION-CURVE sweep (total-change-ratio axis, "
                         "structured mode only, --sweep-seeds x --n-perturb per ratio point) INSTEAD OF "
                         "the default single-suite D2 gate; prints a table. Does NOT change any gate "
                         "definition/threshold (GATE_SUCCESS_RATE_MIN/GATE_OUTLIER_RECALL_MIN read "
                         "unchanged) -- a measurement, not a new judgement. Every other flag above "
                         "(--upload/--plan-dxf/--perturb-mode/--selftest-only) is IGNORED with --sweep "
                         "except --n-perturb/--seed's sibling --json; without --sweep this flag's mere "
                         "existence changes nothing (see run_sweep/_print_sweep_report/_run_sweep_cli).")
    ap.add_argument("--sweep-ratios", type=str, default=None,
                    help="comma-separated total-change-ratio points (fractions of the densified base "
                         "fragment count), e.g. '0.05,0.10,0.15,0.20,0.30,0.40,0.50' (the default). "
                         "Only used with --sweep.")
    ap.add_argument("--sweep-seeds", type=str, default=None,
                    help="comma-separated seeds, e.g. '7,42,123' (the default -- at least 3 recommended "
                         "so mean AND range are meaningful, per this cycle's instruction). Only used "
                         "with --sweep.")
    return ap


def _print_single_gate_report(gate: dict) -> None:
    """CYCLE 11: prints ONE mode's full D2 gate report (density/removal checks, clean
    baseline, per-case summary, the P2 failure-reason breakdown, and ①②③) -- factored
    out of `main` so `--perturb-mode both` can call this twice (structured, fragment)
    and print them clearly labelled side by side, never averaged into one set of
    numbers."""
    mode = gate["perturb_mode"]
    tag = f"[{mode}]"
    dc = gate["density_check"]
    print(f"{tag} P0-Perturb 밀도 보정(Cycle 7 item A): raw N={dc['raw_n_segments']} -> "
          f"densified N={dc['densified_n_segments']} (target_piece_len={DENSIFY_TARGET_PIECE_LEN_M} m); "
          f"무섭동 스켈레톤 불변식(legs/corners/doors/leg-lengths 동일) = "
          f"{'PASS' if dc['ok'] else 'FAIL'} "
          f"(legs {dc['raw_n_legs']}->{dc['densified_n_legs']}, "
          f"corners {dc['raw_n_corners']}->{dc['densified_n_corners']}, "
          f"doors {dc['raw_n_doors']}->{dc['densified_n_doors']})")
    src = gate["single_removal_check"]
    print(f"{tag} 단일 조각 제거 회귀 테스트(이 사이클이 고치려던 버그): {src['n_graceful']}/{src['n_trials']} "
          f"무섭동 위상 유지 = {'PASS' if src['ok'] else 'FAIL'}")

    succ, recall, hold = gate["success"], gate["outlier_recall"], gate["hold"]
    cb = gate["clean_baseline"]
    print(f"{tag} 무섭동 기준해(clean baseline, STAIR, densified): status={cb['status']} "
          f"within_D2={cb['within_d2']}")
    print(f"{tag} D2 게이트 섭동 케이스별 요약 (STAIR, seed={gate['seed']}, mode={mode}):")
    for cs in gate["case_summaries"]:
        print(f"  #{cs['index']:02d} removed={cs['n_removed']} shifted={cs['n_shifted']} "
              f"noise={cs['n_noise']} -> status={cs['status']} hold_reason={cs['hold_reason']} "
              f"margin={cs['margin']} n_cand={cs['n_candidates']} top_score={cs['top_score']} "
              f"top_inlier_ratio={cs['top_inlier_ratio']} error={cs['error']}")
    if gate["contract_violations"]:
        print(f"{tag} [경고] plan_skeleton.validate_match_result 계약 위반 "
              f"{len(gate['contract_violations'])}건 (D2 게이트 판정에는 미반영, 참고용): "
              f"{gate['contract_violations']}", file=sys.stderr)

    print(f"{tag} ① 성공률: {succ['n_ok_within_d2']}/{succ['n']} = {succ['rate']*100:.1f}% "
          f"(gate >= {GATE_SUCCESS_RATE_MIN*100:.0f}%) "
          f"[ok_outside_d2(위험: 확정오답)={succ['n_ok_outside_d2']} hold={succ['n_hold']} "
          f"reject={succ['n_reject']} error={succ['n_error']}]")

    fr = gate["failure_reasons"]
    print(f"{tag}    실패 사유 분포(P2 입력, ①에서 ok_within_d2 아닌 모든 케이스): "
          f"{json.dumps(fr, ensure_ascii=False)}")
    if fr:
        top_reason, top_n = max(fr.items(), key=lambda kv: kv[1])
        n_fail = sum(fr.values())
        if top_reason == "inlier_ratio_below_min":
            print(f"{tag}    해석: inlier_ratio_below_min 이 실패 사유 중 최다({top_n}/{n_fail}건) -- "
                  f"'변경된 벽을 outlier 로 빼는 대신 후보 전체를 거부'하고 있다는 뜻 -- "
                  f"부분매칭(trimmed/RANSAC) 설계의 핵심 실패모드로 보고, 완화 아님.")

    print(f"{tag} ② outlier recall (Cycle 7 분모 교정): 변경 세그먼트 총 {recall['n_all_changed_segments']}건 중 "
          f"not_observable(워크에서 {OUTLIER_RECALL_RADIUS_M:.2f}m 밖)={recall['n_not_observable']}건, "
          f"not_scoreable(shift의 lateral 성분이 매처 leg_lat 톨러런스 이내로 흡수)={recall['n_not_scoreable']}건 "
          f"제외 -- 이 둘은 '측정 대상 아님'이지 '통과'가 아니며 항상 이렇게 보고됨.")
    if recall["recall"] is None:
        print(f"{tag}    scoreable 0건 (case 미확정 전이 exclude={recall['n_excluded_case_no_confirmed_transform']}, "
              f"no_candidate={recall['n_case_no_candidate']}) -> recall 계산 불가 "
              f"(gate >= {GATE_OUTLIER_RECALL_MIN*100:.0f}%)")
    else:
        print(f"{tag}    recall: {recall['n_detected']}/{recall['n_total']} (scoreable, transform-confirmed case만) "
              f"= {recall['recall']*100:.1f}% (radius={recall['radius_m']:.2f}m, "
              f"gate >= {GATE_OUTLIER_RECALL_MIN*100:.0f}%) "
              f"[case 미확정 전이 exclude={recall['n_excluded_case_no_confirmed_transform']} "
              f"no_candidate={recall['n_case_no_candidate']}]")

    print(f"{tag} ③ 모호 시 HOLD(자동확정 금지, 이번 사이클 미변경): 고정 픽스처(2-leg L, 무-door) -> "
          f"status={hold['fixture_status']} hold_reason={hold['fixture_hold_reason']} "
          f"margin={hold['fixture_margin']} => {'PASS' if hold['fixture_ok'] else 'FAIL'}; "
          f"섭동 {gate['n_perturb']}건 중 margin<margin_min인데 status=='ok'로 자동확정된 위반="
          f"{len(hold['perturbation_invariant_violations'])}건 "
          f"{hold['perturbation_invariant_violations'] or ''} "
          f"(reject 는 정상 -- finalize_match 의 하드게이트 우선순위, verify_hold_on_ambiguous 참고)")

    clean_fpr_pct = (gate['clean_span_fpr'] or 0.0) * 100
    print(f"{tag} ④ 무섭동(clean) 도면 span 오경보율(CYCLE 19, ②의 범람 취약점과 짝: dev-core 자진신고): "
          f"{gate['clean_span_fired']}/{gate['n_spans_total']} = {clean_fpr_pct:.1f}% "
          f"(gate <= {GATE_SPAN_CLEAN_FPR_MAX*100:.1f}%) => {'PASS' if gate['gate4_ok'] else 'FAIL'}")
    sp = gate["span_precision"]
    prec_s = "N/A" if sp["precision"] is None else f"{sp['precision']*100:.1f}%"
    recs_s = "N/A" if sp["recall_span"] is None else f"{sp['recall_span']*100:.1f}%"
    f1_s = "N/A" if sp["f1"] is None else f"{sp['f1']*100:.1f}%"
    prev_s = "N/A" if sp["prevalence"] is None else f"{sp['prevalence']*100:.1f}%"
    fired_s = "N/A" if sp["fired_frac"] is None else f"{sp['fired_frac']*100:.1f}%"
    print(f"{tag} ⑤ span 정밀도(CYCLE 19, TP={sp['tp']} FP={sp['fp']} FN={sp['fn']} TN={sp['tn']} "
          f"n_cases={sp['n_cases_included']}): precision={prec_s} recall_span={recs_s} F1={f1_s} "
          f"prevalence(실제변경 span 비율)={prev_s} 발화율={fired_s} "
          f"(gate: precision >= {GATE_SPAN_PRECISION_PREVALENCE_FACTOR}x prevalence) "
          f"=> {'PASS' if gate['gate5_ok'] else 'FAIL'}")

    print(f"{tag} 게이트 판정: {'PASS' if gate['gate_ok'] else 'FAIL'} "
          f"(①{'PASS' if gate['gate1_ok'] else 'FAIL'} "
          f"②{'PASS' if gate['gate2_ok'] else 'FAIL'} "
          f"③{'PASS' if gate['gate3_ok'] else 'FAIL'} "
          f"④{'PASS' if gate['gate4_ok'] else 'FAIL'} "
          f"⑤{'PASS' if gate['gate5_ok'] else 'FAIL'})")


def _print_sweep_report(sweep: dict) -> None:
    """Prints the degradation-curve TABLE (총변경비율 x 성공률/recall/실패사유/HOLD위반/
    ok_outside_d2), per instruction "표를 반드시 포함" -- then the same table's per-seed
    rows (mean AND range come from these, never reported alone)."""
    w = sweep["alloc_weights"]
    print(f"[sweep] 열화곡선 측정 (게이트 정의/임계 불변, mode=structured 전용): "
          f"seeds={sweep['seeds']} n_perturb(지점당 seed당)={sweep['n_perturb']} "
          f"base_fragments={sweep['n_base_fragments']}")
    print(f"[sweep] 배분 규칙(Section 2c, QA 기본 범위의 중앙값 비율 그대로 -- 범위 자체는 미변경): "
          f"총변경비율 T -> remove_frac=T*{w['w_remove']:.4f}(of N), "
          f"noise_frac=T*{w['w_noise']:.4f}(of N), "
          f"shift_frac(of kept)=T*{w['w_shift']:.4f}*N/kept")
    cb = sweep["clean_baseline"]
    print(f"[sweep] 무섭동 기준해: status={cb['status']} within_D2={cb['within_d2']}")
    print("")
    header = (f"{'목표T':>6} {'실측T평균':>9} | {'성공률 평균(범위)':>20} | {'recall 평균(범위)':>26} | "
             f"{'precision 평균':>13} | {'ok_outside_d2':>13} | {'HOLD위반':>8} | {'게이트①②③④⑤':>14} | "
             f"실패사유 분포(3seed 합산)")
    print(header)
    print("-" * len(header))
    for p in sweep["points"]:
        recall_str = ("N/A(scoreable 0)" if p["recall_mean"] is None else
                      f"{p['recall_mean']*100:5.1f}%({p['recall_min']*100:.0f}-{p['recall_max']*100:.0f}%)"
                      f"[{p['recall_n_seeds_scoreable']}/{p['recall_n_seeds_total']}seed]")
        precision_str = ("N/A" if p["precision_mean"] is None
                         else f"{p['precision_mean']*100:5.1f}%[{p['precision_n_seeds_scoreable']}seed]")
        gates = (f"{'P' if p['all_seeds_gate1_ok'] else 'F'}"
                f"{'P' if p['all_seeds_gate2_ok'] else 'F'}"
                f"{'P' if p['all_seeds_gate3_ok'] else 'F'}"
                f"{'P' if p['all_seeds_gate4_ok'] else 'F'}"
                f"{'P' if p['all_seeds_gate5_ok'] else 'F'}")
        row = (f"{p['ratio_target']*100:5.0f}% {p['achieved_ratio_mean']*100:8.1f}% | "
              f"{p['success_rate_mean']*100:5.1f}%({p['success_rate_min']*100:.0f}-{p['success_rate_max']*100:.0f}%) | "
              f"{recall_str:>26} | "
              f"{precision_str:>13} | "
              f"{p['n_ok_outside_d2_total']:>13} | {p['hold_violations_total']:>8} | "
              f"{gates:>14} | {json.dumps(p['failure_reasons_merged'], ensure_ascii=False)}")
        print(row)
    print("")
    print("[sweep] 지점별 seed 상세 (위 평균/범위의 근거, per-seed 원값):")
    for p in sweep["points"]:
        for s in p["per_seed"]:
            recall_s = "N/A" if s["recall"] is None else f"{s['recall']*100:.1f}%"
            sp = s["span_precision"]
            prec_s = "N/A" if sp["precision"] is None else f"{sp['precision']*100:.1f}%"
            prev_s = "N/A" if sp["prevalence"] is None else f"{sp['prevalence']*100:.1f}%"
            gate45 = (f"{'P' if s['gate4_ok'] else 'F'}{'P' if s['gate5_ok'] else 'F'}")
            print(f"  T={p['ratio_target']*100:.0f}% seed={s['seed']:>4} 실측T={s['achieved_ratio_mean']*100:.1f}% "
                  f"({s['achieved_ratio_min']*100:.1f}-{s['achieved_ratio_max']*100:.1f}%) "
                  f"성공률={s['success_rate']*100:.1f}%({s['n_ok_within_d2']}/{s['n']}) "
                  f"ok_outside_d2={s['n_ok_outside_d2']} recall={recall_s} "
                  f"HOLD위반={s['hold_violations']} 실패사유={json.dumps(s['failure_reasons'], ensure_ascii=False)} "
                  f"span_precision={prec_s} prevalence={prev_s} 게이트④⑤={gate45}")


def _run_sweep_cli(args) -> int:
    """CLI dispatch for `--sweep` (CYCLE 12) -- entirely SEPARATE from the default
    single-suite D2 gate path in `main` (`main` calls this and returns IMMEDIATELY, before
    touching `--plan-dxf`/`--selftest-only`/the default `run_d2_gate` call at all, so
    `--n-perturb 20 --seed 7` WITHOUT `--sweep` is provably unaffected by this function's
    existence -- see the cycle report's no-regression check). NOT a pass/fail gate --
    exits 0 once the sweep is MEASURED (a low- or high-ratio point failing
    GATE_SUCCESS_RATE_MIN/GATE_OUTLIER_RECALL_MIN is exactly as valid and reportable a
    measurement as one passing -- that is the whole point of a degradation curve); exits
    2 only if the harness itself could not be built/evaluated (ImportError/RuntimeError
    from `_build_d2_harness_context`, same conditions `run_d2_gate` treats as "not
    evaluable" -- never a fabricated pass)."""
    ratios = (tuple(float(x) for x in args.sweep_ratios.split(","))
             if args.sweep_ratios else SWEEP_RATIOS_DEFAULT)
    seeds = (tuple(int(x) for x in args.sweep_seeds.split(","))
            if args.sweep_seeds else SWEEP_SEEDS_DEFAULT)
    print(f"[sweep] ratios={list(ratios)} seeds={list(seeds)} n_perturb={args.n_perturb} "
          f"(--upload/--plan-dxf/--perturb-mode/--selftest-only 무시됨 -- --sweep 전용 경로)")
    try:
        sweep = run_sweep(ratios=ratios, seeds=seeds, n_perturb=args.n_perturb)
    except ImportError as e:
        print(f"[sweep 평가 불가] {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"[sweep 평가 불가] {e}", file=sys.stderr)
        return 2
    _print_sweep_report(sweep)
    if args.json is not None:
        args.json.write_text(json.dumps(sweep, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"sweep 결과 written: {args.json}")
    return 0


def main(argv=None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    if args.sweep:
        return _run_sweep_cli(args)

    if args.selftest_flood:
        try:
            res = selftest_flood_detection(seed=args.seed, n_perturb=args.n_perturb)
        except ImportError as e:
            print(f"[flood selftest 평가 불가] {e}", file=sys.stderr)
            return 2
        except RuntimeError as e:
            print(f"[flood selftest 평가 불가] {e}", file=sys.stderr)
            return 2
        print("[flood selftest] scan2bim.coarse_match._span_outliers 를 '모든 span 무조건 발화'로 "
              "인메모리 monkeypatch(디스크 미수정, finally 에서 원복) -- dev-core 8d7fcbf 자진신고 "
              "재현: recall 38.8%->91.2% 개선과 함께 '84개 span 전부 깃발 꽂으면 recall 100%'을 "
              "self-report 했던 바로 그 가짜 구현")
        print(f"[flood selftest] 무섭동(clean) 도면(패치 하에서 재빌드): "
              f"{res['clean_span_fired']}/{res['n_spans_total']} span 발화 "
              f"(오경보율={(res['clean_span_fpr'] or 0)*100:.1f}%) -> "
              f"④ {'PASS' if res['gate4_ok'] else 'FAIL'}")
        sp = res["span_precision"]
        prec_s = "N/A" if sp["precision"] is None else f"{sp['precision']*100:.1f}%"
        prev_s = "N/A" if sp["prevalence"] is None else f"{sp['prevalence']*100:.1f}%"
        print(f"[flood selftest] 섭동 스위트(seed={res['seed']} n={res['n_perturb']}): "
              f"precision={prec_s} prevalence={prev_s} (TP={sp['tp']} FP={sp['fp']} FN={sp['fn']}) -> "
              f"⑤ {'PASS' if res['gate5_ok'] else 'FAIL'}")
        print(f"[flood selftest] 참고로 ①②③(가짜 구현이 판정 경로를 건드리지 않았다는 격리 확인): "
              f"①{'PASS' if res['gate1_ok'] else 'FAIL'} ②{'PASS' if res['gate2_ok'] else 'FAIL'} "
              f"③{'PASS' if res['gate3_ok'] else 'FAIL'}")
        if res["caught"]:
            print("[flood selftest] 결과: PASS -- 게이트가 범람을 잡았다(④ 또는 ⑤가 FAIL)")
            return 0
        print("[flood selftest] 결과: FAIL -- 게이트가 범람을 통과시켰다(④⑤ 모두 PASS로 나옴) -- "
              "정밀도 지표가 무력화됐다는 뜻, 즉시 보고 필요", file=sys.stderr)
        return 1

    if args.plan_dxf is not None:
        if not args.plan_dxf.exists():
            print(f"error: --plan-dxf not found: {args.plan_dxf}", file=sys.stderr)
            return 2
        from scan2bim.dxf_plan import load_wall_segments
        base_segments = load_wall_segments(args.plan_dxf)
        plan_src = str(args.plan_dxf)
    else:
        base_segments = synth_l_corridor_segments()
        plan_src = (f"synthetic L-corridor (QA fixture, width={DEFAULT_CORRIDOR_WIDTH} m) -- "
                    "no real SXX/Gasan DXF exists in this repo as of this cycle")

    #: CYCLE 11: trivial 1-fragment-per-wall tagging for the CLI's OWN demo/self-test
    #: section only (this raw base is NOT densified, unlike the D2 gate's STAIR harness
    #: -- see `_run_d2_gate_single`, which builds its own properly-densified `wall_ids`).
    #: Harmless for 'fragment' mode (ignored there).
    demo_wall_ids = np.arange(len(base_segments), dtype=np.int64)

    print(f"plan source (generator self-test only): {plan_src}")
    print(f"base wall segments: N={len(base_segments)}")
    if args.upload is not None:
        print(f"--upload {args.upload} accepted but UNUSED by the D2 gate (see module docstring)")

    modes_to_run = ("structured", "fragment") if args.perturb_mode == "both" else (args.perturb_mode,)

    repro_all_ok = True
    for m in modes_to_run:
        repro = selftest_reproducibility(base_segments, seed=args.seed, n_perturb=min(args.n_perturb, 5),
                                         mode=m, wall_ids=demo_wall_ids)
        print(f"reproducibility selftest [{m}]: seed={args.seed} n={repro['n_perturb']} -> "
              f"{'PASS' if repro['ok'] else 'FAIL'} (mismatches={len(repro['mismatches'])})")
        if not repro["ok"]:
            print(json.dumps(repro, indent=2, ensure_ascii=False), file=sys.stderr)
            repro_all_ok = False
    if not repro_all_ok:
        return 1

    if args.selftest_only:
        return 0

    suites = {}
    for m in modes_to_run:
        suite = generate_perturbation_suite(base_segments, args.seed, args.n_perturb, mode=m,
                                            wall_ids=demo_wall_ids)
        suites[m] = suite
        print(f"generated {len(suite)} perturbations (seed={args.seed}, mode={m}):")
        for item in suite:
            gt = item["ground_truth"]
            p = gt["params"]
            print(f"  #{item['index']:02d} child_seed={item['child_seed']} N_out={gt['n_output']} "
                  f"removed={p['n_removed']} shifted={p['n_shifted']} noise={p['n_noise']}")

    matcher = try_import_matcher()
    if matcher is None:
        print("", file=sys.stderr)
        print("[매처 미구현] scan2bim/coarse_match.py 가 아직 존재하지 않습니다 -- "
              "D2 강건성 게이트(성공률>=%.0f%%, outlier recall>=%.0f%%, 모호 시 HOLD)를 "
              "평가할 매처가 없습니다." % (GATE_SUCCESS_RATE_MIN * 100, GATE_OUTLIER_RECALL_MIN * 100),
              file=sys.stderr)
        print("matcher_missing: import scan2bim.coarse_match failed (module not found)",
              file=sys.stderr)
        if args.json is not None:
            dump = {m: [{"index": it["index"], "seed": it["seed"], "child_seed": it["child_seed"],
                        "segments": it["segments"].tolist(), "ground_truth": it["ground_truth"]}
                       for it in suites[m]] for m in suites}
            args.json.write_text(json.dumps(dump, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"suite written: {args.json}")
        return 2

    print("")
    print(f"[D2 게이트 실배선] tests/test_coarse_match.py 하네스(STAIR corridor, "
          f"4 legs/3 corners/5 doors) 재사용 -- QA 섭동 생성기(seed={args.seed}, "
          f"n={args.n_perturb}, mode={args.perturb_mode})로, 밀도 보정(densify_wall_segments) 후의 "
          f"벽만 섭동, 문/워크는 고정.")
    try:
        gate = run_d2_gate(args.seed, args.n_perturb, perturb_mode=args.perturb_mode)
    except ImportError as e:
        print(f"[D2 게이트 평가 불가] {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"[D2 게이트 평가 불가] {e}", file=sys.stderr)
        return 2

    if args.perturb_mode == "both":
        _print_single_gate_report(gate["structured"])
        print("")
        _print_single_gate_report(gate["fragment"])
    else:
        _print_single_gate_report(gate)

    if args.json is not None:
        dump = {"perturb_mode": args.perturb_mode,
                "suites": {m: [{"index": it["index"], "seed": it["seed"], "child_seed": it["child_seed"],
                                "segments": it["segments"].tolist(), "ground_truth": it["ground_truth"]}
                               for it in suites[m]] for m in suites},
                "d2_gate": gate}
        args.json.write_text(json.dumps(dump, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"suite + D2 gate result written: {args.json}")

    print("")
    if gate["gate_ok"]:
        extra = " (both 모드 모두 통과)" if args.perturb_mode == "both" else ""
        print(f"D2 게이트: PASS (mode={args.perturb_mode}, ①②③ 모두 통과{extra})")
        return 0

    if args.perturb_mode == "both":
        misses = []
        for m in ("structured", "fragment"):
            g = gate[m]
            if g["gate_ok"]:
                continue
            sub = []
            if not g["gate1_ok"]:
                sub.append(f"①{g['success']['rate']*100:.1f}%<{GATE_SUCCESS_RATE_MIN*100:.0f}%")
            if not g["gate2_ok"]:
                rr = "계산불가" if g["outlier_recall"]["recall"] is None else f"{g['outlier_recall']['recall']*100:.1f}%"
                sub.append(f"②{rr}(<{GATE_OUTLIER_RECALL_MIN*100:.0f}%)")
            if not g["gate3_ok"]:
                sub.append("③HOLD위반")
            if not g["gate4_ok"]:
                sub.append(f"④무섭동오경보{g['clean_span_fired']}/{g['n_spans_total']}>0")
            if not g["gate5_ok"]:
                pv = "N/A" if g["span_precision"]["precision"] is None else f"{g['span_precision']['precision']*100:.1f}%"
                sub.append(f"⑤precision{pv}<{GATE_SPAN_PRECISION_PREVALENCE_FACTOR}x prevalence")
            misses.append(f"[{m}] " + "; ".join(sub))
        print("D2 게이트: FAIL -- " + " | ".join(misses), file=sys.stderr)
        return 1

    misses = []
    if not gate["gate1_ok"]:
        misses.append(f"①성공률 {gate['success']['rate']*100:.1f}% < {GATE_SUCCESS_RATE_MIN*100:.0f}%")
    if not gate["gate2_ok"]:
        rr = "계산불가" if gate["outlier_recall"]["recall"] is None else f"{gate['outlier_recall']['recall']*100:.1f}%"
        misses.append(f"②outlier recall {rr} (gate >= {GATE_OUTLIER_RECALL_MIN*100:.0f}%)")
    if not gate["gate3_ok"]:
        misses.append("③모호 시 HOLD 위반")
    if not gate["gate4_ok"]:
        misses.append(f"④무섭동 span 오경보 {gate['clean_span_fired']}/{gate['n_spans_total']} > 0 "
                      f"(gate <= {GATE_SPAN_CLEAN_FPR_MAX*100:.1f}%)")
    if not gate["gate5_ok"]:
        sp = gate["span_precision"]
        pv = "N/A" if sp["precision"] is None else f"{sp['precision']*100:.1f}%"
        pr = "N/A" if sp["prevalence"] is None else f"{sp['prevalence']*100:.1f}%"
        misses.append(f"⑤span precision {pv} < {GATE_SPAN_PRECISION_PREVALENCE_FACTOR}x "
                      f"prevalence({pr})")
    print(f"D2 게이트: FAIL(mode={args.perturb_mode}) -- " + "; ".join(misses), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

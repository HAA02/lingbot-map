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
denser base) changed.

BASE SYNTHETIC PLAN (Section 1, `synth_l_corridor_segments`): no real SXX/Gasan DXF
exists in this repo. Confirmed by `find / -iname '*.dxf' -not -path '*/.git/*'` under
the repo root (only unrelated files under /usr/share and the user's ~/다운로드 turned
up). This fixture stays INDEPENDENT of tests/ on purpose (re-derived, not imported) --
it is a 2-leg L (1 corner), which per `coarse_match.py`'s own documented load-bearing
claim is AMBIGUOUS by construction without a door, so it is used for the CLI's
generator self-test/demo, not for the D2 accuracy gate (see next paragraph for why the
gate uses a different, imported fixture).

D2 GATE FIXTURE (Section 3, `run_d2_gate` / `_load_d2_harness`): DELIBERATELY
DIFFERENT from Section 1's fixture, and DELIBERATELY imported from
`tests/test_coarse_match.py` rather than re-derived, per the cycle-6 instruction to
reuse that harness (still honoured this cycle). The STAIR corridor (4 legs / 3 corners
/ 5 doors) is UNAMBIGUOUS, which the 2-leg L is not -- the D2 success-rate metric needs
an unambiguous plan to be measuring ACCURACY under perturbation, not tie-breaking (the
tie-breaking case, item (3) of the gate, uses a SEPARATE fixed fixture reused from
`tests/test_plan_skeleton.py`'s 2-leg L instead, see `build_ambiguous_fixture`). This
cycle, `run_d2_gate` runs `densify_wall_segments` on that harness's raw 10-segment
`_corridor_walls(STAIR)` output BEFORE perturbing it -- see Section 1b / Section 3.

CLI:
    .venv/bin/python tools/check_plan_match_robust.py [--upload PATH] \\
        [--plan-dxf PLAN.dxf] [--n-perturb 20] [--seed 0] [--selftest-only] [--json OUT.json]

`--upload` is accepted but UNUSED by the D2 gate: no real recon-walk-from-upload
extraction exists in this repo as of this cycle (that is separate P3 scope), and the
D2 gate needs a walk with a CONSTRUCTIVELY known correct answer to measure accuracy
against, which only the synthetic harness provides (dev-core's own test suite is
synthetic-only for the same reason, see its module docstring).

Exit codes: 0 = D2 gate PASS (success-rate AND outlier-recall AND ambiguous-HOLD all
met), or `--selftest-only` PASS; 1 = a generator self-test failed, OR the D2 gate ran
but one or more of its three items missed the threshold (stated on stderr with the
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

    Returns (M,2,2) float64, M >= len(segments) (M == len(segments) only for
    already-short walls where `_piece_count` returns 1, e.g. none at this fixture's
    wall lengths with the default target)."""
    seg = np.asarray(segments, dtype=np.float64).reshape(-1, 2, 2)
    out = []
    for p, q in seg:
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        d = q - p
        L = float(np.linalg.norm(d))
        n = _piece_count(L, target_piece_len, min_pieces, max_pieces)
        if L < 1e-9 or n <= 1:
            out.append((p, q))
            continue
        u = d / L
        breaks = np.linspace(0.0, L, n + 1)
        for s0, s1 in zip(breaks, breaks[1:]):
            out.append((p + u * s0, p + u * s1))
    return np.asarray(out, dtype=np.float64).reshape(-1, 2, 2)


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
# SECTION 2 -- seeded segment perturbation generator (QA-owned parameters; do not relax)
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

    UNCHANGED this cycle (Cycle 7 touched Section 1b / Section 3 only): still
    operates fragment-by-fragment on whatever flat (N,2,2) array it is given, with no
    notion of "which fragments came from the same original wall" -- this stays
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


def derive_seeds(seed: int, n: int) -> list:
    """n independent child seeds from one base seed via numpy's SeedSequence.spawn
    (reproducible AND statistically independent across suite members -- unlike
    `seed + i`, which would correlate adjacent perturbations)."""
    ss = np.random.SeedSequence(int(seed))
    children = ss.spawn(int(n))
    return [int(c.generate_state(1)[0]) for c in children]


def generate_perturbation_suite(base_segments, seed: int, n_perturb: int, **perturb_kwargs) -> list:
    """n_perturb perturbations of `base_segments`, deterministic in (seed, n_perturb,
    **perturb_kwargs). Each item: {"index", "seed", "child_seed", "segments",
    "ground_truth"}."""
    child_seeds = derive_seeds(seed, n_perturb)
    suite = []
    for i, cs in enumerate(child_seeds):
        segs, gt = perturb_wall_segments(base_segments, cs, **perturb_kwargs)
        suite.append({"index": i, "seed": int(seed), "child_seed": int(cs),
                      "segments": segs, "ground_truth": gt})
    return suite


def selftest_reproducibility(base_segments, seed: int = 0, n_perturb: int = 5) -> dict:
    """Runs `generate_perturbation_suite` TWICE with identical arguments and checks
    byte-identical output (segments array equality + ground_truth JSON equality) --
    the reproducibility requirement the CLI must demonstrate every invocation."""
    run1 = generate_perturbation_suite(base_segments, seed, n_perturb)
    run2 = generate_perturbation_suite(base_segments, seed, n_perturb)
    mismatches = []
    for i, (a, b) in enumerate(zip(run1, run2)):
        same_array = a["segments"].shape == b["segments"].shape and bool(
            np.array_equal(a["segments"], b["segments"]))
        same_gt = json.dumps(a["ground_truth"], sort_keys=True) == json.dumps(b["ground_truth"], sort_keys=True)
        same_child_seed = a["child_seed"] == b["child_seed"]
        if not (same_array and same_gt and same_child_seed):
            mismatches.append({"index": i, "same_array": same_array, "same_gt": same_gt,
                               "same_child_seed": same_child_seed})
    return {"ok": not mismatches, "seed": int(seed), "n_perturb": int(n_perturb),
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


def run_d2_gate(seed: int, n_perturb: int) -> dict:
    """Runs the full D2 gate: builds the dev-core STAIR harness (4 legs / 3 corners /
    5 doors, unambiguous), DENSIFIES its wall segments (Cycle 7 item (A) --
    `densify_wall_segments`, verified as a no-op on the clean skeleton and on
    single-fragment removal BEFORE anything is perturbed), perturbs the densified
    WALL SEGMENTS ONLY `n_perturb` times (QA's seeded generator, Section 2 -- doors
    are held fixed, see the module docstring), matches each perturbed plan against the
    SAME walk (generated once, from the UNPERTURBED plan's known true transform -- the
    perturbation simulates a stale/drifted DRAWING, not a different walk), and
    evaluates all three D2 items (item (2)'s denominator redefined per Cycle 7 item
    (B), see `classify_changed_segments` / `compute_outlier_recall`).

    Raises ImportError (matcher/tests-harness missing) or RuntimeError (the density
    pass changed the skeleton it should have left untouched, the single-fragment-
    removal regression probe found a topology collapse, or the harness's OWN
    zero-perturbation baseline was not recovered exactly) -- the caller (`main`) turns
    any of these into exit 2, never a fabricated verdict."""
    cm, ps, tcm, tps = _load_d2_harness()

    raw_segments = tcm._corridor_walls(tcm.STAIR)                        # (10,2,2) m -- dev-core's own fixture
    plan_doors = list(tcm.STAIR_PLAN_DOORS)
    base_segments = densify_wall_segments(raw_segments)                 # Cycle 7 item (A)

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

    suite = generate_perturbation_suite(base_segments, seed, n_perturb)
    rev_by_index = {e["index"]: e for e in cm.recon_events(traj, door_s=door_s)}
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

    succ = compute_success_rate(cases, tf_true, offset_max)
    transform_confirmed_indices = {d["index"] for d in succ["detail"] if d["outcome"] == "ok_within_d2"}

    fallback_leg_lat_tol = max(float(cm.TOL_FLOOR["leg_lat"]),
                               float(cm.TOL_PER_WIDTH["leg_lat"]) * float(tcm.W))
    classification = classify_changed_segments(cases, base_segments, plan_walk,
                                               fallback_leg_lat_tol=fallback_leg_lat_tol)
    recall = compute_outlier_recall(cases, base_segments, classification, rev_by_index,
                                    ps.apply_candidate_transform, transform_confirmed_indices)
    ambiguous = build_ambiguous_fixture(cm, ps, tcm, tps)
    hold = verify_hold_gate(cases, ambiguous)

    #: per-perturbation summary (JSON-safe scalars only -- for --json / the cycle
    #: report; NOT used by the gate math itself, which reads `cases` directly).
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
    return {"n_perturb": n_perturb, "seed": seed,
            "density_check": density_check,
            "single_removal_check": single_removal_check,
            "clean_baseline": {"status": clean_res.get("status"), "within_d2": clean_ok,
                               "detail": clean_detail},
            "case_summaries": case_summaries,
            "contract_violations": contract_violations,
            "success": succ, "outlier_recall": recall, "hold": hold,
            "gate1_ok": gate1_ok, "gate2_ok": gate2_ok, "gate3_ok": gate3_ok,
            "gate_ok": bool(gate1_ok and gate2_ok and gate3_ok)}


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
    ap.add_argument("--selftest-only", action="store_true",
                    help="run ONLY the reproducibility self-test (2x generation, compare) and exit; "
                         "skips suite generation and the D2 gate")
    ap.add_argument("--json", type=Path, default=None,
                    help="write the generated suite (segments + ground_truth per perturbation) plus the "
                         "D2 gate result (if it ran) as JSON")
    return ap


def main(argv=None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

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

    print(f"plan source (generator self-test only): {plan_src}")
    print(f"base wall segments: N={len(base_segments)}")
    if args.upload is not None:
        print(f"--upload {args.upload} accepted but UNUSED by the D2 gate (see module docstring)")

    repro = selftest_reproducibility(base_segments, seed=args.seed, n_perturb=min(args.n_perturb, 5))
    print(f"reproducibility selftest: seed={args.seed} n={repro['n_perturb']} -> "
          f"{'PASS' if repro['ok'] else 'FAIL'} (mismatches={len(repro['mismatches'])})")
    if not repro["ok"]:
        print(json.dumps(repro, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1

    if args.selftest_only:
        return 0

    suite = generate_perturbation_suite(base_segments, args.seed, args.n_perturb)
    print(f"generated {len(suite)} perturbations (seed={args.seed}):")
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
            args.json.write_text(json.dumps(
                [{"index": it["index"], "seed": it["seed"], "child_seed": it["child_seed"],
                  "segments": it["segments"].tolist(), "ground_truth": it["ground_truth"]}
                 for it in suite], indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"suite written: {args.json}")
        return 2

    print("")
    print(f"[D2 게이트 실배선] tests/test_coarse_match.py 하네스(STAIR corridor, "
          f"4 legs/3 corners/5 doors) 재사용 -- QA 섭동 생성기(seed={args.seed}, "
          f"n={args.n_perturb})로, 밀도 보정(densify_wall_segments) 후의 벽만 섭동, 문/워크는 고정.")
    try:
        gate = run_d2_gate(args.seed, args.n_perturb)
    except ImportError as e:
        print(f"[D2 게이트 평가 불가] {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"[D2 게이트 평가 불가] {e}", file=sys.stderr)
        return 2

    dc = gate["density_check"]
    print(f"P0-Perturb 밀도 보정(Cycle 7 item A): raw N={dc['raw_n_segments']} -> "
          f"densified N={dc['densified_n_segments']} (target_piece_len={DENSIFY_TARGET_PIECE_LEN_M} m); "
          f"무섭동 스켈레톤 불변식(legs/corners/doors/leg-lengths 동일) = "
          f"{'PASS' if dc['ok'] else 'FAIL'} "
          f"(legs {dc['raw_n_legs']}->{dc['densified_n_legs']}, "
          f"corners {dc['raw_n_corners']}->{dc['densified_n_corners']}, "
          f"doors {dc['raw_n_doors']}->{dc['densified_n_doors']})")
    src = gate["single_removal_check"]
    print(f"단일 조각 제거 회귀 테스트(이 사이클이 고치려던 버그): {src['n_graceful']}/{src['n_trials']} "
          f"무섭동 위상 유지 = {'PASS' if src['ok'] else 'FAIL'}")

    succ, recall, hold = gate["success"], gate["outlier_recall"], gate["hold"]
    cb = gate["clean_baseline"]
    print(f"무섭동 기준해(clean baseline, STAIR, densified): status={cb['status']} within_D2={cb['within_d2']}")
    print("D2 게이트 섭동 케이스별 요약 (STAIR, seed=%d):" % args.seed)
    for cs in gate["case_summaries"]:
        print(f"  #{cs['index']:02d} removed={cs['n_removed']} shifted={cs['n_shifted']} "
              f"noise={cs['n_noise']} -> status={cs['status']} hold_reason={cs['hold_reason']} "
              f"margin={cs['margin']} n_cand={cs['n_candidates']} top_score={cs['top_score']} "
              f"top_inlier_ratio={cs['top_inlier_ratio']} error={cs['error']}")
    if gate["contract_violations"]:
        print(f"[경고] plan_skeleton.validate_match_result 계약 위반 {len(gate['contract_violations'])}건 "
              f"(D2 게이트 판정에는 미반영, 참고용): {gate['contract_violations']}", file=sys.stderr)

    print(f"① 성공률: {succ['n_ok_within_d2']}/{succ['n']} = {succ['rate']*100:.1f}% "
          f"(gate >= {GATE_SUCCESS_RATE_MIN*100:.0f}%) "
          f"[ok_outside_d2(위험: 확정오답)={succ['n_ok_outside_d2']} hold={succ['n_hold']} "
          f"reject={succ['n_reject']} error={succ['n_error']}]")

    print(f"② outlier recall (Cycle 7 분모 교정): 변경 세그먼트 총 {recall['n_all_changed_segments']}건 중 "
          f"not_observable(워크에서 {OUTLIER_RECALL_RADIUS_M:.2f}m 밖)={recall['n_not_observable']}건, "
          f"not_scoreable(shift의 lateral 성분이 매처 leg_lat 톨러런스 이내로 흡수)={recall['n_not_scoreable']}건 "
          f"제외 -- 이 둘은 '측정 대상 아님'이지 '통과'가 아니며 항상 이렇게 보고됨.")
    if recall["recall"] is None:
        print(f"   scoreable 0건 (case 미확정 전이 exclude={recall['n_excluded_case_no_confirmed_transform']}, "
              f"no_candidate={recall['n_case_no_candidate']}) -> recall 계산 불가 "
              f"(gate >= {GATE_OUTLIER_RECALL_MIN*100:.0f}%)")
    else:
        print(f"   recall: {recall['n_detected']}/{recall['n_total']} (scoreable, transform-confirmed case만) "
              f"= {recall['recall']*100:.1f}% (radius={recall['radius_m']:.2f}m, "
              f"gate >= {GATE_OUTLIER_RECALL_MIN*100:.0f}%) "
              f"[case 미확정 전이 exclude={recall['n_excluded_case_no_confirmed_transform']} "
              f"no_candidate={recall['n_case_no_candidate']}]")

    print(f"③ 모호 시 HOLD(자동확정 금지, 이번 사이클 미변경): 고정 픽스처(2-leg L, 무-door) -> "
          f"status={hold['fixture_status']} hold_reason={hold['fixture_hold_reason']} "
          f"margin={hold['fixture_margin']} => {'PASS' if hold['fixture_ok'] else 'FAIL'}; "
          f"섭동 {args.n_perturb}건 중 margin<margin_min인데 status=='ok'로 자동확정된 위반="
          f"{len(hold['perturbation_invariant_violations'])}건 "
          f"{hold['perturbation_invariant_violations'] or ''} "
          f"(reject 는 정상 -- finalize_match 의 하드게이트 우선순위, verify_hold_on_ambiguous 참고)")

    if args.json is not None:
        dump = {"suite": [{"index": it["index"], "seed": it["seed"], "child_seed": it["child_seed"],
                           "segments": it["segments"].tolist(), "ground_truth": it["ground_truth"]}
                          for it in suite],
                "d2_gate": gate}
        args.json.write_text(json.dumps(dump, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"suite + D2 gate result written: {args.json}")

    print("")
    if gate["gate_ok"]:
        print("D2 게이트: PASS (①②③ 모두 통과)")
        return 0
    misses = []
    if not gate["gate1_ok"]:
        misses.append(f"①성공률 {succ['rate']*100:.1f}% < {GATE_SUCCESS_RATE_MIN*100:.0f}%")
    if not gate["gate2_ok"]:
        rr = "계산불가" if recall["recall"] is None else f"{recall['recall']*100:.1f}%"
        misses.append(f"②outlier recall {rr} (gate >= {GATE_OUTLIER_RECALL_MIN*100:.0f}%)")
    if not gate["gate3_ok"]:
        misses.append("③모호 시 HOLD 위반")
    print("D2 게이트: FAIL -- " + "; ".join(misses), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

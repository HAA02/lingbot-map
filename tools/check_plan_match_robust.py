#!/usr/bin/env python3
"""D2 robustness gate for coplay-planmatch-01: seeded plan-perturbation generator
+ (skeleton) success-rate / outlier-recall / ambiguous-HOLD verdict.

OWNERSHIP: this file, and the perturbation parameters / judgement thresholds it
defines, are QA-OWNED (coplay-planmatch-01-qa). Dev requests to relax them go to
PM, not into an edit here.

THIS CYCLE'S SCOPE (P0-Perturb): only the perturbation generator + the base
synthetic plan are implemented and load-bearing. `scan2bim/coarse_match.py` (the
matcher, dev-core P1) does not exist yet, so `main()` deliberately stops at
exit 2 once the generator/self-test has run -- see `try_import_matcher()`. The
gate math (`compute_success_rate`, `compute_outlier_recall`,
`verify_hold_on_ambiguous`) is defined below as a SKELETON with the D2 thresholds
already fixed, but it is not wired into `main()` yet (next cycle, once a matcher
exists to feed it real per-perturbation verdicts).

PERTURBATION MODEL (operates on `scan2bim.dxf_plan.load_wall_segments`'s (N,2,2)
metre format, whether the array came from a real DXF or the synthetic fixture
below):
  (a) REMOVE   10-30% of wall segments (torn-down / undrawn walls)
  (b) SHIFT    some of the remaining segments 0.3-1.0 m (as-built drift / stale DXF)
  (c) ADD      noise segments the plan never had (furniture / temporary partitions)
All draws come from a single `numpy.random.default_rng(seed)` stream, so the same
seed reproduces byte-identical output (segments array AND ground_truth dict) --
see `selftest_reproducibility()` / the `--selftest-only` CLI flag.

BASE SYNTHETIC PLAN: no real SXX/Gasan DXF exists in this repo. Confirmed by
`find / -iname '*.dxf' -not -path '*/.git/*'` under the repo root during this
cycle's investigation (only unrelated files under /usr/share and the user's
~/다운로드 turned up; nothing under this worktree or lingbot_map/models). Until a
real plan is supplied, `synth_l_corridor_segments()` (known L-shaped corridor,
clear width ~=1.82 m -- the Gasan_7F measured value used throughout this team,
e.g. scan2bim/plan_skeleton.py's DEFAULT_WIDTH_RANGE docstring and
tests/test_plan_skeleton.py) is the ONLY plan fixture available for the D2 suite.
This fixture is independently re-derived here (not imported from tests/), since
this file has no legitimate reason to depend on the dev-core test suite.

Ground truth for outlier recall: `perturb_wall_segments()` returns, alongside the
perturbed array, exactly which original segments were removed/shifted and which
output rows are pure noise. Once coarse_match.py + plan_events() let us map a
changed WALL SEGMENT to the PLAN EVENT(S) it feeds (a leg/corner it participates
in), `compute_outlier_recall()` compares the matcher's emitted outlier list
against this answer key. That mapping does not exist yet either (P1/P2), which is
why the recall function currently raises NotImplementedError instead of guessing.

CLI:
    .venv/bin/python tools/check_plan_match_robust.py [--upload PATH] \\
        [--plan-dxf PLAN.dxf] [--n-perturb 20] [--seed 0] [--selftest-only] [--json OUT.json]

Exit codes: 0 = selftest-only pass (or, in a future cycle, gate PASS); 1 = a
generator self-test failed; 2 = the D2 gate cannot be evaluated because the
matcher is not implemented yet (or, once it exists, because gate wiring itself is
still a stub) -- NEVER a fabricated success.
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
# SECTION 3 -- D2 gate (thresholds fixed now; matcher wiring is a SKELETON this cycle)
# ==========================================================================================

GATE_N_PERTURB_DEFAULT = 20        # D2: "섭동 20개"
GATE_SUCCESS_RATE_MIN = 0.90       # D2: 성공률 >= 90%
GATE_OUTLIER_RECALL_MIN = 0.80     # D2: outlier recall >= 80%


def try_import_matcher():
    """Returns the scan2bim.coarse_match module, or None if it does not exist yet
    (dev-core P1, not implemented as of this cycle)."""
    try:
        from scan2bim import coarse_match
        return coarse_match
    except ImportError:
        return None


def compute_success_rate(results: list) -> float:
    """SKELETON (not wired into main() yet): `results` = one
    scan2bim.plan_skeleton RESULT_FIELDS dict per perturbation. 'success' means
    status == 'ok' (finalize_match already enforces inlier_ratio/residual/margin
    internally -- a status=='ok' result already cleared plan_skeleton.DEFAULT_GATES).
    Gate: >= GATE_SUCCESS_RATE_MIN over GATE_N_PERTURB_DEFAULT perturbations."""
    n = len(results)
    if n == 0:
        return 0.0
    return sum(1 for r in results if r.get("status") == "ok") / n


def compute_outlier_recall(results: list, ground_truths: list) -> float:
    """SKELETON (raises NotImplementedError this cycle): recall of the TRUE changed
    segments (ground_truth['changed_output_indices'] from perturb_wall_segments)
    among the winning candidate's emitted 'outliers'. Needs a segment-index ->
    plan_event_id mapping that only exists once scan2bim/coarse_match.py (dev-core
    P1) and its use of plan_skeleton.plan_events() are implemented -- wiring this
    up without that mapping would mean inventing the answer, which this team's
    design invariant (refuse rather than fabricate) forbids.
    Gate: >= GATE_OUTLIER_RECALL_MIN."""
    raise NotImplementedError(
        "compute_outlier_recall needs scan2bim/coarse_match.py (segment -> plan_event_id "
        "mapping) -- not implemented as of the P0-Perturb cycle")


def verify_hold_on_ambiguous(result: dict) -> bool:
    """SKELETON (usable once results exist, not called from main() yet): a result
    whose top-2 candidate scores are within
    scan2bim.plan_skeleton.DEFAULT_GATES['margin_min'] MUST come back
    status=='hold', hold_reason=='ambiguous_margin', best=None. finalize_match()
    already enforces this SERVER-side; this is the CLIENT-side adversarial check
    that nothing downstream (tools/build_coplay.py --plan-match) auto-confirms a
    tie instead of surfacing the HOLD (P3 적대검증 scope)."""
    from scan2bim.plan_skeleton import DEFAULT_GATES
    margin = result.get("margin")
    if margin is not None and margin < DEFAULT_GATES["margin_min"]:
        return result.get("status") == "hold" and result.get("best") is None
    return True


# ==========================================================================================
# SECTION 4 -- CLI
# ==========================================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upload", type=str, default=None,
                    help="path to a real recon upload/session (e.g. realtime/_uploads/upload_XXXX) "
                         "to eventually match against the perturbed plan. Accepted but UNUSED this "
                         "cycle: no matcher exists yet to consume it (see exit 2 below).")
    ap.add_argument("--plan-dxf", type=Path, default=None,
                    help="real DXF plan (A-WALL/A-GLAZ layers) to perturb. Default: none exists in "
                         f"this repo (confirmed) -- falls back to the QA synthetic L-corridor "
                         f"(width={DEFAULT_CORRIDOR_WIDTH} m).")
    ap.add_argument("--n-perturb", type=int, default=GATE_N_PERTURB_DEFAULT,
                    help=f"perturbations in the D2 suite (default {GATE_N_PERTURB_DEFAULT}, the D2 gate size)")
    ap.add_argument("--seed", type=int, default=0,
                    help="base seed; same seed (+ same --n-perturb) -> same suite, reproducibly")
    ap.add_argument("--selftest-only", action="store_true",
                    help="run ONLY the reproducibility self-test (2x generation, compare) and exit; "
                         "skips suite generation and the matcher-availability check")
    ap.add_argument("--json", type=Path, default=None,
                    help="write the generated suite (segments + ground_truth per perturbation) as JSON")
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

    print(f"plan source: {plan_src}")
    print(f"base wall segments: N={len(base_segments)}")
    if args.upload is not None:
        print(f"--upload {args.upload} accepted but UNUSED this cycle (no matcher to feed it yet)")

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

    if args.json is not None:
        dump = [{"index": it["index"], "seed": it["seed"], "child_seed": it["child_seed"],
                 "segments": it["segments"].tolist(), "ground_truth": it["ground_truth"]}
                for it in suite]
        args.json.write_text(json.dumps(dump, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"suite written: {args.json}")

    matcher = try_import_matcher()
    if matcher is None:
        print("", file=sys.stderr)
        print("[매처 미구현] scan2bim/coarse_match.py 가 아직 존재하지 않습니다 -- "
              "D2 강건성 게이트(성공률>=%.0f%%, outlier recall>=%.0f%%, 모호 시 HOLD)를 "
              "평가할 매처가 없습니다. 이번 사이클(P0-Perturb) 범위는 섭동 생성기까지이며, "
              "매처 연결은 dev-core P1 이후입니다." % (GATE_SUCCESS_RATE_MIN * 100,
                                                   GATE_OUTLIER_RECALL_MIN * 100),
              file=sys.stderr)
        print("matcher_missing: import scan2bim.coarse_match failed (module not found)",
              file=sys.stderr)
        return 2

    print("[매처 발견되었으나 게이트 연결 미구현] scan2bim/coarse_match.py 는 존재하지만 "
          "compute_success_rate/compute_outlier_recall/verify_hold_on_ambiguous 를 실제 "
          "matcher 출력과 연결하는 배선은 아직 스켈레톤입니다 (다음 사이클).", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

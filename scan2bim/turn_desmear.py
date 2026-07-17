"""Turn de-smear: local heading redistribution (step-length-preserving re-integration).

Monocular VO can absorb a physical corridor turn NOT at the instant it happens
but smeared over several seconds afterward (feature-starved fast rotation ->
heading drifts/oscillates), so a rigidly-placed walk (tools/build_coplay.py::
place_rigid) keeps the right SCALE, straight legs and overall shape yet bends in
the wrong PLACE. A single rigid transform cannot fix this — the smear is a shape
distortion inside one ~9 s window, not a global rotation. This module rewrites
only that window's shape: it concentrates the net heading change back onto the
physical-turn pose while preserving every step's LENGTH and the trajectory
outside the window.

Method (b), "local heading re-distribution":
  * decompose the placed XZ path into per-step length L_i and heading theta_i;
  * the net heading change the recon spread across the smear window is the
    difference between the settled headings just before (psi_A) and just after
    (psi_B) it — used AS MEASURED, never snapped to 90 deg or any "clean" angle
    (snapping the walk to the model would violate the project's no-auto-confirm
    invariant); psi_B - psi_A is whatever the recon actually turned;
  * re-integrate the post-corner steps with the SAME lengths but a corrected
    heading so the whole turn happens as a step AT the physical corner index,
    with the smear window's local wiggle preserved as deviation about a
    ratcheted reference (monotone toward psi_B, so the post-turn leg and the
    end-hook tail are left rigid — never counter-rotated).

Guarantees (all asserted in tests/test_turn_desmear.py):
  * poses with index <= corner_idx are returned byte-identical (leg A untouched);
  * total path length (sum of step lengths) is preserved exactly;
  * the tail (post-window) is a rigid image of its input, so its straightness
    (max perpendicular deviation from its own line fit) never increases.

Auto window detection is deliberately conservative and has an explicit override
(window_start_idx / window_end_idx) because a single smeared gap is structurally
ambiguous to locate from heading noise alone.
"""
from __future__ import annotations

import numpy as np


def _angmean(angles: np.ndarray) -> float:
    """Circular mean of angles (radians)."""
    a = np.asarray(angles, dtype=np.float64)
    if a.size == 0:
        return 0.0
    return float(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))


def _wrap(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def _reintegrate_xz(xz, L, th, delta, ci):
    """XZ path with steps k>=ci re-headed by delta[k] (length L[k] kept), anchored
    at xz[ci] (poses <=ci untouched)."""
    out = xz.copy()
    thp = th + delta
    for k in range(ci, len(L)):
        out[k + 1] = out[k] + L[k] * np.array([np.cos(thp[k]), np.sin(thp[k])])
    return out


def _ratchet_delta(th, hs, ci, psiA, psiB, n):
    """Per-step heading correction delta[k] (k=0..n-2): 0 before the corner; after
    it, psiB minus a reference that ratchets monotonically from psiA toward psiB
    (clamped to [psiA,psiB]). Result: full (psiB-psiA) applied right at the corner,
    the residual bled off across the smear as the recon's own heading catches up,
    and exactly 0 once the recon has reached psiB — so the settled leg and tail
    stay rigid (delta=0) even where the raw heading over-rotates past psiB."""
    lo, hi = (psiB, psiA) if psiB < psiA else (psiA, psiB)
    delta = np.zeros(n - 1)
    run = psiA
    for k in range(ci, n - 1):
        run = min(run, hs[k]) if psiB < psiA else max(run, hs[k])
        ref = min(hi, max(lo, run))
        delta[k] = psiB - ref
    return delta


def desmear_turn(rigid_poses, corner_idx, window_start_idx=None, window_end_idx=None,
                 smooth=5, min_turn_angle_deg=20.0):
    """Redistribute a smeared turn onto its physical corner.

    rigid_poses: list of {"c":[x,y,z], "f":[...], "u":[...]} in model coords
        (metres, Y up), e.g. tools/build_coplay.py::place_rigid's output. Only the
        horizontal (XZ) shape is rewritten; Y (eye level + bob) is kept per pose.
    corner_idx: the physical-turn pose index (e.g. from --turn-time-s). Poses with
        index <= corner_idx are returned byte-identical.
    window_start_idx / window_end_idx (optional): explicit smear window (pose
        indices) — the reliable override when auto detection is unstable. psi_A is
        the settled heading just before window_start_idx, psi_B just after
        window_end_idx; net turn = psi_B - psi_A. When omitted, the window end is
        auto-picked so the RE-INTEGRATED corner (detected by
        scan2bim.pipe_path.trajectory_turn_fraction) coincides with the physical
        corner's arclength fraction — which is invariant to the redistribution
        because pre-corner steps and every step length are preserved.

    Returns (poses, meta). meta always carries applied / corner_idx /
    window_start_idx / window_end_idx / net_turn_deg. applied is False (poses
    returned unchanged) only when there is nothing to redistribute (too few poses,
    degenerate path, or no detectable turn)."""
    from scan2bim.pipe_path import trajectory_turn_fraction

    poses = list(rigid_poses)
    n = len(poses)
    meta = {"applied": False, "corner_idx": int(corner_idx),
            "window_start_idx": window_start_idx, "window_end_idx": window_end_idx,
            "net_turn_deg": 0.0}
    if n < 8:
        meta["reason"] = "too few poses"
        return [dict(p) for p in poses], meta

    c = np.array([p["c"] for p in poses], dtype=np.float64)
    f = np.array([p["f"] for p in poses], dtype=np.float64)
    u = np.array([p["u"] for p in poses], dtype=np.float64)
    xz = c[:, [0, 2]]
    Y = c[:, 1]
    d = np.diff(xz, axis=0)
    L = np.linalg.norm(d, axis=1)
    total = float(L.sum())
    if total < 1e-6:
        meta["reason"] = "degenerate path"
        return [dict(p) for p in poses], meta
    th = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    k = max(1, int(smooth))
    hs = np.convolve(th, np.ones(k) / k, mode="same") if k > 1 else th.copy()
    ci = int(np.clip(corner_idx, 1, n - 2))
    medL = float(np.median(L))
    valid = L > 0.3 * medL if medL > 0 else np.ones(len(L), bool)

    # turn direction + rough leg-A heading (only to locate the window/sign; the
    # net turn itself comes from the settled psi_A/psi_B below)
    psiA0 = _angmean(th[max(1, ci - 10):ci + 1])
    dev = hs - psiA0
    cand_all = [j for j in range(ci + 1, n - 1) if valid[j]]
    if not cand_all:
        meta["reason"] = "no valid post-corner steps"
        return [dict(p) for p in poses], meta
    kstar = max(cand_all, key=lambda j: abs(dev[j]))
    sgn = float(np.sign(dev[kstar])) or 1.0

    # window start (departure from leg A) — mainly for psi_A + reporting
    if window_start_idx is not None:
        ws = int(np.clip(window_start_idx, ci, n - 2))
    else:
        ws = next((j for j in range(ci, n - 1) if sgn * dev[j] >= np.radians(15.0)), ci)
    psiA = _angmean(th[max(1, ws - 8):ws + 1])
    arclen = np.concatenate([[0.0], np.cumsum(L)])
    target_frac = float(arclen[ci] / total)

    def _psiB_at(we):
        seg = [j for j in range(max(ws + 1, we - 3), min(n - 1, we + 4)) if valid[j]]
        return _angmean(th[seg]) if seg else float(th[int(np.clip(we, 0, n - 2))])

    if window_end_idx is not None:
        we = int(np.clip(window_end_idx, ws + 1, n - 2))
        psiB = _psiB_at(we)
    else:
        # pick the window end whose re-integrated corner lands on the physical
        # corner's (invariant) arclength fraction — with a real (>min angle) bend.
        # Ties (corner already on target) prefer the LARGER net turn, so a clean
        # (non-over-rotated) turn is captured fully instead of truncated early.
        best = None
        for we_c in range(ws + 3, n - 2):
            if not valid[we_c]:
                continue
            pB = _psiB_at(we_c)
            net_c = _wrap(pB - psiA)
            if abs(net_c) < np.radians(min_turn_angle_deg):
                continue
            delta_c = _ratchet_delta(th, hs, ci, psiA, pB, n)
            nxz = _reintegrate_xz(xz, L, th, delta_c, ci)
            frac_c, ang_c = trajectory_turn_fraction(nxz)
            if ang_c < min_turn_angle_deg:
                continue
            score = (round(abs(frac_c - target_frac), 2), -abs(net_c))
            if best is None or score < best[0]:
                best = (score, we_c, pB)
        if best is None:
            meta["reason"] = "no window end yields a clean corner"
            return [dict(p) for p in poses], meta
        _score, we, psiB = best

    net = _wrap(psiB - psiA)
    delta = _ratchet_delta(th, hs, ci, psiA, psiB, n)
    new_xz = _reintegrate_xz(xz, L, th, delta, ci)

    out = []
    for i in range(n):
        if i <= ci:
            out.append(dict(poses[i]))                      # byte-identical leg A
            continue
        drot = float(delta[min(i, n - 2)])                  # rotate look dir with the step it leaves on
        cs, sn = np.cos(drot), np.sin(drot)
        fx, fz = f[i, 0], f[i, 2]
        ux, uz = u[i, 0], u[i, 2]
        out.append({
            "c": [round(float(new_xz[i, 0]), 3), round(float(Y[i]), 3), round(float(new_xz[i, 1]), 3)],
            "f": [round(float(cs * fx - sn * fz), 4), round(float(f[i, 1]), 4), round(float(sn * fx + cs * fz), 4)],
            "u": [round(float(cs * ux - sn * uz), 4), round(float(u[i, 1]), 4), round(float(sn * ux + cs * uz), 4)],
        })

    meta.update({"applied": True, "window_start_idx": int(ws), "window_end_idx": int(we),
                 "net_turn_deg": round(float(np.degrees(net)), 2),
                 "psi_a_deg": round(float(np.degrees(psiA)), 2),
                 "psi_b_deg": round(float(np.degrees(psiB)), 2),
                 "target_arclen_frac": round(target_frac, 4)})
    return out, meta

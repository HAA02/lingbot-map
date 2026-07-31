"""DOOR-PASSING DETECTION FROM THE RGB VIDEO ALONE (no cloud, no plan).

WHY A SECOND MODALITY (measured, cycle 10 -> 11)
------------------------------------------------
`door_detect` reads the door event off the recon POINT CLOUD (lateral free-space
contraction). It passes 36/36 synthetic checks and returns 0 detections on
realtime/_uploads/upload_1781521406685, for a reason it diagnoses itself:

    median measured passage width 0.0107 < 0.6 x calibrated w0 0.1801
    "cloud likely carries points ON the camera path (monocular smear)"

independently confirmed: the median distance from the walk to the nearest cloud point is
0.0503 recon units and the 10 % quantile 0.0010, against a 0.0394 pose step — the cloud
carries points ON the trajectory, so it does not encode the lateral free space at all.
No threshold change can recover a quantity the data does not contain. This module
therefore measures the SAME physical event (the camera crossing a door plane) from a
channel that upload does have: the video.

CIRCULARITY IS STILL FORBIDDEN
------------------------------
As in `door_detect`, nothing here may read the DXF, the BIM, `plan_skeleton` or a
`coarse_match` result — door evidence derived from the drawing could not then be used to
disambiguate the drawing. The inputs are ONLY the video frames plus, optionally, the
recon's own poses (for the arclength axis and an extra turn diagnostic). No learned
weights and no network access: OpenCV gradients, 1-D peak tracking, least squares.

THE OBSERVABLE (aperture crossing = symmetric divergence about the focus of expansion)
--------------------------------------------------------------------------------------
Walk straight at speed v past a STATIC vertical edge whose perpendicular offset from the
path is X and which the camera crosses at time T. Its image column obeys

    x(t) = foe + K / (T - t),        K = fx * X / v                            (1)

a Mobius function of time: it drifts away from the focus of expansion, accelerates, and
leaves the frame just before T. Three quantities fall out of a 3-parameter linear fit of
(1) — the CROSSING TIME T, the FOCUS OF EXPANSION foe, and K, whose sign is the side the
edge passes on and whose magnitude is |X|/v in units of fx (seconds of walking).

A DOORWAY the camera WALKS THROUGH is then, and only then:

  (a) TWO such edges — the jambs — one with K<0 and one with K>0, i.e. the camera passes
      one on its left and the other on its right. This is the visual twin of
      `door_detect`'s two-sided contraction gate, and it is what kills the whole
      "passed alongside" family at once: a column, a window, a poster or a door-shaped
      wall decoration is passed on ONE side, so both of its vertical edges carry the SAME
      sign of K. Nothing that is not an opening can be straddled.
  (b) with the SAME crossing time (they are two edges of one frame, one plane), and
  (c) with a LINTEL: a transverse horizontal member between the two jambs whose image row
      obeys (1) as well, rising out of the top of the frame at that same T. This is the
      gate that separates a door from any other pair of opposite-side features at equal
      depth — two facing wall posters, or the corridor's own opposite walls, straddle the
      path and cross together, but nothing spans between them. Longitudinal edges (the
      corridor's ceiling line, a skirting) cannot fake it: they run to a STATIONARY
      vanishing point, so their row does not diverge at any finite T and (1) does not fit.
  (d) with the walk locally STRAIGHT: yaw translates the whole image, which breaks (1)
      and moves the focus of expansion. Measured frame to frame from the profile itself
      (so the gate exists on video-only input), a big turn is a hard reject, a small one a
      confidence discount — the same soft/hard pair `door_detect` uses.
  (e) and the divergence must COMPLETE: both jambs must actually leave the frame. A dead
      end (a door-shaped end wall you stop in front of) produces the approach and never
      the exit, and its fitted T falls outside the extrapolation window.

WHAT IT CANNOT DO
-----------------
* Same scope as `door_detect`: it detects doors WALKED THROUGH. A door in a side wall,
  open or shut, is passed alongside and is rejected by (a) BY DESIGN.
* It needs the jambs to be IN FRAME. MEASURED on upload_1781521406685 from its own
  poses: the camera is carried pitched 15.3 deg up (p10 10.9, p90 34.0) — with a phone's
  ~39 deg vertical field that puts the bottom of the frame near -5 deg elevation, so the
  video is ceiling and upper wall, and a jamb is only seen over its top ~1.5 m. That is
  why the column profile is taken over the LOWER rows (`row_band_frac`) and why
  `info['view']` reports the profile's own mass distribution: a rig aimed higher still
  can put the whole opening out of frame, and then this channel is blind too, which the
  diagnostics must say instead of returning silence.
* Everything except the WIDTH estimate and the turn gate is independent of the lens: T,
  foe and every hard gate come out of (1) in pixels. `hfov_deg` converts K to metres/
  seconds-of-walking and pixel shift to degrees, and is an assumption, so it is reported.

REFUSAL PHILOSOPHY (inherited from door_detect / estimate_plan_transform)
-------------------------------------------------------------------------
Every pair that reaches the scoring stage is RETURNED in `candidates` with the gate it
failed; `doors` holds only the accepted ones and is empty rather than optimistic. A
fabricated door time would corrupt the very ambiguity it exists to resolve.

CONTRACT
--------
`detect_doors` returns the `door_detect` shape ({"schema","doors","candidates","profile",
"info"}), and `door_times` / `door_s_values` are the same helpers, so a result drops
straight into `coarse_match.door_arclengths(traj, times, door_times(result))` and into
build_coplay's `--door-times` wiring.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# Physical door/corridor dimensions live in ONE place; this module is the second reader.
from .door_detect import CORRIDOR_CLEAR_M, DOOR_CLEAR_M  # noqa: F401  (documented reuse)

SCHEMA = "coplay.door_detect_rgb/1.0"

#: Ratio band a walked opening must occupy on its corridor, DOOR_CLEAR_M /
#: CORRIDOR_CLEAR_M widened exactly as in door_detect. Only used when the caller supplies
#: a `width_hint` (a corridor width in recon units); without it the module reports the
#: measured width and gates nothing on it, because it has no metric reference.
WIDTH_RATIO_BAND = (0.30, 0.70)

DEFAULTS = {
    # ---- frame preparation ---------------------------------------------------------
    # Work width. A 0.9 m opening seen from 3 m subtends 0.30 rad = 0.46 of a 65 deg
    # frame, i.e. 220 px here, and the jamb peak localises to ~1 px = 0.4 % of the frame.
    # Full 1920 buys nothing and costs 16x the gradient.
    "work_width": 480,
    "blur_sigma": 1.0,
    # Rows the vertical-edge column profile is summed over, as fractions of the frame
    # height. The upper rows are CEILING for any hand-carried rig (measured 15 deg up-tilt
    # on the real upload) and a camera pitched up also splays vertical world lines toward
    # a vanishing point far BELOW the frame, so a tall band would smear a jamb peak
    # (drift = (x-cx) * rows / 5627 px at that pitch: 14 px over 270 rows at x-cx = 300).
    # The lower 55 % is where jambs, columns and wall edges are, and it is short enough to
    # keep that smear near 3 work-pixels.
    "row_band_frac": (0.45, 0.98),
    # 1-D smoothing of the column profile, in work pixels (a jamb is 2-4 px wide here).
    "profile_smooth_px": 2.5,

    # ---- vertical structure peaks ---------------------------------------------------
    # Height and prominence in units of the frame's OWN median column mass, so exposure,
    # texture and lens changes do not move the bar.
    "peak_min_height": 1.2,
    "peak_prominence": 0.35,
    "max_peaks": 30,

    # ---- tracking -------------------------------------------------------------------
    "track_gate_px": 5.0,      # base association radius (work px)
    "track_gate_vel": 1.5,     # ...plus this multiple of the predicted per-frame motion
    "track_max_gap": 3,        # frames a track may go unmatched before it is closed
    "min_track_frames": 8,     # a Mobius fit has 3 parameters; 8 samples is the floor
    "min_x_span_frac": 0.10,   # of the work width: a track that barely moves cannot date
    # the last sample must lie this close to the left or right border for the divergence
    # to count as COMPLETED (gate (e)).
    "exit_border_frac": 0.12,
    "mono_frac": 0.80,         # fraction of steps that must increase |x - foe|
    "min_r2": 0.90,            # of the Mobius fit (1) — this is the "static edge" test
    # T must sit after the last sample (the edge leaves the frame BEFORE the crossing) and
    # within this multiple of the track's own duration — a scale-free extrapolation limit.
    "extrap_frac": 2.0,
    "back_frac": 0.25,

    # ---- pairing --------------------------------------------------------------------
    "pair_t_frac": 0.20,       # |T_L - T_R| limit, as a fraction of the mean duration
    "pair_t_min_s": 0.10,      # ...never tighter than this (frame quantisation)
    "min_overlap_frac": 0.35,  # the two jambs must be seen TOGETHER this much
    "sym_band": (0.20, 0.80),  # |X_left| / (|X_left|+|X_right|): the walker need not be
    #                            centred in the opening, but 4:1 is not a doorway
    "width_ratio_band": WIDTH_RATIO_BAND,

    # ---- lintel ---------------------------------------------------------------------
    "lintel_rows": 96,         # coarse horizontal-edge map kept for every frame: 96 rows
    "lintel_cols": 64,         # x 64 cols x 4 B x 1060 frames = 26 MB on the real upload
    "lintel_min_frames": 6,
    # |T_lintel - T_jambs| limit, as a fraction of the pair's duration. MEASURED on the
    # one-door fixture: the door's own head agrees to 0.04 s, while the head BORROWED by
    # the wall-stripe pair 0.7 m further on disagrees by 0.73 s — 0.15 separates them.
    "lintel_t_frac": 0.15,
    # The member must SPAN the jamb pair. MEASURED necessity: without these two numbers
    # every pair of opposite corridor wall stripes in tests/test_door_detect_rgb.py's
    # corridor (1.3 m apart, straddling, perfectly coincident crossing times) borrowed the
    # real doorway's lintel and was accepted — 3 false doors on the one-door fixture at
    # confidence 0.86-0.98. A member is only evidence for the pair it actually connects.
    "lintel_cover_frac": 0.55,      # of the inter-jamb span, at the member's own row
    "lintel_overhang_frac": 0.60,   # how far it may continue past the jambs, same units
    # rows (of `lintel_rows`) the span test may look across: a transverse member slants
    # in the image the moment the camera is carried off the direction of travel.
    "lintel_row_win": 3,

    # ---- arrival (gate (f)): the walk must actually REACH the crossing plane ---------
    # An edge at lateral offset X leaves a hfov frame Z = X/tan(hfov/2) BEFORE its plane
    # is crossed — 1.43 m for a corridor wall on a 65 deg lens. So "it left the frame" is
    # NOT "we went through": MEASURED on the dead-end fixture, the corridor/end-wall
    # corners diverge out of the frame while the camera is still 1.4 m short, stop, and
    # were accepted at confidence 0.91. The walk must therefore still be MOVING between
    # the last observation and T, which the profile's own expansion rate reports.
    "arrival_frac": 0.35,           # of the pair's own expansion rate, in the gap
    "arrival_min_frames": 2,        # frames that must exist past T to witness it at all

    # ---- turn gates (same convention and numbers as door_detect) --------------------
    "turn_soft_deg": 12.0,
    "turn_hard_deg": 35.0,

    # ---- lens assumption (affects the WIDTH estimate and the turn gate only) --------
    "hfov_deg": 65.0,

    # ---- acceptance -----------------------------------------------------------------
    "trust_frames": 20,        # pair frames for full density trust
    # MEASURED over tests/test_door_detect_rgb.py's synthetic suite: true doorways score
    # 0.922-0.939 (clean, two-door, noisy and 8 deg-yaw fixtures) and every adversarial
    # candidate scores 0.000 because a HARD gate fires. That gap is NOT this threshold's
    # doing and the number must not be read as if it were: with the gates switched off,
    # the dead end's SHAPE scores 0.910 — indistinguishable from a real door, because a
    # door-shaped frame you stop in front of looks exactly like one until you ask whether
    # the walk arrived. 0.45 only catches the degraded-but-ungated middle.
    "min_confidence": 0.45,
}


# --------------------------------------------------------------------------------------
# SECTION 1 — parameters, frames, profiles
# --------------------------------------------------------------------------------------

def _params(over=None) -> dict:
    p = dict(DEFAULTS)
    for k, v in (over or {}).items():
        if k not in DEFAULTS:
            raise ValueError(f"unknown door_detect_rgb parameter {k!r}")
        p[k] = v
    return p


def _open(video, fps=None, max_frames=None):
    """(fps, frame iterator). `video` is a path, or an (N,H,W[,3]) array / frame list."""
    if isinstance(video, (str, Path)):
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise ValueError(f"cannot open video {video}")
        f = float(fps) if fps else float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not np.isfinite(f) or f <= 0:
            cap.release()
            raise ValueError(f"cannot determine fps for {video} — pass fps=")

        def it():
            i = 0
            try:
                while True:
                    ok, fr = cap.read()
                    if not ok:
                        break
                    yield fr
                    i += 1
                    if max_frames and i >= max_frames:
                        break
            finally:
                cap.release()
        return f, it()
    frames = list(video) if not isinstance(video, np.ndarray) else list(video)
    if fps is None:
        raise ValueError("fps= is required when frames are passed directly")
    if max_frames:
        frames = frames[:int(max_frames)]
    return float(fps), iter(frames)


def frame_profiles(video, fps=None, params=None, max_frames=None) -> dict:
    """Per-frame observables: the vertical-edge COLUMN profile (where jambs live), a
    coarse horizontal-edge map (where the lintel lives) and the frame-to-frame global
    horizontal shift (the yaw proxy).

    Returns {"fps","t","col","hmap","shift_px","size","info"} with
      col       (N, work_width)  vertical-edge mass per column, each frame normalised by
                                 its OWN median so exposure cannot move the peak bar;
      hmap      (N, lintel_rows, lintel_cols) horizontal-edge mass, box-averaged;
      shift_px  (N,) shift of frame i against i-1, by 1-D correlation of `col`.
    """
    p = _params(params)
    W = int(p["work_width"])
    R, C = int(p["lintel_rows"]), int(p["lintel_cols"])
    fps, frames = _open(video, fps, max_frames)
    cols, hmaps = [], []
    H = None
    for fr in frames:
        a = np.asarray(fr)
        if a.ndim == 3:
            a = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        a = a.astype(np.uint8) if a.dtype != np.uint8 else a
        if H is None:
            H = max(8, int(round(W * a.shape[0] / max(a.shape[1], 1))))
        g = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(g, (0, 0), float(p["blur_sigma"]))
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        agx, agy = np.abs(gx), np.abs(gy)
        vert = np.maximum(agx - agy, 0.0)         # near-vertical edges only
        horz = np.maximum(agy - agx, 0.0)         # near-horizontal edges only
        r0 = int(round(p["row_band_frac"][0] * H))
        r1 = int(round(p["row_band_frac"][1] * H))
        r0, r1 = max(0, min(r0, H - 2)), max(1, min(r1, H))
        v = vert[r0:r1].sum(axis=0)
        v = gaussian_filter1d(v, float(p["profile_smooth_px"]))
        # Normalise by the frame's OWN edge mass, so gain/exposure cannot move the peak
        # bar. The scale is the median OR a quarter of the mean, whichever is larger:
        # MEASURED defect — on a bare end wall more than half the columns carry no
        # vertical edge at all, the median is exactly 0, and dividing by it zeroed the
        # entire profile (0 tracks, so the dead-end fixture tested nothing). The mean is
        # zero only for a frame with no vertical structure whatsoever.
        scale = max(float(np.median(v)), 0.25 * float(v.mean()))
        cols.append(v / scale if scale > 1e-9 else np.zeros_like(v))
        hmaps.append(cv2.resize(horz, (C, R), interpolation=cv2.INTER_AREA))
    n = len(cols)
    info = {"n_frames": int(n), "fps": float(fps),
            "work_size": [int(W), int(H or 0)],
            "row_band_px": [int(round(p["row_band_frac"][0] * (H or 0))),
                            int(round(p["row_band_frac"][1] * (H or 0)))]}
    if n == 0:
        info["fail"] = "video carried no frames"
        return {"fps": float(fps), "t": np.zeros(0), "col": np.zeros((0, W)),
                "hmap": np.zeros((0, R, C), np.float32), "shift_px": np.zeros(0),
                "size": (W, H or 0), "info": info}
    col = np.asarray(cols, dtype=np.float32)
    hmap = np.asarray(hmaps, dtype=np.float32)
    t = np.arange(n, dtype=np.float64) / float(fps)
    shift = _global_shift(col, max_shift=max(4, W // 12))
    info["mean_col_mass"] = round(float(col.mean()), 4)
    return {"fps": float(fps), "t": t, "col": col, "hmap": hmap, "shift_px": shift,
            "size": (W, int(H)), "info": info}


def _global_shift(col: np.ndarray, max_shift: int) -> np.ndarray:
    """Frame-to-frame horizontal shift of the whole column profile, in work pixels.

    A yaw of theta translates every image feature by ~fx*theta; forward motion expands the
    profile SYMMETRICALLY about the focus of expansion and so leaves the correlation peak
    at zero. That makes this the video-only rotation observable (gate (d))."""
    n = len(col)
    out = np.zeros(n)
    if n < 2:
        return out
    a = col - col.mean(axis=1, keepdims=True)
    for i in range(1, n):
        best, bs = -np.inf, 0
        for s in range(-max_shift, max_shift + 1):
            x = a[i - 1] if s == 0 else (np.roll(a[i - 1], s))
            if s > 0:
                v = float(x[s:] @ a[i][s:])
            elif s < 0:
                v = float(x[:s] @ a[i][:s])
            else:
                v = float(x @ a[i])
            if v > best:
                best, bs = v, s
        out[i] = float(bs)
    return out


# --------------------------------------------------------------------------------------
# SECTION 2 — peak tracking and the Mobius (time-to-crossing) fit
# --------------------------------------------------------------------------------------

def _peaks(prof: np.ndarray, min_h: float, prom: float, max_peaks: int) -> np.ndarray:
    """Sub-pixel positions of the vertical structures in one profile."""
    idx, props = find_peaks(prof, height=min_h, prominence=prom)
    if not len(idx):
        return np.zeros(0)
    if len(idx) > max_peaks:                       # keep the strongest
        keep = np.argsort(props["prominences"])[::-1][:max_peaks]
        idx = np.sort(idx[keep])
    out = []
    for j in idx:
        if 0 < j < len(prof) - 1:
            a, b, c = float(prof[j - 1]), float(prof[j]), float(prof[j + 1])
            den = a - 2 * b + c
            out.append(j + (0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0))
        else:
            out.append(float(j))
    return np.asarray(out, dtype=np.float64)


def track_peaks(prof: np.ndarray, gate: float, gate_vel: float, max_gap: int,
                min_h: float, prom: float, max_peaks: int, min_len: int) -> list:
    """Follow profile peaks through time.

    Nearest-neighbour association against a CONSTANT-VELOCITY prediction: a diverging
    jamb accelerates (eq. 1), so a fixed radius would drop it exactly when the evidence
    gets strongest. The assignment is greedy over ALL (track, peak) pairs by cost, not in
    track order — MEASURED reason: a fast wall feature sweeping past a slower jamb claimed
    the jamb's peak whenever it happened to be listed first, and the hijacked track then
    failed its own Mobius fit (r2 0.76), losing the door's left jamb entirely.
    Returns [{"i": frame indices, "x": positions, "gaps": int}, ...] with tracks shorter
    than `min_len` observations dropped."""
    n = len(prof)
    active: list = []
    done: list = []
    for f in range(n):
        pk = _peaks(prof[f], min_h, prom, max_peaks)
        used = np.zeros(len(pk), dtype=bool)
        taken = np.zeros(len(active), dtype=bool)
        cost = []
        for a, tr in enumerate(active):
            dt = max(1, f - tr["i"][-1])
            pred = tr["x"][-1] + tr["v"] * (f - tr["i"][-1])
            r = gate + gate_vel * abs(tr["v"]) * dt
            for j in range(len(pk)):
                d = abs(pk[j] - pred)
                if d <= r:
                    cost.append((d, a, j))
        for d, a, j in sorted(cost):
            if taken[a] or used[j]:
                continue
            tr = active[a]
            dt = f - tr["i"][-1]
            v_new = (pk[j] - tr["x"][-1]) / max(dt, 1)
            tr["v"] = 0.5 * tr["v"] + 0.5 * v_new if len(tr["i"]) > 1 else v_new
            tr["i"].append(f)
            tr["x"].append(float(pk[j]))
            taken[a], used[j] = True, True
        nxt = []
        for a, tr in enumerate(active):
            if taken[a]:
                nxt.append(tr)
            elif f - tr["i"][-1] <= max_gap:
                tr["gaps"] += 1
                nxt.append(tr)
            else:
                done.append(tr)
        for j in range(len(pk)):
            if not used[j]:
                nxt.append({"i": [f], "x": [float(pk[j])], "v": 0.0, "gaps": 0})
        active = nxt
    done.extend(active)
    out = []
    for tr in done:
        if len(tr["i"]) >= int(min_len):
            out.append({"i": np.asarray(tr["i"], dtype=int),
                        "x": np.asarray(tr["x"], dtype=np.float64),
                        "gaps": int(tr["gaps"])})
    out.sort(key=lambda t: (t["i"][0], t["x"][0]))
    return out


def mobius_fit(t: np.ndarray, x: np.ndarray, trim: float = 0.2) -> dict:
    """Least squares for x(t) = foe + K/(T - t)  (eq. 1), then one trimmed refit.

    Multiplying out gives x*T + foe*t - (foe*T + K) = x*t, which is LINEAR in the three
    unknowns (T, foe, g = foe*T + K) — so the crossing time comes out of a 3x3 normal
    equation with no initial guess and no iteration. The refit drops the worst `trim`
    fraction of residuals, which is what makes a tracker hiccup (a frame where the peak
    jumped to a neighbouring structure) survivable.

    Returns {"T","foe","K","r2","n"}; r2 is NaN when the system is degenerate."""
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    bad = {"T": float("nan"), "foe": float("nan"), "K": float("nan"),
           "r2": float("nan"), "n": int(len(t))}
    if len(t) < 4:
        return bad

    def solve(m):
        A = np.stack([x[m], t[m], -np.ones(int(m.sum()))], axis=1)
        b = x[m] * t[m]
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None
        T, foe, g = [float(v) for v in sol]
        if not np.isfinite([T, foe, g]).all():
            return None
        return T, foe, g

    m = np.ones(len(t), dtype=bool)
    s = solve(m)
    if s is None:
        return bad
    for _ in range(2):
        T, foe, g = s
        den = T - t
        den = np.where(np.abs(den) < 1e-6, np.sign(den) * 1e-6 + 1e-12, den)
        pred = foe + (g - foe * T) / den
        res = np.abs(pred - x)
        k = int(np.floor((1.0 - trim) * len(t)))
        if k < 4:
            break
        m = np.zeros(len(t), dtype=bool)
        m[np.argsort(res)[:k]] = True
        s2 = solve(m)
        if s2 is None:
            break
        s = s2
    T, foe, g = s
    K = g - foe * T
    den = T - t
    den = np.where(np.abs(den) < 1e-6, np.sign(den) * 1e-6 + 1e-12, den)
    pred = foe + K / den
    ss_res = float(np.sum((pred - x) ** 2))
    ss_tot = float(np.sum((x - x.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return {"T": float(T), "foe": float(foe), "K": float(K), "r2": float(r2),
            "n": int(len(t))}


def expansion_rate(tracks: list, t: np.ndarray, foe: float, W: int) -> np.ndarray:
    """Per-frame FORWARD-MOTION rate of the image, from the peak tracks themselves.

    For any static feature, eq. (1) gives d/dt log|x - foe| = 1/(T - t) — a positive
    number that grows as the crossing approaches and is ZERO when the camera stops. The
    median over the features alive in a frame is therefore a video-only "am I still
    walking" signal, in units of 1/s, and it is what tells a doorway the walk went
    through from a doorway the walk stopped in front of (gate (f)). Frames with no
    usable feature get NaN, never 0 — "unknown" and "stopped" must not be confused."""
    n = len(t)
    acc = [[] for _ in range(n)]
    lo = 0.15 * 0.5 * W                      # ignore features sitting on the focus point
    for tr in tracks:
        i, x = tr["i"], tr["x"]
        for k in range(len(i) - 1):
            dt = float(t[i[k + 1]] - t[i[k]])
            if dt <= 1e-9:
                continue
            mid = 0.5 * (x[k] + x[k + 1]) - foe
            if abs(mid) < lo:
                continue
            r = float((x[k + 1] - x[k]) / (mid * dt))
            for f in range(i[k], i[k + 1]):
                acc[f].append(r)
    out = np.full(n, np.nan)
    for f in range(n):
        if acc[f]:
            out[f] = float(np.median(acc[f]))
    return out


def _diverging(tr: dict, t: np.ndarray, W: int, p: dict) -> dict:
    """Turn one peak track into a crossing candidate, or say why it is not one."""
    ti, xi = t[tr["i"]], tr["x"]
    fit = mobius_fit(ti, xi)
    out = {"t0": float(ti[0]), "t1": float(ti[-1]), "n": int(len(ti)),
           "x0": float(xi[0]), "x1": float(xi[-1]),
           "span": float(xi.max() - xi.min()), **fit, "reject": []}
    dur = float(ti[-1] - ti[0])
    out["duration"] = dur
    if out["span"] < p["min_x_span_frac"] * W:
        out["reject"].append("static_edge")
    b = p["exit_border_frac"] * W
    if not (xi[-1] <= b or xi[-1] >= W - b):
        out["reject"].append("no_frame_exit")
    if not np.isfinite(fit["r2"]) or fit["r2"] < p["min_r2"]:
        out["reject"].append("poor_mobius_fit")
    T, foe = fit["T"], fit["foe"]
    if not np.isfinite(T):
        out["reject"].append("no_crossing_time")
    else:
        if T > ti[-1] + p["extrap_frac"] * max(dur, 1e-9):
            out["reject"].append("crossing_too_far")     # dead end: never completed
        if T < ti[-1] - p["back_frac"] * max(dur, 1e-9):
            out["reject"].append("crossing_in_past")
    if np.isfinite(foe):
        d = np.abs(xi - foe)
        out["mono"] = float(np.mean(np.diff(d) > 0)) if len(d) > 1 else 0.0
        if out["mono"] < p["mono_frac"]:
            out["reject"].append("not_diverging")
    else:
        out["mono"] = 0.0
        out["reject"].append("not_diverging")
    out["side"] = int(np.sign(fit["K"])) if np.isfinite(fit["K"]) else 0
    out["ok"] = not out["reject"]
    return out


# --------------------------------------------------------------------------------------
# SECTION 3 — the lintel (transverse member between the jambs)
# --------------------------------------------------------------------------------------

def _span(row_mass: np.ndarray, c0: int, c1: int) -> tuple:
    """(coverage, overhang) of a horizontal member across the column band [c0, c1).

    coverage = share of the band that carries edge mass at this row; overhang = how far
    the same run continues OUTSIDE the band, in units of the band width. A door head
    covers its opening and stops at the jambs (1.0, ~0.0); the partition-to-ceiling line
    above it, or any longitudinal edge, either fails to fill the band or runs straight
    past it. `row_mass` is a max over a few rows, because a transverse member is only
    horizontal in the image when the camera looks straight down the walk — MEASURED: at
    8 deg of carry yaw the door head slants by 2.4 coarse rows across its own span and a
    single-row test read coverage 0.39 on a true doorway."""
    n = len(row_mass)
    c0, c1 = max(0, min(c0, n - 1)), max(1, min(c1, n))
    if c1 - c0 < 2:
        return 0.0, 0.0
    band = row_mass[c0:c1]
    thr = max(0.5 * float(band.max()), 2.0 * float(np.median(row_mass)))
    if thr <= 1e-9:
        return 0.0, 0.0
    on = row_mass > thr
    cov = float(np.mean(on[c0:c1]))
    over = 0
    j = c1
    while j < n and on[j]:
        over += 1
        j += 1
    j = c0 - 1
    while j >= 0 and on[j]:
        over += 1
        j -= 1
    return cov, float(over) / float(c1 - c0)


def _lintel(hmap: np.ndarray, frames: np.ndarray, xl: np.ndarray, xr: np.ndarray,
            t: np.ndarray, W: int, p: dict) -> dict:
    """Look for a horizontal member SPANNING the two jambs whose row diverges at the same T.

    Same eq. (1), applied to the row: a transverse member above eye level rises and leaves
    the top of the frame at the crossing. A LONGITUDINAL edge (ceiling line, skirting)
    runs to a stationary vanishing point, so no finite T fits it. And a transverse member
    that does not actually connect THESE two jambs (the door head one aperture further in,
    the partition's own line at the ceiling) is filtered by `_span`: without that test a
    pair of opposite wall stripes simply borrows the nearest door's lintel."""
    R, C = hmap.shape[1], hmap.shape[2]
    out = {"found": False, "T": None, "r2": None, "n": 0, "coverage": None,
           "overhang": None, "reason": None}
    if len(frames) < int(p["lintel_min_frames"]):
        out["reason"] = "pair_too_short"
        return out
    prof = np.zeros((len(frames), R), dtype=np.float32)
    bands = []
    for k, f in enumerate(frames):
        c0 = int(np.floor(min(xl[k], xr[k]) / W * C))
        c1 = int(np.ceil(max(xl[k], xr[k]) / W * C))
        c0, c1 = max(0, min(c0, C - 1)), max(1, min(c1, C))
        if c1 - c0 < 1:
            c1 = c0 + 1
        bands.append((c0, c1))
        v = hmap[f, :, c0:c1].mean(axis=1)
        med = float(np.median(v))
        prof[k] = v / med if med > 1e-9 else v
    trs = track_peaks(prof, gate=1.5, gate_vel=p["track_gate_vel"],
                      max_gap=int(p["track_max_gap"]), min_h=p["peak_min_height"],
                      prom=p["peak_prominence"], max_peaks=8,
                      min_len=int(p["lintel_min_frames"]))
    if not trs:
        out["reason"] = "no_horizontal_member"
        return out
    tt = t[frames]
    best, saw_rising = None, False
    for tr in trs:
        fit = mobius_fit(tt[tr["i"]], tr["x"])
        if not np.isfinite(fit["r2"]) or fit["r2"] < p["min_r2"]:
            continue
        if not (np.isfinite(fit["T"]) and np.isfinite(fit["K"])) or fit["K"] >= 0:
            continue                 # the row must go UP (toward row 0) as t -> T
        saw_rising = True
        cov, over = [], []
        win = int(p["lintel_row_win"])
        for k, r in zip(tr["i"], tr["x"]):
            r0 = int(np.clip(round(r) - win, 0, R - 1))
            r1 = int(np.clip(round(r) + win + 1, 1, R))
            c, o = _span(hmap[frames[k], r0:r1].max(axis=0), *bands[k])
            cov.append(c)
            over.append(o)
        cov_m, over_m = float(np.median(cov)), float(np.median(over))
        if cov_m < p["lintel_cover_frac"] or over_m > p["lintel_overhang_frac"]:
            continue
        score = fit["r2"] * cov_m / (1.0 + over_m)
        if best is None or score > best["score"]:
            best = {**fit, "n": int(len(tr["i"])), "coverage": cov_m,
                    "overhang": over_m, "score": score}
    if best is None:
        out["reason"] = ("no_member_spanning_the_pair" if saw_rising
                         else "no_rising_transverse_member")
        return out
    out.update({"found": True, "T": float(best["T"]), "r2": float(best["r2"]),
                "n": int(best["n"]), "coverage": round(best["coverage"], 3),
                "overhang": round(best["overhang"], 3)})
    return out


# --------------------------------------------------------------------------------------
# SECTION 4 — the entry point
# --------------------------------------------------------------------------------------

def _score_tri(x, lo, hi) -> float:
    if not np.isfinite(x) or x <= lo or x >= hi:
        return 0.0
    mid = 0.5 * (lo + hi)
    return float(1.0 - abs(x - mid) / (0.5 * (hi - lo)))


def detect_doors(video, fps=None, pose_times=None, traj_xz=None, width_hint=None,
                 params=None, keep_profile: bool = True, max_frames=None) -> dict:
    """Detect door-PASSING events in an RGB walk-through video. See the module docstring.

    video       path to a video, or an (N,H,W[,3]) frame array / list (then pass fps=).
    pose_times  (P,) time axis of the recon poses (build_coplay uses t_i = duration*i/(P-1));
                with `traj_xz` it fills each detection's arclength `s` and adds a
                pose-derived turn diagnostic. NEITHER is required: the detector is
                video-only and the turn GATE always uses the image-measured shift.
    traj_xz     (P,2) gravity-aligned recon walk, recon units.
    width_hint  corridor clear width in RECON units (a scale prior, never plan geometry).
                Only with it can the measured opening width be gated as a RATIO.

    Returns the `door_detect` shape:
      {"schema","doors","candidates","profile","info"} where candidate =
      {"time_s","frame","s","confidence","accepted","reject","scores","t_cross_left",
       "t_cross_right","foe_px","width_px_s","width_recon","width_ratio","sym_frac",
       "turn_deg","arrival_ratio","lintel","nested","n_pair_frames"}"""
    p = _params(params)
    prof = frame_profiles(video, fps=fps, params=params, max_frames=max_frames)
    info = dict(prof["info"])
    info["params"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()}
    out = {"schema": SCHEMA, "doors": [], "candidates": [], "profile": None, "info": info}
    col, t, hmap = prof["col"], prof["t"], prof["hmap"]
    n, W = col.shape[0], (col.shape[1] if col.ndim == 2 else 0)
    if n < int(p["min_track_frames"]):
        info["fail"] = f"only {n} frames — need >= {int(p['min_track_frames'])}"
        return out

    # px -> deg, for the turn gate and the metric width only (documented assumption).
    fx = 0.5 * W / np.tan(np.radians(0.5 * float(p["hfov_deg"])))
    info["fx_px"] = round(float(fx), 2)

    tracks = track_peaks(col, gate=p["track_gate_px"], gate_vel=p["track_gate_vel"],
                         max_gap=int(p["track_max_gap"]), min_h=p["peak_min_height"],
                         prom=p["peak_prominence"], max_peaks=int(p["max_peaks"]),
                         min_len=int(p["min_track_frames"]))
    cand_tracks = [(_diverging(tr, t, W, p), tr) for tr in tracks]
    good = [(d, tr) for d, tr in cand_tracks if d["ok"]]
    info["n_tracks"] = len(tracks)
    info["n_diverging_tracks"] = len(good)
    rej_hist: dict = {}
    for d, _ in cand_tracks:
        for r in (d["reject"] or ["ok"]):
            rej_hist[r] = rej_hist.get(r, 0) + 1
    info["track_reject_hist"] = rej_hist
    foes = [d["foe"] for d, _ in good if np.isfinite(d["foe"])]
    info["view"] = {
        "median_peaks_per_frame": float(np.median([len(_peaks(
            col[f], p["peak_min_height"], p["peak_prominence"], int(p["max_peaks"])))
            for f in range(0, n, max(1, n // 40))])),
        "median_foe_px": (round(float(np.median(foes)), 1) if foes else None),
        "foe_offset_frac": (round(float(np.median(foes) - 0.5 * W) / (0.5 * W), 3)
                            if foes else None),
        "median_abs_shift_px": round(float(np.median(np.abs(prof["shift_px"]))), 3),
    }

    # optional pose-side arclength / heading, for reporting and the `s` field
    s_axis = tp = None
    if traj_xz is not None and pose_times is not None:
        tr2 = np.asarray(traj_xz, float).reshape(-1, 2)
        tp = np.asarray(pose_times, float).reshape(-1)
        if len(tp) != len(tr2):
            raise ValueError(f"pose_times {len(tp)} != trajectory {len(tr2)}")
        s_axis = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(tr2, axis=0), axis=1))]
        info["walk_arclength"] = round(float(s_axis[-1]), 5)

    foe_g = float(np.median(foes)) if foes else 0.5 * W
    rho = expansion_rate(tracks, t, foe_g, W)
    info["view"]["median_expansion_rate"] = (
        round(float(np.nanmedian(rho)), 4) if np.isfinite(rho).any() else None)

    pairs = []
    for a in range(len(good)):
        for b in range(a + 1, len(good)):
            da, ta = good[a]
            db, tb = good[b]
            if da["side"] == db["side"] or da["side"] == 0 or db["side"] == 0:
                continue                     # gate (a): must be straddled
            L, Ltr, Rr, Rtr = (da, ta, db, tb) if da["side"] < 0 else (db, tb, da, ta)
            pairs.append(_pair_candidate(L, Ltr, Rr, Rtr, t, col, hmap, prof["shift_px"],
                                         rho, W, fx, p, s_axis, tp, width_hint))
    # ONE door produces MANY pairs: the clear opening (jamb to jamb), the outer reveal
    # around it and the partition's own junction with the corridor walls all cross at the
    # same instant, and MEASURED on the one-door fixture they score 0.92, 0.90 and 0.98 —
    # the widest reading is not the worst one. So the cluster is collapsed by crossing
    # time and the NARROWEST reading that passed every gate represents it: nested
    # apertures are one constriction, and the constriction is the door (`door_detect`
    # reads the same event as the contraction of the passage, not as the corridor around
    # it). The readings that lose are kept on the survivor as `nested`, never dropped.
    pairs.sort(key=lambda c: (0 if c["accepted"] else 1,
                              c["width_px_s"] if c["accepted"] else len(c["reject"]),
                              c["time_s"]))
    kept: list = []
    for c in pairs:
        tol = max(p["pair_t_min_s"], p["pair_t_frac"] * max(c["_dur"], 1e-9))
        hit = next((k for k in kept if abs(c["time_s"] - k["time_s"]) <= tol), None)
        if hit is not None:
            hit["nested"].append({"time_s": c["time_s"], "width_px_s": c["width_px_s"],
                                  "confidence": c["confidence"], "reject": c["reject"]})
            continue
        c["nested"] = []
        kept.append(c)
    for c in kept:
        c.pop("_dur", None)
    kept.sort(key=lambda c: c["time_s"])
    out["candidates"] = kept
    out["doors"] = [c for c in kept if c["accepted"]]
    info["n_candidates"] = len(kept)
    info["n_doors"] = len(out["doors"])
    hist: dict = {}
    for c in kept:
        for r in (c["reject"] or ["accepted"]):
            hist[r] = hist.get(r, 0) + 1
    info["candidate_reject_hist"] = hist
    if not out["doors"]:
        info["warn"] = _no_door_warning(info, n)
    if keep_profile:
        out["profile"] = {
            "t": [round(float(v), 4) for v in t],
            "shift_px": [round(float(v), 3) for v in prof["shift_px"]],
            "tracks": [{"t0": round(d["t0"], 3), "t1": round(d["t1"], 3), "n": d["n"],
                        "x0": round(d["x0"], 1), "x1": round(d["x1"], 1),
                        "T": (round(d["T"], 3) if np.isfinite(d["T"]) else None),
                        "foe": (round(d["foe"], 1) if np.isfinite(d["foe"]) else None),
                        "K": (round(d["K"], 1) if np.isfinite(d["K"]) else None),
                        "r2": (round(d["r2"], 4) if np.isfinite(d["r2"]) else None),
                        "side": d["side"], "reject": d["reject"]}
                       for d, _ in cand_tracks],
        }
    return out


def _no_door_warning(info: dict, n: int) -> str:
    v = info.get("view", {})
    hist = info.get("track_reject_hist", {})
    top = sorted(((k, c) for k, c in hist.items() if k != "ok"),
                 key=lambda kv: -kv[1])[:3]
    return ("no door-crossing accepted over %d frames: %d/%d peak tracks diverge with a "
            "crossing time (top track rejections: %s); median %.1f vertical structures "
            "per frame, median |frame shift| %.2f px. If the structures are there but "
            "nothing diverges, the camera is not looking along the walk (an up-tilted rig "
            "puts the opening out of frame) or the walk did not pass through one."
            % (n, info.get("n_diverging_tracks", 0), info.get("n_tracks", 0),
               ", ".join(f"{k}x{c}" for k, c in top) or "none",
               v.get("median_peaks_per_frame", float("nan")),
               v.get("median_abs_shift_px", float("nan"))))


def _pair_candidate(L, Ltr, R, Rtr, t, col, hmap, shift_px, rho, W, fx, p, s_axis, tp,
                    width_hint) -> dict:
    """Score one left/right jamb pair. Hard gates append to `reject`; the confidence is a
    geometric mean over the shape axes (one bad axis cannot be hidden) times a density
    trust — the same composition `door_detect` uses."""
    reject: list = []
    dur = 0.5 * ((L["t1"] - L["t0"]) + (R["t1"] - R["t0"]))
    tol = max(p["pair_t_min_s"], p["pair_t_frac"] * max(dur, 1e-9))
    dT = abs(L["T"] - R["T"])
    if dT > tol:
        reject.append("crossing_times_disagree")     # gate (b)

    # frames where BOTH jambs were actually observed
    fi = np.intersect1d(Ltr["i"], Rtr["i"])
    short = min(len(Ltr["i"]), len(Rtr["i"]))
    ov = len(fi) / max(short, 1)
    if ov < p["min_overlap_frac"] or len(fi) < int(p["lintel_min_frames"]):
        reject.append("jambs_not_seen_together")
    xl = np.interp(fi, Ltr["i"], Ltr["x"])
    xr = np.interp(fi, Rtr["i"], Rtr["x"])

    T = 0.5 * (L["T"] + R["T"])
    # |X|/v in seconds of walking: |K| = fx*|X|/v  (eq. 1)
    hw_l, hw_r = abs(L["K"]) / fx, abs(R["K"]) / fx
    width_s = hw_l + hw_r
    sym = hw_l / width_s if width_s > 1e-12 else float("nan")
    lo, hi = p["sym_band"]
    if not (lo <= sym <= hi):
        reject.append("asymmetric_pair")

    # arclength + metric width, when the poses give a speed
    s_val = width_recon = width_ratio = None
    if s_axis is not None:
        s_val = float(np.interp(T, tp, s_axis))
        v_rec = float(np.interp(T, tp, np.gradient(s_axis, tp)))
        width_recon = float(width_s * abs(v_rec))
        if width_hint is not None and float(width_hint) > 0:
            width_ratio = width_recon / float(width_hint)
            wlo, whi = p["width_ratio_band"]
            if not (wlo <= width_ratio <= whi):
                reject.append("width_out_of_band")

    # gate (d): image-measured yaw across the event window
    f0, f1 = int(min(Ltr["i"][0], Rtr["i"][0])), int(max(Ltr["i"][-1], Rtr["i"][-1]))
    turn_px = float(np.sum(np.abs(shift_px[f0:f1 + 1])))
    turn_deg = float(np.degrees(np.arctan(turn_px / fx))) if fx > 0 else 0.0
    if turn_deg > p["turn_hard_deg"]:
        reject.append("turning")

    # gate (f): the walk must still be moving between the last sight of the jambs and T,
    # otherwise the frame exit was the lens running out, not the camera going through.
    arrival = None
    iT = int(np.searchsorted(t, T))
    if iT >= len(t) - int(p["arrival_min_frames"]):
        reject.append("crossing_not_witnessed")     # video ends before the crossing
    else:
        seen = rho[f0:f1 + 1]
        gap = rho[f1:iT + 1]
        r_seen = float(np.nanmedian(seen)) if np.isfinite(seen).any() else float("nan")
        r_gap = float(np.nanmedian(gap)) if np.isfinite(gap).any() else float("nan")
        if np.isfinite(r_seen) and abs(r_seen) > 1e-9 and np.isfinite(r_gap):
            arrival = float(r_gap / r_seen)
            if arrival < p["arrival_frac"]:
                reject.append("walk_stopped_before_crossing")
        elif iT > f1 + 1:
            reject.append("arrival_unmeasured")

    lin = _lintel(hmap, fi, xl, xr, t, W, p) if len(fi) else {
        "found": False, "T": None, "r2": None, "n": 0, "reason": "no_common_frames"}
    if not lin["found"]:
        reject.append("no_lintel:" + str(lin.get("reason")))
    lin_score = 0.0
    if lin["found"]:
        dTl = abs(float(lin["T"]) - T)
        ltol = max(p["pair_t_min_s"], p["lintel_t_frac"] * max(dur, 1e-9))
        lin_score = float(np.clip(1.0 - dTl / ltol, 0.0, 1.0))
        if lin_score <= 0.0:
            reject.append("lintel_time_disagrees")

    # The brightness step across the aperture (the space beyond a door is lit differently
    # from the corridor) was considered as a fifth cue and is NOT used: a painted panel or
    # a poster steps in brightness exactly like an opening does — tests/…::wall_poster is
    # literally a bright rectangle the size of a door — so it adds no separation, only a
    # knob. The geometry (a, b, c, e, f) is what carries the discrimination.
    scores = {
        "cross_time": round(float(np.clip(1.0 - dT / max(tol, 1e-9), 0.0, 1.0)), 4),
        "symmetry": round(_score_tri(sym, lo, hi), 4),
        "fit": round(float(np.clip((min(L["r2"], R["r2"]) - p["min_r2"])
                                   / max(1.0 - p["min_r2"], 1e-9), 0.0, 1.0)), 4),
        "straight": round(float(np.clip(
            (p["turn_hard_deg"] - max(turn_deg, p["turn_soft_deg"]))
            / max(p["turn_hard_deg"] - p["turn_soft_deg"], 1e-9), 0.0, 1.0)), 4),
        "lintel": round(lin_score, 4),
        "trust": round(float(np.clip(len(fi) / max(int(p["trust_frames"]), 1), 0.0, 1.0))
                       * float(np.clip(ov / max(p["min_overlap_frac"], 1e-9), 0.0, 1.0)), 4),
    }
    if width_ratio is not None:
        scores["width_ratio"] = round(_score_tri(width_ratio, *p["width_ratio_band"]), 4)
    shape = [scores["cross_time"], scores["symmetry"], scores["fit"], scores["straight"],
             scores["lintel"]] + ([scores["width_ratio"]] if width_ratio is not None
                                  else [])
    conf = float(np.exp(np.mean(np.log(np.maximum(shape, 1e-12))))) * scores["trust"]
    conf = 0.0 if reject else conf
    if conf < p["min_confidence"] and "below_min_confidence" not in reject:
        reject.append("below_min_confidence")
    fps = 1.0 / max(float(t[1] - t[0]), 1e-9) if len(t) > 1 else 0.0
    return {
        "time_s": round(float(T), 4),
        "frame": int(round(float(T) * fps)),
        "s": (round(s_val, 5) if s_val is not None else None),
        "t_cross_left": round(float(L["T"]), 4),
        "t_cross_right": round(float(R["T"]), 4),
        "foe_px": round(float(0.5 * (L["foe"] + R["foe"])), 1),
        "width_px_s": round(float(width_s), 4),
        "width_recon": (round(width_recon, 5) if width_recon is not None else None),
        "width_ratio": (round(width_ratio, 4) if width_ratio is not None else None),
        "sym_frac": round(float(sym), 4) if np.isfinite(sym) else None,
        "turn_deg": round(turn_deg, 2),
        "arrival_ratio": (round(arrival, 3) if arrival is not None else None),
        "lintel": lin,
        "n_pair_frames": int(len(fi)),
        "overlap_frac": round(float(ov), 3),
        "r2": [round(float(L["r2"]), 4), round(float(R["r2"]), 4)],
        "confidence": round(conf, 4),
        "scores": scores,
        "accepted": not reject,
        "reject": reject,
        "_dur": float(dur),
    }


# --------------------------------------------------------------------------------------
# SECTION 5 — the same two helpers door_detect exposes (identical semantics)
# --------------------------------------------------------------------------------------

def door_times(result: dict, min_confidence=None) -> list:
    """Accepted detections -> times in VIDEO seconds, for `--door-times` and for
    `coarse_match.door_arclengths(traj, pose_times, door_times(result))`."""
    lo = 0.0 if min_confidence is None else float(min_confidence)
    return [float(c["time_s"]) for c in result.get("doors", []) if c["confidence"] >= lo]


def door_s_values(result: dict, min_confidence=None) -> list:
    """Accepted detections -> recon arclengths, for `coarse_match(door_s=...)`.

    Needs `traj_xz`/`pose_times` at detection time; without them the detector has no
    arclength axis and this raises rather than inventing one. `min_confidence` can only
    RAISE the bar: a rejected candidate failed a hard gate, and promoting it would be the
    fabrication this module refuses."""
    lo = 0.0 if min_confidence is None else float(min_confidence)
    ds = [c for c in result.get("doors", []) if c["confidence"] >= lo]
    if any(c.get("s") is None for c in ds):
        raise ValueError("detections carry no arclength — pass traj_xz= and pose_times= "
                         "to detect_doors()")
    return [float(c["s"]) for c in ds]


# --------------------------------------------------------------------------------------
# SECTION 6 — reproduction CLI (observation, not judgement)
# --------------------------------------------------------------------------------------
# `python -m scan2bim.door_detect_rgb <video> [--lbp X.lbp2] [--duration 35.3]
#                                    [--out DIR] [--metric-scale 1.97]`
# prints EVERY candidate, accepted or not, with the gate it failed — the table a human
# needs to line the detector up against the video. It judges nothing: with no annotated
# door in any upload there is no ground truth to score against.

def _cli(argv=None):
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--lbp", help="matching .lbp2/3/4 payload, for the arclength axis")
    ap.add_argument("--duration", type=float, default=None,
                    help="video seconds spanned by the poses (build_coplay's axis); "
                         "defaults to the decoded frame count / fps")
    ap.add_argument("--metric-scale", type=float, default=None,
                    help="recon units -> metres, POST HOC: it is applied to the reported "
                         "width only and is never seen by the detector")
    ap.add_argument("--width-hint", type=float, default=None,
                    help="corridor clear width in RECON units (enables the width gate)")
    ap.add_argument("--out", help="directory for the JSON result + candidate filmstrips")
    ap.add_argument("--max-frames", type=int, default=None)
    a = ap.parse_args(argv)

    traj = ptimes = None
    if a.lbp:
        from .door_detect import gravity_align, read_lbp
        d = read_lbp(a.lbp)
        g = gravity_align(d["poses"], d["points"])
        traj = g["traj_xz"]
        n = len(traj)
        dur = a.duration
        if dur is None:
            cap = cv2.VideoCapture(str(a.video))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            dur = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0) / max(fps, 1e-9)
            cap.release()
        ptimes = np.arange(n) * float(dur) / max(n - 1, 1)

    r = detect_doors(a.video, pose_times=ptimes, traj_xz=traj,
                     width_hint=a.width_hint, max_frames=a.max_frames)
    i = r["info"]
    print(f"# {SCHEMA}  {a.video}")
    print(f"# {i['n_frames']} frames @ {i['fps']:.3f} fps, work {i['work_size']}, "
          f"rows {i['row_band_px']}, fx {i.get('fx_px')} px (hfov "
          f"{i['params']['hfov_deg']} deg assumed)")
    print(f"# tracks {i['n_tracks']} -> diverging {i['n_diverging_tracks']} -> pairs "
          f"{i['n_candidates']} -> ACCEPTED {i['n_doors']}")
    print(f"# view {i.get('view')}")
    print(f"# track rejections   {i.get('track_reject_hist')}")
    print(f"# pair rejections    {i.get('candidate_reject_hist')}")
    if i.get("warn"):
        print(f"# WARN {i['warn']}")
    sc = a.metric_scale
    hdr = ["t_s", "frame", "s_recon", "width_recon", "width_m", "conf", "shape",
           "n_pair", "turn_deg", "arrival", "lintel", "verdict"]
    print("\t".join(hdr))
    for c in r["candidates"]:
        w = c["width_recon"]
        print("\t".join([
            f"{c['time_s']:.3f}", str(c["frame"]),
            ("-" if c["s"] is None else f"{c['s']:.4f}"),
            ("-" if w is None else f"{w:.4f}"),
            ("-" if (w is None or sc is None) else f"{w * sc:.2f}"),
            f"{c['confidence']:.3f}", f"{shape_score(c):.3f}",
            str(c["n_pair_frames"]), f"{c['turn_deg']:.1f}",
            ("-" if c["arrival_ratio"] is None else f"{c['arrival_ratio']:.2f}"),
            (str(c["lintel"]["reason"]) if not c["lintel"]["found"]
             else f"T={c['lintel']['T']:.2f},cov={c['lintel']['coverage']}"),
            ("ACCEPTED" if c["accepted"] else ",".join(c["reject"]))]))
    print(f"# door_times = {door_times(r)}")
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(a.video).stem
        (out / f"{stem}.door_rgb.json").write_text(json.dumps(r, indent=1))
        print(f"# wrote {out / (stem + '.door_rgb.json')}")
    return r


def shape_score(cand: dict) -> float:
    """The confidence a candidate would carry if no HARD gate had fired.

    Reported next to `confidence` because the two say different things: `confidence` is 0
    the moment any gate fires, so a table of confidences cannot show HOW close a rejected
    candidate came. MEASURED, and the reason this is worth printing: the dead-end fixture
    scores 0.910 here against 0.922-0.939 for real doorways — its shape is not
    distinguishable, only its arrival is."""
    s = cand.get("scores", {})
    ax = [s.get("cross_time", 0.0), s.get("symmetry", 0.0), s.get("fit", 0.0),
          s.get("straight", 0.0), s.get("lintel", 0.0)]
    if "width_ratio" in s:
        ax.append(s["width_ratio"])
    return float(np.exp(np.mean(np.log(np.maximum(ax, 1e-12))))) * s.get("trust", 0.0)


if __name__ == "__main__":       # pragma: no cover
    _cli()

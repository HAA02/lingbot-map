"""AUTOMATIC DOOR-PASSING DETECTION, from the RECON ALONE.

WHY THIS MODULE EXISTS (measured, cycle 7)
------------------------------------------
`coarse_match` can place a walk on a floor plan, but repeated / symmetric corridor
geometry ties two placements exactly: the 2-leg L fixture holds at margin 0.0 and a
DOOR event is the ONLY evidence measured to break it (margin 0.0 -> 0.22, see
tests/test_coarse_match.TestAmbiguityHolds). That evidence had no source: the uploads
carry no door annotation. This module manufactures it from the recon's own shape.

CIRCULARITY IS FORBIDDEN, BY CONSTRUCTION
-----------------------------------------
Nothing here may read the DXF, the BIM, `plan_skeleton` or a `coarse_match` result — if
the door evidence came from the drawing, "the door disambiguates the drawing" would be a
circular argument. The inputs are therefore ONLY:

    traj_xz    the gravity-aligned recon walk (recon units, monocular / up to scale)
    cam_y      the walk's own height (the pose Y in the same gravity-aligned frame)
    pts_yup    the recon point cloud, same frame

The one external number this module will accept is `width_hint`: a corridor width IN
RECON UNITS, i.e. a SCALE prior (metres per recon unit), never plan geometry. It is
optional; without it the module self-calibrates its bin/band scale from the cloud.

THE OBSERVABLE (lateral free-space contraction)
----------------------------------------------
At each pose, look sideways (perpendicular to the direction of travel) inside a
longitudinal slab and a height band, and measure the distance to the nearest STRUCTURE
on the left and on the right. Their sum is the clear passage width w(s) along the walk.
Walking THROUGH a doorway contracts w to the door's clear width and then re-opens it:

    corridor 1.82 m  ->  door 0.8-1.0 m  ->  corridor 1.82 m

so a door is a contraction-then-recovery of w over an arclength of the order of the door
frame's thickness. Four things are required, each killing a specific false positive:

  (a) BOTH sides must contract, each against its OWN shoulder value. A column, pilaster
      or duct riser contracts one side only - and a 0.75 m deep one lands INSIDE the
      width band (w 1.07 m, ratio 0.59 on a 1.82 m corridor), so the ratio test alone
      does not catch it. Two-sidedness does.
  (b) the contracted width / local baseline must sit in `width_ratio_band`, from the
      door-leaf standard (see the constants below);
  (c) the contraction must RECOVER on both sides within `recover_search_frac`, and its
      arclength must stay inside `span_frac_band` - a long narrow neck is a narrow
      corridor, not a door;
  (d) the walk must be locally STRAIGHT: a corner swings the lateral direction across
      the geometry and the width measurement stops meaning anything. A big turn is a
      hard reject, a small one a confidence discount.

WHAT IT CANNOT DO (stated so no one over-reads a detection)
----------------------------------------------------------
* It detects doors WALKED THROUGH. A closed door flush in a side wall changes no
  geometry at all, and an OPEN side door widens one side instead of contracting two, so
  neither is a contraction event. `plan_skeleton.attach_doors` keeps every A-DOOR insert
  within `door_radius` of a centreline, side doors included, so the plan side generally
  has MORE doors than this detector can ever return. That asymmetry is fine for the
  matcher (an unexplained plan landmark costs nothing; an INVENTED walk event does), but
  it means recall against the drawing is bounded by how many doors were walked through.
* It needs a cloud that actually resolves both walls at eye level. MEASURED on
  realtime/_uploads/upload_1781521406685 (144 poses, 787 521 points): only 73 of 144
  poses hold >= 30 points on BOTH sides at any height (median 54 left vs 15 092 right),
  and the nearest structure lands in the first histogram bin — >= 1 % of a side's eye-band
  mass sits within 0.014 recon units (2.8 cm) of the walk, so the median passage width
  reads 0.027 where s_h = 1.97 puts the 1.82 m corridor at 0.924. A one-sided, smeared
  cloud. That upload therefore yields 0 detections, and `info['coverage']` +
  `info['warn']` say why, so a 0-detection run is diagnosable instead of mysterious.

REFUSAL PHILOSOPHY (inherited from coarse_match / estimate_plan_transform)
--------------------------------------------------------------------------
Sub-threshold candidates are RETURNED, with their confidence and the gate they failed,
never silently dropped: the user has to be able to line them up against the video. And
`doors` (the accepted list) is empty rather than optimistic - a fabricated door time
would corrupt the very ambiguity it is supposed to resolve.

UNITS
-----
Everything is in RECON units and every threshold is a RATIO of a measured width, so the
module is scale-free (pinned by `test_scale_free...`: the same corridor pushed through
the anisotropic recon transform yields the same detections). Times, when `pose_times` is
given, are that array's units (build_coplay's pose time axis is t_i = duration*i/(n-1),
video seconds).
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

SCHEMA = "coplay.door_detect/1.0"

# --------------------------------------------------------------------------------------
# CONSTANTS — every value is a physical dimension or a directly measured quantity.
# Nothing here is a fitted knob; the derivations are the comments.
# --------------------------------------------------------------------------------------

#: Clear width of a single-leaf interior door incl. frame reveal. KS F 3109 / Korean
#: practice: 800-1000 mm leaf, so the walked clear opening is ~0.75-1.10 m.
DOOR_CLEAR_M = (0.75, 1.10)

#: The team's MEASURED corridor clear width (Gasan_7F, Revit DXF 1.821 m) — the
#: reference the ratio band is derived from, NOT a value read at run time.
CORRIDOR_CLEAR_M = 1.82

#: Door height (KS F 3109 standard leaf 2100 mm). The height band must stay BELOW the
#: lintel: above it the cross wall is solid and spans the whole corridor, which would
#: read as a zero-width passage right where the door is.
DOOR_HEIGHT_M = 2.1

#: Assumed camera carry height (metric_scale.camera_height_scale's own default) — used
#: only to turn the two heights above into fractions of the corridor width.
EYE_HEIGHT_M = 1.5

#: Fraction of the cloud's vertical extent trimmed off the top and bottom when the width
#: SCALE is being calibrated (walls span the whole height, slabs sit at the extremes).
#: The same 0.15 `wall_anchor._BAND_MARGIN` uses — one convention, not two.
MID_BAND_MARGIN = 0.15

#: The median measured passage width must reach this fraction of the calibrated corridor
#: width, or the cloud gets a smear WARNING (see `lateral_profile`).
SMEAR_WARN_FRAC = 0.6

DEFAULTS = {
    # ---- height band, as fractions of the reference width, measured from the pose Y --
    # below: (1.5 - 0.5)/1.82 = 0.55  -> 0.5 m off the floor, clear of the slab/skirting
    # above: (1.96 - 1.5)/1.82 = 0.25 -> 1.96 m, just under a 2.1 m door lintel
    "band_below_frac": 0.55,
    "band_above_frac": 0.25,

    # ---- lateral measurement -------------------------------------------------------
    # longitudinal half-slab. 0.15 W = 0.27 m: thin enough that a door frame is resolved
    # (the contraction spans 2*0.27 = 0.55 m = 0.30 W, inside span_frac_band), thick
    # enough to hold points at a realistic recon density.
    "lon_win_frac": 0.15,
    # search range for the nearest structure, and the histogram bin used to find it.
    "max_range_frac": 1.6,
    "bin_frac": 0.03,
    # A real surface crossing the slab deposits at least this share of a side's points
    # into ONE bin; isolated reconstruction floaters (which a plain low percentile
    # latches onto — measured 0.003 recon units on the real upload) do not. This is why
    # the estimator is a density onset and not a percentile.
    "onset_mass_frac": 0.01,
    # a side with fewer points than this is UNMEASURED (never guessed).
    "min_side_points": 30,
    # ...and this many is "fully trusted"; in between the confidence is discounted.
    "trust_side_points": 200,
    # poses used to estimate the direction of travel (+-k), so pose-to-pose jitter does
    # not rotate the lateral axis.
    "fwd_smooth": 5,

    # ---- contraction / recovery ----------------------------------------------------
    # A door on a 1.82 m corridor gives w/baseline = 0.41-0.60 (DOOR_CLEAR_M /
    # CORRIDOR_CLEAR_M). Widened to (0.30, 0.70) to absorb the onset estimator's bias
    # under noise (measured -0.1 W at sigma 0.05 m) and other corridor widths. A door
    # nearly as wide as its corridor is NOT a contraction event and is out of scope.
    "width_ratio_band": (0.30, 0.70),
    # w must fall below enter_frac x baseline to open a run, and both sides must come
    # back above recover_frac x their shoulder for it to be a door and not a narrowing.
    "enter_frac": 0.80,
    "recover_frac": 0.88,
    # each side, on its own, must lose this much of its shoulder distance: the
    # two-sidedness gate that rejects a column (a) — 0.25 rejects the 0.75 m riser
    # fixture, whose unaffected side loses 0.00.
    "side_drop_min": 0.20,
    # contraction arclength / baseline width. Lower: 0.04 W = 0.07 m, one pose step.
    # Upper: 0.75 W = 1.37 m — a door frame plus slab thickness plus the 2 x lon_win
    # smearing the slab itself adds; beyond that it is a narrow corridor.
    "span_frac_band": (0.04, 0.75),
    # how far (in baseline widths) to look for the shoulders before giving up.
    "recover_search_frac": 3.0,
    # half-window of the rolling baseline median. 3 W = 5.5 m each side: wide enough
    # that a 3 m narrow neck cannot become its own baseline (and be missed).
    "baseline_win_frac": 3.0,

    # ---- turn gates ----------------------------------------------------------------
    # accumulated |heading change| across the shoulders. Above hard_deg the lateral axis
    # has swung too far for w to mean anything (a 90 deg corner); below soft_deg it is
    # free; in between it is a linear confidence discount.
    "turn_soft_deg": 12.0,
    "turn_hard_deg": 35.0,

    # ---- acceptance ----------------------------------------------------------------
    # geometric mean of the four shape scores x the density trust. MEASURED separation
    # over tests/test_door_detect.py's synthetic suite (15 true doorways across clean,
    # noisy, thinned, metric and anisotropic-recon-unit fixtures): true doorways score
    # 0.59-0.96, every adversarial fixture (one-sided riser, corner, wide opening, 3 m
    # neck, dead end, smear) scores 0.00 because a HARD gate fires, and the best
    # sub-threshold candidate on the real upload scored 0.18. 0.45 therefore sits in an
    # empty gap instead of on top of a distribution.
    "min_confidence": 0.45,
}


# --------------------------------------------------------------------------------------
# SECTION 1 — geometry helpers
# --------------------------------------------------------------------------------------

def _params(over=None) -> dict:
    p = dict(DEFAULTS)
    for k, v in (over or {}).items():
        if k not in DEFAULTS:
            raise ValueError(f"unknown door_detect parameter {k!r}")
        p[k] = v
    return p


def _wrap180(a: float) -> float:
    return float((float(a) + 180.0) % 360.0 - 180.0)


def _forward_frame(traj: np.ndarray, k: int):
    """(arclength, forward unit vectors, left-normal unit vectors, heading degrees).

    The direction of travel is taken over +-k poses instead of the adjacent difference:
    at a 0.04 recon-unit pose step, pose-to-pose jitter would otherwise rotate the
    lateral axis by tens of degrees and the "lateral" distance would be a mix of
    sideways and forward geometry."""
    n = len(traj)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(traj, axis=0), axis=1))]
    i = np.arange(n)
    lo = np.maximum(i - int(k), 0)
    hi = np.minimum(i + int(k), n - 1)
    f = traj[hi] - traj[lo]
    bad = np.linalg.norm(f, axis=1) < 1e-12
    if bad.any():                                   # degenerate window -> local step
        f[bad] = (traj[np.minimum(i + 1, n - 1)] - traj[np.maximum(i - 1, 0)])[bad]
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-12)
    nrm = np.stack([-f[:, 1], f[:, 0]], axis=1)
    head = np.degrees(np.arctan2(f[:, 1], f[:, 0]))
    return s, f, nrm, head


def _onset(d: np.ndarray, bin_w: float, max_range: float, mass_frac: float,
           min_pts: int) -> float:
    """Distance to the nearest STRUCTURE among one side's lateral offsets `d` (>0).

    A density onset, not a percentile: the first bin holding at least `mass_frac` of the
    side's mass. Measured reason — on the real upload a 5th percentile returns 0.003
    recon units (a near-camera floater) where the corridor half-width is 0.46; requiring
    a bin to carry 1 % of the side's points removes stray points without needing a peak
    (a door JAMB is an edge and never a spike: the corridor wall behind it carries ~18x
    the density, so a prominence-based estimator would step right over the door)."""
    d = d[(d > 0.0) & (d <= max_range)]
    if len(d) < int(min_pts):
        return float("nan")
    nb = max(4, int(np.ceil(max_range / max(bin_w, 1e-9))))
    hist, edges = np.histogram(d, bins=nb, range=(0.0, max_range))
    need = max(1.0, mass_frac * float(len(d)))
    j = np.nonzero(hist >= need)[0]
    if not len(j):
        return float("nan")
    return float(0.5 * (edges[j[0]] + edges[j[0] + 1]))


# --------------------------------------------------------------------------------------
# SECTION 2 — the lateral free-space profile
# --------------------------------------------------------------------------------------

def lateral_profile(traj_xz, pts_yup, cam_y=None, width_hint=None, params=None) -> dict:
    """Per-pose left/right free distance and clear width w(s) along the walk.

    traj_xz     (N,2) gravity-aligned recon walk (recon units), time-ordered.
    pts_yup     (M,3) gravity-aligned cloud [X, Y, Z] in the SAME frame/units.
    cam_y       (N,) pose height in that frame. Without it the height band falls back to
                `wall_anchor`'s mid-band convention (15 % trimmed off top and bottom),
                which is reported as band_source='cloud_midband' because it is weaker: a
                door lintel can then leak into the band.
    width_hint  corridor clear width in RECON units, if a scale prior is available.
                Without it, w0 = 2 x median|lateral offset| self-calibrates the bin and
                band scale off the cloud.

    Returns {"s","left","right","width","n_left","n_right","measured","heading_deg",
             "w0","info"} with NaN wherever a side had too few points to measure —
    never a guessed distance."""
    p = _params(params)
    traj = np.asarray(traj_xz, dtype=np.float64).reshape(-1, 2)
    pts = np.asarray(pts_yup, dtype=np.float64).reshape(-1, 3)
    n = len(traj)
    info: dict = {"n_poses": int(n), "n_points": int(len(pts))}
    empty = {"s": np.zeros(0), "left": np.zeros(0), "right": np.zeros(0),
             "width": np.zeros(0), "n_left": np.zeros(0, dtype=int),
             "n_right": np.zeros(0, dtype=int), "measured": np.zeros(0, dtype=bool),
             "heading_deg": np.zeros(0), "w0": 0.0, "info": info}
    if n < 3 or len(pts) < int(p["min_side_points"]):
        info["fail"] = "not enough poses or points"
        return empty

    s, fwd, nrm, head = _forward_frame(traj, p["fwd_smooth"])
    if float(s[-1]) <= 1e-12:
        info["fail"] = "degenerate walk (zero arclength)"
        return empty

    if cam_y is not None:
        cy = np.asarray(cam_y, dtype=np.float64).reshape(-1)
        if len(cy) != n:
            raise ValueError(f"cam_y {len(cy)} != trajectory {n}")
        info["band_source"] = "pose_y"
    else:                                       # wall_anchor's mid-band, degraded
        ylo, yhi = float(pts[:, 1].min()), float(pts[:, 1].max())
        cy = np.full(n, 0.5 * (ylo + yhi))
        info["band_source"] = "cloud_midband"

    # --- scale calibration: w0 sets the bin width, the band and the search range ------
    xz = pts[:, [0, 2]]
    tree = cKDTree(xz)
    if width_hint is not None and float(width_hint) > 0:
        w0 = float(width_hint)
        info["w0_source"] = "width_hint"
    else:
        # 2 x the median distance from the walk to the nearest wall point ~= the passage
        # width; that sets the bin, the band and the search range. Self-calibrating, so
        # the module stays scale-free; a smeared cloud reports a small w0, which the
        # coverage diagnostics then expose rather than hide.
        #
        # The FLOOR and CEILING must be dropped first, and they cannot be dropped by the
        # pose-Y band because that band is expressed in w0 - the very number being
        # calibrated. wall_anchor's convention (trim MID_BAND_MARGIN of the vertical
        # extent off the top and bottom; walls span the whole height, slabs sit at the
        # extremes) is w0-free, so it is what is used here. MEASURED reason: with floor
        # and ceiling planes present, the nearest cloud point in plan view is directly
        # underfoot and the calibration collapsed to w0 0.037 instead of 1.82.
        ylo, yhi = np.percentile(pts[:, 1], [1.0, 99.0])
        yr = float(yhi - ylo)
        mid = (((pts[:, 1] > ylo + MID_BAND_MARGIN * yr)
                & (pts[:, 1] < yhi - MID_BAND_MARGIN * yr)) if yr > 1e-9
               else np.ones(len(pts), bool))
        if int(mid.sum()) < int(p["min_side_points"]):
            info["fail"] = "no mid-band points to calibrate a width scale"
            return empty
        rough = np.asarray(cKDTree(xz[mid]).query(traj, k=1)[0], dtype=np.float64)
        rough = rough[np.isfinite(rough)]
        w0 = float(2.0 * np.median(rough)) if len(rough) else 0.0
        if not np.isfinite(w0) or w0 <= 1e-9:
            info["fail"] = "cannot calibrate a width scale from the cloud"
            return empty
        info["w0_source"] = "self_calibrated(2*median wall distance, mid-band)"
    info["w0"] = round(w0, 5)

    lon_win = p["lon_win_frac"] * w0
    max_range = p["max_range_frac"] * w0
    bin_w = p["bin_frac"] * w0
    below, above = p["band_below_frac"] * w0, p["band_above_frac"] * w0
    info["band"] = [round(-below, 4), round(above, 4)]
    info["lon_win"] = round(lon_win, 4)
    info["max_range"] = round(max_range, 4)

    r = float(np.hypot(lon_win, max_range))
    idx = tree.query_ball_point(traj, r)
    left = np.full(n, np.nan)
    right = np.full(n, np.nan)
    nl = np.zeros(n, dtype=int)
    nr = np.zeros(n, dtype=int)
    for i in range(n):
        j = np.asarray(idx[i], dtype=np.int64)
        if not len(j):
            continue
        yb = (pts[j, 1] > cy[i] - below) & (pts[j, 1] < cy[i] + above)
        j = j[yb]
        if not len(j):
            continue
        q = xz[j] - traj[i]
        lon = q @ fwd[i]
        lat = q @ nrm[i]
        m = np.abs(lon) <= lon_win
        lat = lat[m]
        dl, dr = lat[lat > 0.0], -lat[lat < 0.0]
        nl[i], nr[i] = len(dl), len(dr)
        left[i] = _onset(dl, bin_w, max_range, p["onset_mass_frac"], p["min_side_points"])
        right[i] = _onset(dr, bin_w, max_range, p["onset_mass_frac"], p["min_side_points"])

    width = left + right
    measured = np.isfinite(width)
    med_w = float(np.nanmedian(width)) if measured.any() else float("nan")
    info["coverage"] = {
        "measured_poses": int(measured.sum()), "n_poses": int(n),
        "measured_frac": round(float(measured.mean()), 4),
        "left_only": int(np.sum(np.isfinite(left) & ~np.isfinite(right))),
        "right_only": int(np.sum(np.isfinite(right) & ~np.isfinite(left))),
        "median_side_points": [int(np.median(nl)), int(np.median(nr))],
        "median_width": (round(med_w, 4) if np.isfinite(med_w) else None),
        "width_vs_w0": (round(med_w / w0, 4) if np.isfinite(med_w) and w0 > 1e-12
                        else None),
    }
    # SMEAR / PATH-CONTAMINATION WARNING. Where you walk inside a corridor does not
    # change the SUM of the two side distances, so the measured width should equal the
    # calibrated corridor width to within the onset estimator's bias (worst case 0.25 w0,
    # measured at 0.10 m cloud noise). A median far below that means the nearest
    # "structure" is not a wall — on upload_1781521406685 the eye band puts >= 1 % of its
    # mass within 0.014 recon units (2.8 cm) of the walk, i.e. reconstruction points sit
    # ON the camera path, and the median width reads 0.027 where the corridor is 0.924.
    # Reported rather than silently swallowed: a 0-detection run must be diagnosable.
    if np.isfinite(med_w) and w0 > 1e-12 and med_w < SMEAR_WARN_FRAC * w0:
        info["warn"] = (f"median measured passage width {med_w:.4f} is below "
                        f"{SMEAR_WARN_FRAC:g} x the calibrated corridor width {w0:.4f} — "
                        f"the cloud likely carries points ON the camera path (monocular "
                        f"smear) or resolves only one side; door detection is not "
                        f"trustworthy on this cloud")
    return {"s": s, "left": left, "right": right, "width": width, "n_left": nl,
            "n_right": nr, "measured": measured, "heading_deg": head, "w0": w0,
            "info": info}


def _baseline(s: np.ndarray, width: np.ndarray, measured: np.ndarray,
              half_win: float) -> np.ndarray:
    """Rolling median of the measured width (a LOCAL corridor width): a walk can cross
    corridors of different widths, and a global median would then call the narrower one
    a door. Falls back to the global median where a window holds too few samples."""
    out = np.full(len(s), np.nan)
    if not measured.any():
        return out
    gmed = float(np.nanmedian(width[measured]))
    for i in range(len(s)):
        m = measured & (np.abs(s - s[i]) <= half_win)
        out[i] = float(np.median(width[m])) if int(m.sum()) >= 5 else gmed
    return out


def _runs(mask: np.ndarray) -> list:
    """Contiguous True runs of `mask` as (i0, i1) inclusive index pairs."""
    out = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def _cross(s: np.ndarray, w: np.ndarray, i_in: int, i_out: int, level: float) -> float:
    """Arclength where w crosses `level` between the (out-of-run, in-run) pose pair —
    linear interpolation, so the reported span does not quantise to the pose step."""
    if i_out < 0 or i_out >= len(s) or not np.isfinite(w[i_out]) or not np.isfinite(w[i_in]):
        return float(s[i_in])
    dw = w[i_out] - w[i_in]
    if abs(dw) < 1e-12:
        return float(s[i_in])
    t = float(np.clip((w[i_out] - level) / dw, 0.0, 1.0))
    return float(s[i_out] + t * (s[i_in] - s[i_out]))


def _shoulder(i: int, step: int, s: np.ndarray, width: np.ndarray, measured: np.ndarray,
              base: np.ndarray, recover_frac: float, limit: float):
    """Walk outward from run edge `i` to the first pose whose width has RECOVERED.
    Returns (index, value, fail_reason)."""
    j = i + step
    while 0 <= j < len(s) and abs(s[j] - s[i]) <= limit:
        if not measured[j]:
            return None, float("nan"), "unmeasured_gap"
        if width[j] >= recover_frac * base[j]:
            return j, float(width[j]), None
        j += step
    return None, float("nan"), "no_recovery"


def _tri(x: float, lo: float, hi: float) -> float:
    """1.0 at the middle of [lo, hi], 0.0 at (and outside) the edges."""
    if not np.isfinite(x) or x <= lo or x >= hi:
        return 0.0
    mid = 0.5 * (lo + hi)
    return float(1.0 - abs(x - mid) / (0.5 * (hi - lo)))


# --------------------------------------------------------------------------------------
# SECTION 3 — the entry point
# --------------------------------------------------------------------------------------

def detect_doors(traj_xz, pts_yup, cam_y=None, pose_times=None, width_hint=None,
                 params=None, keep_profile: bool = True) -> dict:
    """Detect door-PASSING events in a recon walk. See the module docstring.

    Returns a JSON-safe dict:
      {"schema": SCHEMA,
       "doors":      [candidate, ...]    accepted, ordered by arclength
       "candidates": [candidate, ...]    ALL of them, accepted or not, by arclength
       "profile":    {"s","width","left","right","baseline","n_left","n_right"} | None
       "info":       {...}               coverage / calibration / parameters}

    candidate = {"s", "time_s", "pose_index", "pose_index_min", "width",
                 "baseline_width", "width_ratio", "span", "span_frac", "left_drop",
                 "right_drop", "turn_deg", "n_points", "confidence", "scores": {...},
                 "accepted": bool, "reject": [reason, ...]}

    `time_s` is None unless `pose_times` is given (build_coplay's pose time axis
    t_i = duration*i/(n-1) — the same convention `--door-times` reads back)."""
    p = _params(params)
    prof = lateral_profile(traj_xz, pts_yup, cam_y=cam_y, width_hint=width_hint,
                           params=params)
    info = dict(prof["info"])
    info["params"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()}
    out = {"schema": SCHEMA, "doors": [], "candidates": [], "profile": None, "info": info}
    s, width, measured = prof["s"], prof["width"], prof["measured"]
    if not len(s):
        return out

    t = None
    if pose_times is not None:
        t = np.asarray(pose_times, dtype=np.float64).reshape(-1)
        if len(t) != len(s):
            raise ValueError(f"pose_times {len(t)} != trajectory {len(s)}")

    w0 = float(prof["w0"])
    base = _baseline(s, width, measured, p["baseline_win_frac"] * w0)
    if keep_profile:
        def _r(a):
            return [None if not np.isfinite(v) else round(float(v), 4) for v in a]
        out["profile"] = {"s": _r(s), "width": _r(width), "left": _r(prof["left"]),
                          "right": _r(prof["right"]), "baseline": _r(base),
                          "n_left": [int(v) for v in prof["n_left"]],
                          "n_right": [int(v) for v in prof["n_right"]]}
    if not measured.any():
        info["fail"] = "no pose could measure both sides"
        return out

    head = prof["heading_deg"]
    below = measured & np.isfinite(base) & (width <= p["enter_frac"] * base)
    info["n_runs"] = len(_runs(below))
    for i0, i1 in _runs(below):
        m = i0 + int(np.argmin(width[i0:i1 + 1]))
        b = float(base[m])
        reject: list = []
        jl, wl_sh, fl = _shoulder(i0, -1, s, width, measured, base, p["recover_frac"],
                                  p["recover_search_frac"] * w0)
        jr, wr_sh, fr = _shoulder(i1, +1, s, width, measured, base, p["recover_frac"],
                                  p["recover_search_frac"] * w0)
        for f in (fl, fr):
            if f and f not in reject:
                reject.append(f)

        w_min = float(width[m])
        ratio = w_min / b if b > 1e-12 else float("nan")
        s_lo = _cross(s, width, i0, i0 - 1, p["enter_frac"] * b)
        s_hi = _cross(s, width, i1, i1 + 1, p["enter_frac"] * b)
        span = float(max(s_hi - s_lo, 0.0))
        span_frac = span / b if b > 1e-12 else float("nan")

        # per-side drop against that side's OWN shoulder (gate (a))
        drops = []
        for arr, j0, j1 in ((prof["left"], jl, jr), (prof["right"], jl, jr)):
            sh = [arr[j] for j in (j0, j1) if j is not None and np.isfinite(arr[j])]
            if not sh:
                drops.append(float("nan"))
                continue
            ref = float(np.median(sh))
            drops.append(float(1.0 - arr[m] / ref) if ref > 1e-12 else float("nan"))
        d_left, d_right = drops

        # accumulated |turn| across the shoulders (gate (d))
        a = jl if jl is not None else i0
        bb = jr if jr is not None else i1
        turn = float(np.sum(np.abs([_wrap180(head[k + 1] - head[k])
                                    for k in range(a, max(a, bb))]))) if bb > a else 0.0

        n_pts = int(min(prof["n_left"][i0:i1 + 1].min(), prof["n_right"][i0:i1 + 1].min()))

        lo, hi = p["width_ratio_band"]
        if not (lo <= ratio <= hi):
            reject.append("width_ratio_out_of_band")
        slo, shi = p["span_frac_band"]
        if not (slo <= span_frac <= shi):
            reject.append("span_too_short" if span_frac < slo else "span_too_long")
        if not (np.isfinite(d_left) and np.isfinite(d_right)):
            reject.append("side_drop_unmeasured")
        elif min(d_left, d_right) < p["side_drop_min"]:
            reject.append("one_sided_contraction")
        if turn > p["turn_hard_deg"]:
            reject.append("turning")

        scores = {
            "width_ratio": round(_tri(ratio, lo, hi), 4),
            "span": round(_tri(span_frac, slo, shi), 4),
            "symmetry": round(float(np.clip(1.0 - abs(d_left - d_right), 0.0, 1.0))
                              if np.isfinite(d_left) and np.isfinite(d_right) else 0.0, 4),
            "straight": round(float(np.clip(
                (p["turn_hard_deg"] - max(turn, p["turn_soft_deg"]))
                / max(p["turn_hard_deg"] - p["turn_soft_deg"], 1e-9), 0.0, 1.0)), 4),
            "density": round(float(np.clip(n_pts / max(p["trust_side_points"], 1),
                                           0.0, 1.0)), 4),
        }
        shape = [scores["width_ratio"], scores["span"], scores["symmetry"],
                 scores["straight"]]
        # geometric mean over the SHAPE axes (one bad axis cannot be hidden by three
        # good ones) times the density TRUST discount (not a shape defect, a doubt).
        conf = float(np.exp(np.mean(np.log(np.maximum(shape, 1e-12))))) * scores["density"]
        conf = 0.0 if reject else conf
        if conf < p["min_confidence"] and "below_min_confidence" not in reject:
            reject.append("below_min_confidence")

        # The event arclength is the CENTRE of the contraction, not its argmin pose: the
        # slab makes w flat across the door (the jamb stays the nearest structure for
        # +-lon_win), so argmin lands on the first of a tie and biased the reported s
        # 0.2 m early (measured on the synthetic suite). The centre is unbiased.
        s_mid = 0.5 * (s_lo + s_hi)
        cand = {
            "s": round(float(s_mid), 5),
            "time_s": (round(float(np.interp(float(s_mid), s, t)), 4) if t is not None
                       else None),
            "pose_index": int(np.argmin(np.abs(s - s_mid))),
            "pose_index_min": int(m),
            "width": round(w_min, 4),
            "baseline_width": round(b, 4),
            "width_ratio": round(float(ratio), 4) if np.isfinite(ratio) else None,
            "span": round(span, 4),
            "span_frac": round(float(span_frac), 4) if np.isfinite(span_frac) else None,
            "left_drop": round(float(d_left), 4) if np.isfinite(d_left) else None,
            "right_drop": round(float(d_right), 4) if np.isfinite(d_right) else None,
            "turn_deg": round(turn, 2),
            "n_points": n_pts,
            "confidence": round(conf, 4),
            "scores": scores,
            "accepted": not reject,
            "reject": reject,
        }
        out["candidates"].append(cand)

    out["candidates"].sort(key=lambda c: c["s"])
    out["doors"] = [c for c in out["candidates"] if c["accepted"]]
    info["n_candidates"] = len(out["candidates"])
    info["n_doors"] = len(out["doors"])
    return out


def door_s_values(result: dict, min_confidence=None) -> list:
    """Accepted detections -> arclengths for `coarse_match(door_s=...)` (recon units).

    `min_confidence` raises the bar above `DEFAULTS['min_confidence']`; it can never
    lower it — a rejected candidate carries a failed hard gate, not a low score, so
    promoting it would be exactly the fabrication this module refuses."""
    lo = 0.0 if min_confidence is None else float(min_confidence)
    return [float(c["s"]) for c in result.get("doors", []) if c["confidence"] >= lo]


def door_times(result: dict, min_confidence=None) -> list:
    """Accepted detections -> times in `pose_times`' units (video seconds), for the
    `--door-times` wiring. Raises if the detections carry no time axis."""
    ds = [c for c in result.get("doors", [])
          if c["confidence"] >= (0.0 if min_confidence is None else float(min_confidence))]
    if any(c["time_s"] is None for c in ds):
        raise ValueError("detections carry no time_s — pass pose_times to detect_doors()")
    return [float(c["time_s"]) for c in ds]


# --------------------------------------------------------------------------------------
# SECTION 4 — offline observation helper (reading a stored upload)
# --------------------------------------------------------------------------------------
# Production callers (build_coplay) already hold the gravity-aligned cloud and poses and
# should pass those straight to detect_doors(). This helper exists so a stored upload can
# be re-observed without a viewer; it mirrors, and does not re-invent, two existing
# conventions: the LBP2/3/4 layout written by realtime/inference_worker.py (also read by
# tools/validate_wall_anchor.parse_lbp2) and build_coplay's viewer_pose/place_gravity
# gravity alignment.

def read_lbp(path) -> dict:
    """Minimal LBP2/LBP3/LBP4 reader: {"magic","points" (M,3),"poses" (N,12)}.
    Layout: magic(4) flags(u32) num_pts(u32) num_poses(u32) seq(u32), then xyz f32x3,
    rgb u8x3, [normals f32x3 (LBP3/4)], [src u32 + component u32 (LBP4)], poses f32x12.
    The format carries NO timestamps — the pose time axis has to come from the caller
    (build_coplay uses t_i = duration*i/(n-1))."""
    data = Path(path).read_bytes()
    if len(data) < 20:
        raise ValueError(f"payload too small: {path}")
    magic, _flags, npts, nposes, _seq = struct.unpack("<4sIIII", data[:20])
    if magic not in (b"LBP2", b"LBP3", b"LBP4"):
        raise ValueError(f"unsupported payload magic {magic!r} in {path}")
    off = 20
    xyz = np.frombuffer(data, dtype="<f4", count=npts * 3, offset=off).reshape(-1, 3)
    off += npts * 12 + npts * 3
    if magic in (b"LBP3", b"LBP4"):
        off += npts * 12
    if magic == b"LBP4":
        off += npts * 8
    poses = np.frombuffer(data, dtype="<f4", count=nposes * 12, offset=off).reshape(-1, 12)
    return {"magic": magic.decode("ascii"), "points": xyz.astype(np.float64),
            "poses": poses.astype(np.float64)}


def gravity_align(poses12, points) -> dict:
    """(N,12) recon extrinsics + (M,3) cloud -> {"traj_xz","cam_y","pts_yup"}.

    Identical algebra to tools/build_coplay.viewer_pose + place_gravity's gravity step
    (camera centre -R^T t, up = R^T col1, cloud Y/Z flipped into the viewer frame, then
    the mean up rotated onto +Y). No scaling and no model offset are applied: this module
    works in RECON units."""
    c = np.asarray(poses12, dtype=np.float64).reshape(-1, 12)
    R = np.empty((len(c), 3, 3))
    R[:, 0] = np.stack([c[:, 0], -c[:, 1], -c[:, 2]], axis=1)
    R[:, 1] = np.stack([-c[:, 4], c[:, 5], c[:, 6]], axis=1)
    R[:, 2] = np.stack([-c[:, 8], c[:, 9], c[:, 10]], axis=1)
    t = np.stack([c[:, 3], -c[:, 7], -c[:, 11]], axis=1)
    Rt = np.transpose(R, (0, 2, 1))
    centers = -np.einsum("nij,nj->ni", Rt, t)
    up = Rt[:, :, 1]
    g = up.mean(0)
    g = g / (np.linalg.norm(g) + 1e-12)
    target = np.array([0.0, 1.0, 0.0])
    v = np.cross(g, target)
    cc = float(g @ target)
    if np.linalg.norm(v) < 1e-9:
        Rg = np.eye(3) if cc > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        Rg = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + cc))
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3).copy()
    if P.size:
        P[:, 1] *= -1.0
        P[:, 2] *= -1.0
        P = P @ Rg.T
    C = centers @ Rg.T
    return {"traj_xz": C[:, [0, 2]], "cam_y": C[:, 1], "pts_yup": P}

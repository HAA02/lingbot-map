"""`scan2bim.door_detect` — automatic door-passing detection from the recon alone.

The fixtures are SYNTHETIC and constructive, so the answer is known exactly: a corridor
of the team's measured 1.82 m clear width (`test_coarse_match._corridor_walls`, extended
here from wall SEGMENTS to a sampled 3-D CLOUD by `_sample_segs`), doorway cross-walls
whose clear opening is chosen from the door standard, and a walk down the centreline.
Nothing in this file — or in the module — reads a DXF, a BIM model or a matcher result:
door evidence derived from the drawing would make "the door disambiguates the drawing"
circular.

WHAT IS PINNED HERE
-------------------
 1. the profile measures the KNOWN corridor width (1.82 m within 0.08 m) and the height
    band really does exclude the floor/ceiling slabs and the door LINTEL;
 2. recall / precision on three doorways of 0.80 / 0.90 / 1.00 m: 3/3 detected, no
    accepted false positive, arclength error <= 0.05 m;
 3. every adversarial shape the module claims to reject is actually rejected, WITH the
    gate that did it: a one-sided 0.75 m riser (whose width ratio 0.58 is INSIDE the door
    band, so only two-sidedness catches it), a 90 deg corner, a 1.6 m wide opening, a 3 m
    narrow neck, a dead end, and a stretch where the cloud is missing;
 4. noise tolerance, MEASURED not asserted-away: sigma 0.00/0.02/0.05 m -> recall 3/3
    with zero false positives, sigma 0.10 m -> 2/3 (see TestNoiseTolerance for the table);
 5. scale-freeness: the same corridor pushed through the anisotropic recon transform
    (s_f 3.53 / s_h 1.97, chi -1) yields the SAME detection, so the module works in
    monocular recon units;
 6. THE REASON THIS MODULE EXISTS — an auto-detected door turns the 2-leg L fixture from
    `coarse_match` HOLD 'ambiguous_margin' (margin 0.0) into a confirmed 'ok'
    (margin 0.22), with the transform recovered exactly. TestBreaksTheAmbiguity.

NOTE: no real upload is exercised here; the fixtures under realtime/_uploads are
symlinked, read-only and carry no ground truth, so they are OBSERVED in the cycle report
instead of asserted on.
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scan2bim import coarse_match as cm
from scan2bim import door_detect as dd
from scan2bim import plan_skeleton as ps
from test_coarse_match import TRUE, _corridor_walls, _densify, _make_walk, _to_recon
from test_plan_skeleton import CENTER_A, CENTER_B, W, _l_corridor_segments

#: floor / ceiling / door-lintel heights (m). The lintel is the KS F 3109 standard leaf
#: height the module's band has to stay under.
FLOOR, CEIL, LINTEL = 0.0, 2.6, 2.1
EYE = 1.5                       # camera carry height (metric_scale's own default)
STEP = 0.10                     # pose spacing (m) — the walk, not a polyline
S_TOL = 0.35                    # a detection counts as a hit within this arclength (m)


# --------------------------------------------------------------------------------------
# fixture builders: wall SEGMENTS -> sampled 3-D cloud
# --------------------------------------------------------------------------------------

def _sample_segs(segs, h_lo, h_hi, ds: float = 0.03, dh: float = 0.05, sigma: float = 0.0,
                 rng=None) -> np.ndarray:
    """(N,2,2) plan-metre wall segments -> (M,3) gravity-aligned Y-up cloud, sampled
    every `ds` along the wall and every `dh` in height over [h_lo, h_hi]. Plan (x, y)
    becomes (X, Z); height becomes Y. `sigma` adds isotropic gaussian noise (m)."""
    out = []
    hs = np.arange(float(h_lo), float(h_hi) + 1e-9, float(dh))
    for a, b in np.asarray(segs, dtype=np.float64).reshape(-1, 2, 2):
        L = float(np.linalg.norm(b - a))
        m = max(2, int(np.ceil(L / float(ds))) + 1)
        line = a + (b - a) * np.linspace(0.0, 1.0, m)[:, None]
        rep = np.repeat(line, len(hs), axis=0)
        out.append(np.stack([rep[:, 0], np.tile(hs, len(line)), rep[:, 1]], axis=1))
    pts = np.vstack(out) if out else np.zeros((0, 3))
    if sigma > 0.0:
        pts = pts + (rng or np.random.default_rng(0)).normal(0.0, float(sigma), pts.shape)
    return pts


def _doorway(centre, along, clear: float, half: float = W / 2.0) -> list:
    """Cross-wall stubs of a doorway: the wall runs through `centre` along `along` and
    leaves a `clear` metre opening centred on the walk. `half` is how far each stub
    reaches (the corridor half width, so the stubs meet the side walls)."""
    c = np.asarray(centre, dtype=np.float64)
    u = np.asarray(along, dtype=np.float64)
    u = u / np.linalg.norm(u)
    return [(c - half * u, c - 0.5 * float(clear) * u),
            (c + 0.5 * float(clear) * u, c + half * u)]


def _cloud(spine, doors=(), extra=(), sigma: float = 0.0, seed: int = 0, walls=None,
           slabs: bool = False):
    """(cloud, trajectory, cam_y) for a corridor walk.

    doors  [(arclength_position_xy, along_dir, clear_width), ...] cross-wall doorways;
           each also gets the SOLID LINTEL above it (the band must exclude it, else the
           passage reads as zero width exactly where the door is).
    extra  [(segments, h_lo, h_hi), ...] arbitrary added geometry (columns, necks).
    slabs  add floor and ceiling planes (they must not affect the width)."""
    rng = np.random.default_rng(seed)
    segs = _corridor_walls(spine, width=W) if walls is None else np.asarray(walls)
    parts = [_sample_segs(segs, FLOOR + 0.05, CEIL, sigma=sigma, rng=rng)]
    for centre, along, clear in doors:
        parts.append(_sample_segs(_doorway(centre, along, clear), FLOOR + 0.05, LINTEL,
                                  sigma=sigma, rng=rng))
        u = np.asarray(along, dtype=np.float64) / np.linalg.norm(along)
        c = np.asarray(centre, dtype=np.float64)
        parts.append(_sample_segs([(c - W / 2 * u, c + W / 2 * u)], LINTEL, CEIL,
                                  sigma=sigma, rng=rng))
    for segs_x, lo, hi in extra:
        parts.append(_sample_segs(segs_x, lo, hi, sigma=sigma, rng=rng))
    if slabs:
        c = np.asarray(spine, dtype=np.float64).reshape(-1, 2)
        for h in (FLOOR, CEIL):
            grid = []
            for a, b in zip(c, c[1:]):
                u = (b - a) / np.linalg.norm(b - a)
                n = np.array([-u[1], u[0]])
                t = np.linspace(0.0, float(np.linalg.norm(b - a)), 60)
                o = np.linspace(-W / 2, W / 2, 12)
                pp = (a[None, None, :] + t[:, None, None] * u + o[None, :, None] * n)
                grid.append(pp.reshape(-1, 2))
            g = np.vstack(grid)
            parts.append(np.stack([g[:, 0], np.full(len(g), h), g[:, 1]], axis=1))
    traj = _densify(spine, STEP)
    return np.vstack(parts), traj, np.full(len(traj), EYE)


def _arclen_of(spine, xy) -> float:
    """True arclength of a point on the (axis-aligned) spine — the ground truth an
    arclength detection is compared against."""
    c = np.asarray(spine, dtype=np.float64).reshape(-1, 2)
    p = np.asarray(xy, dtype=np.float64).reshape(2)
    s = 0.0
    for a, b in zip(c, c[1:]):
        L = float(np.linalg.norm(b - a))
        u = (b - a) / L
        t = float((p - a) @ u)
        if -1e-9 <= t <= L + 1e-9 and abs(float((p - a) @ np.array([-u[1], u[0]]))) < 1e-6:
            return s + t
        s += L
    raise AssertionError(f"{xy} is not on the spine {spine}")


def _score(res: dict, truth_s, tol: float = S_TOL):
    """(recall, precision, per-truth arclength errors) of the ACCEPTED detections."""
    got = [c["s"] for c in res["doors"]]
    errs = [min((abs(g - t) for g in got), default=float("inf")) for t in truth_s]
    hits = [t for t, e in zip(truth_s, errs) if e <= tol]
    tp = sum(1 for g in got if min(abs(g - t) for t in truth_s) <= tol)
    recall = len(hits) / max(len(truth_s), 1)
    precision = (tp / len(got)) if got else 1.0
    return recall, precision, errs


# --------------------------------------------------------------------------------------
# 1 — the profile itself (so a later failure cannot be blamed on the measurement)
# --------------------------------------------------------------------------------------

STRAIGHT = [(0.0, 0.0), (0.0, 24.0)]
DOORS3 = [((0.0, 5.0), (1.0, 0.0), 0.90),
          ((0.0, 12.0), (1.0, 0.0), 0.80),
          ((0.0, 19.0), (1.0, 0.0), 1.00)]
TRUTH3 = [5.0, 12.0, 19.0]


class TestLateralProfile(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cloud, cls.traj, cls.cam = _cloud(STRAIGHT, DOORS3)

    def test_measures_the_known_corridor_width(self):
        prof = dd.lateral_profile(self.traj, self.cloud, cam_y=self.cam)
        self.assertEqual(prof["info"]["band_source"], "pose_y")
        self.assertTrue(prof["measured"].all(), prof["info"]["coverage"])
        med = float(np.nanmedian(prof["width"]))
        self.assertAlmostEqual(med, W, delta=0.08,
                               msg=f"median clear width {med:.3f} != {W} +-0.08")

    def test_floor_and_ceiling_slabs_do_not_narrow_the_passage(self):
        """The height band's whole job: a floor point sits at lateral ~0 and would read
        as a wall right next to the camera."""
        base = dd.lateral_profile(self.traj, self.cloud, cam_y=self.cam)
        cloud2, traj2, cam2 = _cloud(STRAIGHT, DOORS3, slabs=True)
        with_slabs = dd.lateral_profile(traj2, cloud2, cam_y=cam2)
        self.assertGreater(len(cloud2), len(self.cloud))
        self.assertAlmostEqual(float(np.nanmedian(with_slabs["width"])),
                               float(np.nanmedian(base["width"])), delta=0.02)

    def test_the_door_lintel_is_excluded_too(self):
        """Above a 2.1 m lintel the cross wall is SOLID across the corridor. If the band
        reached it, w at the door would collapse to ~0 and the door would be rejected as
        'width_ratio_out_of_band' instead of detected."""
        prof = dd.lateral_profile(self.traj, self.cloud, cam_y=self.cam)
        i = int(np.argmin(np.abs(prof["s"] - 5.0)))
        self.assertGreater(float(prof["width"][i]), 0.6, "lintel leaked into the band")

    def test_a_missing_stretch_is_unmeasured_not_guessed(self):
        keep = ~((self.cloud[:, 2] > 9.0) & (self.cloud[:, 2] < 10.5))
        prof = dd.lateral_profile(self.traj, self.cloud[keep], cam_y=self.cam)
        gap = (prof["s"] > 9.3) & (prof["s"] < 10.2)
        self.assertFalse(prof["measured"][gap].any(), "a hole in the cloud became a width")
        self.assertTrue(np.isnan(prof["width"][gap]).all())
        self.assertLess(prof["info"]["coverage"]["measured_frac"], 1.0)

    def test_band_falls_back_and_says_so_without_cam_y(self):
        prof = dd.lateral_profile(self.traj, self.cloud)
        self.assertEqual(prof["info"]["band_source"], "cloud_midband")

    def test_degenerate_inputs_measure_nothing(self):
        for traj, pts in (([[0.0, 0.0]], self.cloud),
                          (self.traj, np.zeros((0, 3))),
                          ([[1.0, 1.0]] * 5, self.cloud)):
            prof = dd.lateral_profile(traj, pts)
            self.assertEqual(len(prof["s"]), 0, prof["info"])
            self.assertIn("fail", prof["info"])

    def test_cam_y_length_is_checked(self):
        with self.assertRaises(ValueError):
            dd.lateral_profile(self.traj, self.cloud, cam_y=[EYE, EYE])


# --------------------------------------------------------------------------------------
# 2 — recall / precision on known doorways
# --------------------------------------------------------------------------------------

class TestRecallPrecision(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cloud, cls.traj, cls.cam = _cloud(STRAIGHT, DOORS3)
        cls.res = dd.detect_doors(cls.traj, cls.cloud, cls.cam,
                                  pose_times=np.linspace(0.0, 48.0, len(cls.traj)))

    def test_all_three_doorways_are_found_with_no_false_positive(self):
        recall, precision, errs = _score(self.res, TRUTH3)
        self.assertEqual((recall, precision), (1.0, 1.0),
                         [(c["s"], c["confidence"], c["reject"]) for c in self.res["candidates"]])
        self.assertLessEqual(max(errs), 0.05, f"arclength errors {errs}")

    def test_reported_widths_match_the_built_openings(self):
        for (centre, _along, clear) in DOORS3:
            s = _arclen_of(STRAIGHT, centre)
            c = min(self.res["doors"], key=lambda d: abs(d["s"] - s))
            self.assertAlmostEqual(c["width"], clear, delta=0.10,
                                   msg=f"opening {clear} m read as {c['width']} m")
            self.assertGreater(c["confidence"], dd.DEFAULTS["min_confidence"])

    def test_times_come_from_the_pose_time_axis(self):
        times = dd.door_times(self.res)
        self.assertEqual(len(times), 3)
        s = self.res["profile"]["s"]
        for c, t in zip(self.res["doors"], times):
            self.assertAlmostEqual(t, 48.0 * c["s"] / s[-1], delta=0.2)

    def test_door_times_refuses_without_a_time_axis(self):
        res = dd.detect_doors(self.traj, self.cloud, self.cam)
        self.assertTrue(res["doors"])
        self.assertIsNone(res["doors"][0]["time_s"])
        with self.assertRaises(ValueError):
            dd.door_times(res)
        self.assertEqual(len(dd.door_s_values(res)), 3)

    def test_pose_times_length_is_checked(self):
        with self.assertRaises(ValueError):
            dd.detect_doors(self.traj, self.cloud, self.cam, pose_times=[0.0, 1.0])

    def test_result_is_json_safe_and_deterministic(self):
        again = dd.detect_doors(self.traj, self.cloud, self.cam,
                                pose_times=np.linspace(0.0, 48.0, len(self.traj)))
        self.assertEqual(json.dumps(again, sort_keys=True), json.dumps(self.res, sort_keys=True))
        self.assertEqual(self.res["schema"], dd.SCHEMA)

    def test_candidates_are_sorted_and_carry_the_full_audit_row(self):
        ss = [c["s"] for c in self.res["candidates"]]
        self.assertEqual(ss, sorted(ss))
        for c in self.res["candidates"]:
            for k in ("s", "width", "baseline_width", "width_ratio", "span", "span_frac",
                      "turn_deg", "n_points", "confidence", "scores", "accepted", "reject"):
                self.assertIn(k, c)
            self.assertEqual(c["accepted"], not c["reject"])

    def test_an_unknown_parameter_is_refused(self):
        with self.assertRaises(ValueError):
            dd.detect_doors(self.traj, self.cloud, self.cam, params={"door_frac": 0.5})


# --------------------------------------------------------------------------------------
# 3 — the false positives the module claims to reject
# --------------------------------------------------------------------------------------

class TestNotADoor(unittest.TestCase):
    """Each fixture pairs the adversarial shape with a REAL doorway, so 'rejects
    everything' cannot pass as 'rejects the right thing'."""

    def _run(self, spine, doors, extra=(), walls=None, drop=None):
        cloud, traj, cam = _cloud(spine, doors, extra=extra, walls=walls)
        if drop is not None:
            lo, hi = drop
            cloud = cloud[~((cloud[:, 2] > lo) & (cloud[:, 2] < hi))]
        return dd.detect_doors(traj, cloud, cam)

    def _at(self, res, s, tol=0.6):
        return [c for c in res["candidates"] if abs(c["s"] - s) <= tol]

    def test_a_one_sided_riser_is_not_a_door(self):
        """0.75 m deep pilaster: w 1.07 m -> ratio 0.58, INSIDE the door band. Only the
        two-sided requirement can reject it, which is why that gate exists."""
        x = W / 2.0
        col = [((x - 0.75, 7.65), (x - 0.75, 8.35)), ((x - 0.75, 7.65), (x, 7.65)),
               ((x - 0.75, 8.35), (x, 8.35))]
        res = self._run([(0.0, 0.0), (0.0, 16.0)], [DOORS3[0]], extra=[(col, 0.05, CEIL)])
        self.assertEqual([round(c["s"], 2) for c in res["doors"]], [5.0])
        bad = self._at(res, 8.0)
        self.assertTrue(bad, "the riser must still be REPORTED, not hidden")
        self.assertFalse(any(c["accepted"] for c in bad), bad)
        self.assertIn("one_sided_contraction", bad[0]["reject"])
        self.assertGreaterEqual(bad[0]["width_ratio"], dd.DEFAULTS["width_ratio_band"][0])
        self.assertLess(min(bad[0]["left_drop"], bad[0]["right_drop"]),
                        dd.DEFAULTS["side_drop_min"])

    def test_a_90_degree_corner_is_not_a_door(self):
        res = self._run([(0.0, 0.0), (0.0, 12.0), (12.0, 12.0)], [((0.0, 4.0), (1.0, 0.0), 0.9)])
        self.assertEqual([round(c["s"], 2) for c in res["doors"]], [4.0])
        self.assertFalse([c for c in res["doors"] if abs(c["s"] - 12.0) <= 1.5],
                         "the corner was accepted as a door")

    def test_a_doorway_inside_a_turn_is_refused_not_guessed(self):
        """A real doorway 0.3 m before a 90 deg corner: the lateral axis swings through
        it, so the measurement is untrustworthy and the module REFUSES (turn gate)
        rather than emitting a time it cannot stand behind."""
        res = self._run([(0.0, 0.0), (0.0, 10.0), (10.0, 10.0)],
                        [((0.0, 9.7), (1.0, 0.0), 0.9)])
        near = self._at(res, 9.7, tol=1.0)
        self.assertTrue(near, "the contraction must be reported")
        self.assertFalse(any(c["accepted"] for c in near), near)
        self.assertIn("turning", near[0]["reject"])
        self.assertGreater(near[0]["turn_deg"], dd.DEFAULTS["turn_hard_deg"])

    def test_a_wide_opening_is_not_a_door(self):
        res = self._run([(0.0, 0.0), (0.0, 16.0)],
                        [DOORS3[0], ((0.0, 10.0), (1.0, 0.0), 1.60)])
        self.assertEqual([round(c["s"], 2) for c in res["doors"]], [5.0])
        for c in self._at(res, 10.0):
            self.assertFalse(c["accepted"], c)

    def test_a_three_metre_narrow_neck_is_not_a_door(self):
        neck = [((-0.45, 8.0), (-0.45, 11.0)), ((0.45, 8.0), (0.45, 11.0)),
                ((-W / 2, 8.0), (-0.45, 8.0)), ((0.45, 8.0), (W / 2, 8.0)),
                ((-W / 2, 11.0), (-0.45, 11.0)), ((0.45, 11.0), (W / 2, 11.0))]
        res = self._run([(0.0, 0.0), (0.0, 18.0)], [((0.0, 4.0), (1.0, 0.0), 0.9)],
                        extra=[(neck, 0.05, CEIL)])
        self.assertEqual([round(c["s"], 2) for c in res["doors"]], [4.0])
        mid = self._at(res, 9.5, tol=1.6)
        self.assertTrue(mid)
        self.assertFalse(any(c["accepted"] for c in mid), mid)
        self.assertIn("span_too_long", mid[0]["reject"])

    def test_a_dead_end_is_not_a_door(self):
        """`_corridor_walls` caps both ends; the cap contracts BOTH sides to ~0 and never
        recovers, which is the difference between a door and a wall."""
        res = self._run(STRAIGHT, DOORS3)
        ends = [c for c in res["candidates"] if c["s"] < 1.0 or c["s"] > 23.0]
        self.assertEqual(len(ends), 2, [c["s"] for c in res["candidates"]])
        for c in ends:
            self.assertFalse(c["accepted"], c)
            self.assertIn("no_recovery", c["reject"])

    def test_a_sparse_stretch_yields_no_confident_door(self):
        res = self._run([(0.0, 0.0), (0.0, 18.0)],
                        [DOORS3[0], ((0.0, 12.0), (1.0, 0.0), 0.90)],
                        drop=(11.2, 12.8))
        self.assertEqual([round(c["s"], 2) for c in res["doors"]], [5.0],
                         "a door whose cloud is missing must not be invented")
        self.assertLess(res["info"]["coverage"]["measured_frac"], 1.0)

    def test_a_smeared_cloud_is_warned_about_not_reported_as_narrow_doors(self):
        """The failure mode MEASURED on realtime/_uploads/upload_1781521406685: the recon
        puts points ON the camera path, so the nearest "structure" is 0.014 recon units
        away and the median passage width reads 0.027 where the corridor is 0.924. Here
        that is reproduced synthetically — the module must WARN and find nothing, never
        report a corridor full of 5 cm doors."""
        cloud, traj, cam = _cloud(STRAIGHT, DOORS3)
        rng = np.random.default_rng(5)
        k = len(cloud) // 3
        j = rng.integers(0, len(traj), k)
        smear = np.stack([traj[j, 0] + rng.normal(0.0, 0.10, k),
                          EYE + rng.normal(0.0, 0.30, k),
                          traj[j, 1] + rng.normal(0.0, 0.10, k)], axis=1)
        mix = np.vstack([cloud, smear])
        # WITH a scale prior the band is right and the widths collapse: warn + 0 doors.
        # (The real upload reports the same signature, width_vs_w0 0.0296.)
        hinted = dd.detect_doors(traj, mix, cam, width_hint=W)
        cov = hinted["info"]["coverage"]
        self.assertLess(cov["width_vs_w0"], dd.SMEAR_WARN_FRAC, cov)
        self.assertIn("warn", hinted["info"])
        self.assertEqual(hinted["doors"], [],
                         f"smear became doors: {[(c['s'], c['width']) for c in hinted['doors']]}")
        # WITHOUT one the smear also poisons the self-calibration; that too must be said
        # out loud rather than produce a corridor full of 5 cm doors.
        blind = dd.detect_doors(traj, mix, cam)
        self.assertEqual(blind["doors"], [])
        self.assertTrue(blind["info"].get("fail") or blind["info"].get("warn"),
                        blind["info"]["coverage"])

    def test_a_clean_cloud_is_not_warned_about(self):
        cloud, traj, cam = _cloud(STRAIGHT, DOORS3)
        res = dd.detect_doors(traj, cloud, cam)
        self.assertNotIn("warn", res["info"], res["info"]["coverage"])
        self.assertGreater(res["info"]["coverage"]["width_vs_w0"], 0.9)

    def test_a_low_density_door_is_discounted_by_the_density_score(self):
        """Same doorway, 1/5 of the points: still found, but the confidence carries the
        doubt instead of hiding it. (Thinning to 1/40 puts the sides below
        `min_side_points` and the stretch becomes UNMEASURED — also correct, but that is
        what test_a_sparse_stretch_yields_no_confident_door pins.)"""
        cloud, traj, cam = _cloud([(0.0, 0.0), (0.0, 12.0)], [DOORS3[0]])
        dense = dd.detect_doors(traj, cloud, cam)
        rng = np.random.default_rng(3)
        thin = cloud[rng.random(len(cloud)) < 0.20]
        sparse = dd.detect_doors(traj, thin, cam)
        d0 = min(dense["candidates"], key=lambda c: abs(c["s"] - 5.0))
        s0 = min(sparse["candidates"], key=lambda c: abs(c["s"] - 5.0))
        self.assertLess(s0["n_points"], d0["n_points"])
        self.assertLess(s0["scores"]["density"], d0["scores"]["density"])
        self.assertLessEqual(s0["confidence"], d0["confidence"])


# --------------------------------------------------------------------------------------
# 4 — noise tolerance (measured, then asserted)
# --------------------------------------------------------------------------------------

class TestNoiseTolerance(unittest.TestCase):
    """MEASURED table (straight 24 m corridor, 3 doorways, isotropic gaussian cloud
    noise, seed 7 / 11), reproduced by the two tests below:

        sigma (m)   recall   precision   |ds| max (m)   median width (m)   w0 (m)
        0.00        3/3      1.00        0.000          1.780              1.820
        0.02        3/3      1.00        0.002          1.760              1.697
        0.05        3/3      1.00        0.030          1.556              1.501
        0.10        3/3      1.00        0.030          1.390              1.173
        0.15        1/3      1.00        0.053          1.210              0.851
        0.20        0/3      1.00        -              0.283              0.531

    Recall degrades from sigma 0.15 m and the mechanism is visible in the last two
    columns: the noise skirt pulls the density ONSET inward, every measured width shrinks
    (1.82 -> 1.21), and doors start falling out of `width_ratio_band`. PRECISION never
    degrades - it is 1.00 at every level, including the one where nothing is found at all.
    That asymmetry is deliberate: a wrong door time would confirm a wrong placement,
    while a missing one only leaves the matcher in HOLD."""

    #: sigma -> (recall, precision) as measured above; asserted as a FLOOR on recall.
    TABLE = {0.00: (1.0, 1.0), 0.02: (1.0, 1.0), 0.05: (1.0, 1.0), 0.10: (1.0, 1.0)}

    def test_measured_table(self):
        for sigma, (recall_exp, prec_exp) in self.TABLE.items():
            cloud, traj, cam = _cloud(STRAIGHT, DOORS3, sigma=sigma, seed=7)
            res = dd.detect_doors(traj, cloud, cam)
            recall, precision, errs = _score(res, TRUTH3)
            msg = (f"sigma {sigma}: recall {recall:.3f} precision {precision:.3f} "
                   f"errs {[round(e, 3) for e in errs]} "
                   f"medw {res['info']['coverage']['median_width']}")
            self.assertGreaterEqual(recall, recall_exp - 1e-9, msg)
            self.assertEqual(precision, prec_exp, msg)
            self.assertLessEqual(max(e for e in errs if e <= S_TOL), 0.06, msg)

    def test_precision_never_degrades_into_invention(self):
        """Past the recall cliff the module must go QUIET, not creative."""
        for sigma, recall_exp in ((0.15, 1 / 3), (0.20, 0.0)):
            cloud, traj, cam = _cloud(STRAIGHT, DOORS3, sigma=sigma, seed=11)
            res = dd.detect_doors(traj, cloud, cam)
            recall, precision, _ = _score(res, TRUTH3)
            self.assertEqual(precision, 1.0,
                             f"sigma {sigma} invented a door: "
                             f"{[(c['s'], c['confidence']) for c in res['doors']]}")
            self.assertAlmostEqual(recall, recall_exp, delta=1 / 3 + 1e-9,
                                   msg=f"sigma {sigma} recall {recall:.3f}")


# --------------------------------------------------------------------------------------
# 5 — scale-freeness (the module must work in monocular recon units)
# --------------------------------------------------------------------------------------

L_SPINE = [(CENTER_A, 0.0), (CENTER_A, CENTER_B), (10.0, CENTER_B)]
L_DOOR_XY = (CENTER_A, 4.0)
L_DOOR_CLEAR = 0.90


def _l_cloud_metric(sigma: float = 0.0, seed: int = 0):
    """The L corridor of tests/test_plan_skeleton (the SAME walls the plan skeleton is
    built from) plus a doorway across corridor A at y = 4.0, as a metric cloud + walk."""
    rng = np.random.default_rng(seed)
    parts = [_sample_segs(_l_corridor_segments(), FLOOR + 0.05, CEIL, sigma=sigma, rng=rng),
             _sample_segs(_doorway(L_DOOR_XY, (1.0, 0.0), L_DOOR_CLEAR), FLOOR + 0.05,
                          LINTEL, sigma=sigma, rng=rng),
             _sample_segs([((0.0, 4.0), (W, 4.0))], LINTEL, CEIL, sigma=sigma, rng=rng)]
    return np.vstack(parts), _densify(L_SPINE, STEP)


def _l_cloud_recon(sigma: float = 0.0, seed: int = 0):
    """The same thing in RECON units: XZ through the inverse of the true transform
    (anisotropic s_f 3.53 / s_h 1.97, chi -1) and heights divided by s_h, i.e. a recon
    whose vertical and lateral scales agree (s_v ~ s_h, the pipeline's own band)."""
    metric, _ = _l_cloud_metric(sigma=sigma, seed=seed)
    tf, traj, _plan_walk, _ = _make_walk(L_SPINE, step=STEP)
    xz = _to_recon(tf, metric[:, [0, 2]])
    cloud = np.stack([xz[:, 0], metric[:, 1] / TRUE["s_h"], xz[:, 1]], axis=1)
    return tf, cloud, traj, np.full(len(traj), EYE / TRUE["s_h"])


class TestScaleFree(unittest.TestCase):

    def test_the_same_door_is_found_in_metres_and_in_recon_units(self):
        metric, traj_m = _l_cloud_metric()
        res_m = dd.detect_doors(traj_m, metric, np.full(len(traj_m), EYE))
        _tf, cloud_r, traj_r, cam_r = _l_cloud_recon()
        res_r = dd.detect_doors(traj_r, cloud_r, cam_r)
        self.assertEqual(len(res_m["doors"]), 1, [c["s"] for c in res_m["candidates"]])
        self.assertEqual(len(res_r["doors"]), 1, [c["s"] for c in res_r["candidates"]])
        s_true = _arclen_of(L_SPINE, L_DOOR_XY)
        self.assertAlmostEqual(res_m["doors"][0]["s"], s_true, delta=0.05)
        # recon arclength x s_f is the same event, in metres
        self.assertAlmostEqual(res_r["doors"][0]["s"] * TRUE["s_f"], s_true, delta=0.05)

    def test_the_local_baseline_survives_two_corridors_of_different_recon_width(self):
        """The anisotropy makes the L's two legs 0.92 and 0.52 recon units wide. A GLOBAL
        baseline would call the narrower leg a 3 m door; the rolling median does not."""
        _tf, cloud_r, traj_r, cam_r = _l_cloud_recon()
        prof = dd.lateral_profile(traj_r, cloud_r, cam_y=cam_r)
        s = prof["s"]
        legA = prof["measured"] & (s < 4.0 / TRUE["s_f"])
        legB = prof["measured"] & (s > (CENTER_B + 2.0) / TRUE["s_f"])
        wA = float(np.median(prof["width"][legA])) * TRUE["s_h"]
        wB = float(np.median(prof["width"][legB])) * TRUE["s_f"]
        self.assertAlmostEqual(wA, W, delta=0.10, msg=f"leg A width {wA:.3f} m")
        self.assertAlmostEqual(wB, W, delta=0.10, msg=f"leg B width {wB:.3f} m")
        self.assertEqual(len(dd.detect_doors(traj_r, cloud_r, cam_r)["doors"]), 1)

    def test_a_scale_prior_is_accepted_but_not_required(self):
        _tf, cloud_r, traj_r, cam_r = _l_cloud_recon()
        hinted = dd.detect_doors(traj_r, cloud_r, cam_r, width_hint=W / TRUE["s_h"])
        self.assertEqual(hinted["info"]["w0_source"], "width_hint")
        self.assertEqual(len(hinted["doors"]), 1, hinted["candidates"])
        self.assertAlmostEqual(hinted["doors"][0]["s"],
                               dd.detect_doors(traj_r, cloud_r, cam_r)["doors"][0]["s"],
                               delta=0.05)


# --------------------------------------------------------------------------------------
# 6 — WHY THIS MODULE EXISTS: the detected door breaks the matcher's tie
# --------------------------------------------------------------------------------------

class TestBreaksTheAmbiguity(unittest.TestCase):
    """`coarse_match` measured the 2-leg L to be genuinely ambiguous (margin 0.0, HOLD)
    and a DOOR to be the only evidence that breaks it. Here that door is produced by
    `door_detect` from the recon cloud alone — no plan, no annotation, no hand-set time."""

    @classmethod
    def setUpClass(cls):
        cls.tf, cls.cloud, cls.traj, cls.cam = _l_cloud_recon()
        cls.det = dd.detect_doors(cls.traj, cls.cloud, cls.cam)
        cls.skel = ps.corridor_skeleton(_l_corridor_segments(), doors=[(W, 4.0)])

    def test_the_fixture_is_ambiguous_without_door_evidence(self):
        res = cm.coarse_match(self.skel, self.traj)
        self.assertEqual((res["status"], res["hold_reason"]), ("hold", "ambiguous_margin"))
        self.assertEqual(res["margin"], 0.0)
        self.assertIsNone(res["best"])

    def test_the_detected_door_lands_where_the_walk_really_passed_one(self):
        self.assertEqual(len(self.det["doors"]), 1, self.det["candidates"])
        s_true = _arclen_of(L_SPINE, L_DOOR_XY) / TRUE["s_f"]      # recon units
        self.assertAlmostEqual(self.det["doors"][0]["s"], s_true, delta=0.02,
                               msg=f"{self.det['doors'][0]}")

    def test_the_auto_detected_door_confirms_the_placement(self):
        door_s = dd.door_s_values(self.det)
        res = cm.coarse_match(self.skel, self.traj, door_s=door_s)
        ok, errs = ps.validate_match_result(res)
        self.assertTrue(ok, errs)
        self.assertEqual(res["status"], "ok", (res["hold_reason"], res["info"]))
        self.assertGreaterEqual(res["margin"], res["gates"]["margin_min"])
        got = res["best"]["transform"]
        self.assertEqual(int(got["chi"]), int(self.tf["chi"]))
        self.assertLessEqual(abs(cm._wrap180(got["yaw_deg"] - self.tf["yaw_deg"])), 0.5)
        for k in ("s_f", "s_h"):
            self.assertLessEqual(abs(got[k] - self.tf[k]) / self.tf[k], 0.02, k)
        self.assertLessEqual(float(np.linalg.norm(np.asarray(got["translation"])
                                                 - np.asarray(self.tf["translation"]))),
                             0.1, got)

    def test_the_matcher_actually_uses_it_as_an_inlier(self):
        res = cm.coarse_match(self.skel, self.traj, door_s=dd.door_s_values(self.det))
        rev = cm.recon_events(self.traj, door_s=dd.door_s_values(self.det))
        door_idx = [e["index"] for e in rev if e["kind"] == "door"]
        matched = [ri for ri, _pid in res["best"]["matches"]]
        self.assertTrue(set(door_idx) <= set(matched),
                        f"the detected door was not an inlier: {res['best']['outliers']}")

    def test_no_detection_means_no_evidence_rather_than_a_guess(self):
        """Delete the doorway from the cloud: the detector returns nothing and the
        matcher goes straight back to HOLD. Nothing anywhere invents a door."""
        metric, _ = _l_cloud_metric()
        keep = ~((metric[:, 2] > 3.8) & (metric[:, 2] < 4.2))
        xz = _to_recon(self.tf, metric[keep][:, [0, 2]])
        cloud = np.stack([xz[:, 0], metric[keep][:, 1] / TRUE["s_h"], xz[:, 1]], axis=1)
        det = dd.detect_doors(self.traj, cloud, self.cam)
        self.assertEqual(det["doors"], [], det["candidates"])
        res = cm.coarse_match(self.skel, self.traj, door_s=dd.door_s_values(det))
        self.assertEqual((res["status"], res["hold_reason"]), ("hold", "ambiguous_margin"))

    def test_door_s_values_cannot_promote_a_rejected_candidate(self):
        rejected = [c for c in self.det["candidates"] if not c["accepted"]]
        self.assertTrue(rejected, "the fixture must exercise a rejected candidate")
        for lo in (0.0, 0.9):
            for s in dd.door_s_values(self.det, min_confidence=lo):
                self.assertTrue(any(abs(c["s"] - s) < 1e-9 for c in self.det["doors"]))
        self.assertEqual(dd.door_s_values(self.det, min_confidence=1.01), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

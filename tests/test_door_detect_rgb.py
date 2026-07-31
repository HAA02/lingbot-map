"""Ground truth for `scan2bim.door_detect_rgb`.

There is no annotated door in any upload, so the truth has to be MANUFACTURED: every
fixture here renders a walk through a scene whose geometry — and therefore whose door
crossing time — is known exactly, with a pinhole camera (`_render`, 40 lines, no
dependency beyond cv2), and asserts what the detector must and must not return.

The adversarial half is the point. Each of these is a thing a corridor really contains
and each is rejected by a DIFFERENT gate, so a threshold slackened to rescue one of them
shows up as a failure here:

    column         passed on one side          -> both edges share a side, never a pair
    side window    passed on one side          -> same
    wall poster    a door-shaped painted panel -> same
    facing posters straddle the walk, same T   -> no transverse member between them
    dead end       a door-shaped frame ahead   -> the divergence never completes
    turn           the walk swings 90 deg      -> the yaw gate

`TestConfidenceSeparation` measures the gap `min_confidence` sits in instead of asserting
it from taste, and `TestContract` pushes a detection all the way into
`coarse_match.door_arclengths`, which is the interface build_coplay's `--door-times` uses.
"""
import numpy as np
import pytest

from scan2bim import coarse_match
from scan2bim.door_detect_rgb import (
    SCHEMA, DEFAULTS, detect_doors, door_times, door_s_values, frame_profiles,
    mobius_fit, track_peaks,
)

SIZE = (640, 360)
HFOV = 65.0
FPS = 30.0
NEAR = 0.12


# ======================================================================================
# a pinhole renderer — the only source of ground truth in this file
# ======================================================================================

def _basis(cam):
    yaw, pit = np.radians(cam["yaw"]), np.radians(cam["pitch"])
    f = np.array([np.sin(yaw) * np.cos(pit), np.sin(pit), np.cos(yaw) * np.cos(pit)])
    r = np.cross(np.array([0.0, 1.0, 0.0]), f)
    r /= np.linalg.norm(r)
    return f, r, np.cross(f, r)


def _cam_coords(pts, cam):
    f, r, u = _basis(cam)
    q = np.asarray(pts, float).reshape(-1, 3) - np.asarray(cam["pos"], float)
    return np.stack([q @ r, q @ u, q @ f], axis=1)


def _to_px(c, size, hfov):
    w, h = size
    fx = 0.5 * w / np.tan(np.radians(0.5 * hfov))
    z = np.maximum(c[:, 2], 1e-6)
    return np.stack([0.5 * w + fx * c[:, 0] / z, 0.5 * h - fx * c[:, 1] / z], axis=1)


def _clip_near(poly):
    """Sutherland-Hodgman against z = NEAR, so geometry behind the camera cannot wrap."""
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ain, bin_ = a[2] >= NEAR, b[2] >= NEAR
        if ain:
            out.append(a)
        if ain != bin_:
            t = (NEAR - a[2]) / (b[2] - a[2])
            out.append(a + t * (b - a))
    return np.asarray(out) if out else np.zeros((0, 3))


def _render(scene, cams, size=SIZE, hfov=HFOV, noise=0.0, gain=None, seed=0):
    """(N,H,W) uint8 frames. `scene` = {"quads": [(pts,gray)], "segs": [(a,b,gray,th)]}."""
    import cv2
    rng = np.random.default_rng(seed)
    w, h = size
    frames = []
    for k, cam in enumerate(cams):
        img = np.full((h, w), 96, np.uint8)
        for pts, gray in scene.get("quads", []):
            c = _clip_near(_cam_coords(pts, cam))
            if len(c) >= 3:
                cv2.fillPoly(img, [np.round(_to_px(c, size, hfov)).astype(np.int32)],
                             int(gray), lineType=cv2.LINE_AA)
        for a, b, gray, th in scene.get("segs", []):
            c = _clip_near(np.asarray([a, b], float) if False else
                           _cam_coords([a, b], cam))
            if len(c) < 2:
                continue
            px = _to_px(c[:2], size, hfov)
            if not np.isfinite(px).all() or np.abs(px).max() > 20000:
                continue
            cv2.line(img, tuple(np.round(px[0]).astype(int)),
                     tuple(np.round(px[1]).astype(int)), int(gray), int(th),
                     lineType=cv2.LINE_AA)
        f = img.astype(np.float32)
        if gain is not None:
            f *= float(gain[k])
        if noise > 0:
            f += rng.normal(0.0, noise, f.shape)
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    return np.asarray(frames)


def _walk(n, v=1.0, fps=FPS, z0=0.0, pitch=10.0, yaw=0.0, height=1.5, stop_at=None):
    """Straight constant-speed walk down +Z, camera pitched up like a carried phone."""
    cams = []
    z = z0
    for i in range(n):
        if stop_at is None or z < stop_at:
            z = z0 + v * i / fps
            z = z if stop_at is None else min(z, stop_at)
        y = yaw(i / fps) if callable(yaw) else yaw
        cams.append({"pos": (0.0, height, z), "yaw": y, "pitch": pitch})
    return cams


def _corridor(z0=-3.0, z1=16.0, hw=0.91, ch=2.7, dz=2.6):
    """A corridor whose two walls carry MIRRORED vertical features.

    That is deliberately the worst case for gate (a): every stripe station puts a pair of
    edges on opposite sides of the walk with perfectly coincident crossing times, i.e. it
    looks like an opening in every respect except that nothing spans between the two. It
    is what forced `_lintel`'s span test, and it keeps forcing it."""
    segs = []
    for sx in (-1, 1):
        x = sx * hw
        for y in (0.0, 0.12, ch):                      # longitudinal: floor, skirting,
            segs.append(((x, y, z0), (x, y, z1), 40, 2))   # ceiling — static image lines
        for z in np.arange(z0 + 0.4, z1, dz):          # vertical wall stripes
            segs.append(((x, 0.0, z), (x, ch, z), 55, 2))
    segs.append(((-hw, ch, z0), (hw, ch, z0), 40, 2))
    return {"quads": [((( -hw, 0, z0), (-hw, ch, z0), (-hw, ch, z1), (-hw, 0, z1)), 150),
                      (((hw, 0, z0), (hw, ch, z0), (hw, ch, z1), (hw, 0, z1)), 150),
                      (((-hw, 0, z0), (hw, 0, z0), (hw, 0, z1), (-hw, 0, z1)), 70),
                      (((-hw, ch, z0), (hw, ch, z0), (hw, ch, z1), (-hw, ch, z1)), 110)],
            "segs": segs}


def _add_door(scene, zd, hw=0.91, ch=2.7, half=0.45, head=2.1, gray=185):
    """A partition across the corridor with a walked opening: two jambs + a LINTEL."""
    scene["quads"] += [
        (((-hw, 0, zd), (-half, 0, zd), (-half, ch, zd), (-hw, ch, zd)), gray),
        (((half, 0, zd), (hw, 0, zd), (hw, ch, zd), (half, ch, zd)), gray),
        (((-half, head, zd), (half, head, zd), (half, ch, zd), (-half, ch, zd)), gray),
    ]
    scene["segs"] += [
        ((-half, 0, zd), (-half, head, zd), 25, 3),
        ((half, 0, zd), (half, head, zd), 25, 3),
        ((-half, head, zd), (half, head, zd), 25, 3),      # the lintel
        ((-hw, ch, zd), (hw, ch, zd), 40, 2),
    ]
    return scene


def _add_panel(scene, zc, side, w=0.9, y0=0.0, y1=2.1, hw=0.91, gray=205):
    """A door-shaped PAINTED panel on a side wall: the same rectangle, no opening."""
    x = side * hw
    p = (((x, y0, zc), (x, y1, zc), (x, y1, zc + w), (x, y0, zc + w)), gray)
    scene["quads"].append(p)
    scene["segs"] += [((x, y0, zc), (x, y1, zc), 25, 3),
                      ((x, y0, zc + w), (x, y1, zc + w), 25, 3),
                      ((x, y1, zc), (x, y1, zc + w), 25, 3)]
    return scene


def _add_column(scene, zc, xc=0.55, w=0.35, ch=2.7, gray=175):
    x0, x1, z0, z1 = xc - 0.5 * w, xc + 0.5 * w, zc, zc + w
    scene["quads"].append((((x0, 0, z0), (x1, 0, z0), (x1, ch, z0), (x0, ch, z0)), gray))
    scene["segs"] += [((x0, 0, z0), (x0, ch, z0), 20, 3),
                      ((x1, 0, z0), (x1, ch, z0), 20, 3),
                      ((x0, ch, z0), (x1, ch, z0), 20, 3),
                      ((x1, 0, z0), (x1, 0, z1), 20, 2)]
    return scene


def _add_endwall(scene, zd, hw=0.91, ch=2.7, half=0.45, head=2.1, gray=195):
    """A door-shaped frame on a wall the walk STOPS in front of."""
    scene["quads"].append((((-hw, 0, zd), (hw, 0, zd), (hw, ch, zd), (-hw, ch, zd)), gray))
    scene["segs"] += [((-half, 0, zd), (-half, head, zd), 25, 3),
                      ((half, 0, zd), (half, head, zd), 25, 3),
                      ((-half, head, zd), (half, head, zd), 25, 3)]
    return scene


def _detect(frames, **kw):
    return detect_doors(frames, fps=FPS, **kw)


# ======================================================================================
# fixtures shared across classes (rendering + detection is the expensive part)
# ======================================================================================

def _door_scene(zd=3.2, n=110, **render_kw):
    scene = _add_door(_corridor(), zd)
    return _render(scene, _walk(n), **render_kw)


@pytest.fixture(scope="module")
def one_door():
    return _detect(_door_scene())


@pytest.fixture(scope="module")
def two_doors():
    scene = _add_door(_add_door(_corridor(), 3.2), 9.0)
    return _detect(_render(scene, _walk(330)))


@pytest.fixture(scope="module")
def adversarial():
    """{name: result} for every scene that must NOT yield a door."""
    out = {}
    out["column"] = _detect(_render(_add_column(_corridor(), 3.2), _walk(110)))
    out["side_window"] = _detect(_render(
        _add_panel(_corridor(), 3.0, +1, w=1.6, y0=1.1, y1=2.3, gray=225), _walk(110)))
    out["wall_poster"] = _detect(_render(_add_panel(_corridor(), 3.0, -1), _walk(110)))
    facing = _add_panel(_add_panel(_corridor(), 3.0, -1), 3.0, +1)
    out["facing_posters"] = _detect(_render(facing, _walk(110)))
    out["dead_end"] = _detect(_render(_add_endwall(_corridor(z1=4.4), 4.4),
                                      _walk(150, stop_at=3.1)))
    out["turn"] = _detect(_render(
        _add_door(_corridor(), 3.2),
        _walk(110, yaw=lambda s: -70.0 * np.clip((s - 1.0) / 1.6, 0.0, 1.0))))
    return out


# ======================================================================================
class TestObservableLayer:
    """The pieces the detector is built out of, pinned on their own."""

    def test_mobius_recovers_the_crossing_time(self):
        # x(t) = foe + fx*X/(v*(T-t)) — the model the whole module rests on.
        t = np.linspace(0.0, 2.0, 40)
        T, foe, K = 3.0, 305.0, 260.0
        x = foe + K / (T - t)
        fit = mobius_fit(t, x)
        assert fit["T"] == pytest.approx(T, abs=1e-6)
        assert fit["foe"] == pytest.approx(foe, abs=1e-6)
        assert fit["K"] == pytest.approx(K, abs=1e-4)
        assert fit["r2"] > 0.999

    def test_mobius_survives_a_tracker_hiccup(self):
        t = np.linspace(0.0, 2.0, 40)
        x = 305.0 + 260.0 / (3.0 - t)
        x[7] += 45.0                                    # peak jumped to a neighbour
        x[22] -= 38.0
        assert mobius_fit(t, x)["T"] == pytest.approx(3.0, abs=0.15)

    def test_mobius_refuses_a_static_edge(self):
        t = np.linspace(0.0, 2.0, 40)
        x = np.full_like(t, 300.0) + np.random.default_rng(0).normal(0, 0.2, 40)
        fit = mobius_fit(t, x)
        assert not (np.isfinite(fit["r2"]) and fit["r2"] > 0.9 and fit["T"] < 4.0)

    def test_profiles_and_tracking_run_on_rendered_frames(self):
        prof = frame_profiles(_door_scene(n=60), fps=FPS)
        assert prof["col"].shape[0] == 60
        assert prof["col"].shape[1] == DEFAULTS["work_width"]
        assert prof["hmap"].shape[1:] == (DEFAULTS["lintel_rows"], DEFAULTS["lintel_cols"])
        assert prof["info"]["fps"] == pytest.approx(FPS)
        trs = track_peaks(prof["col"], gate=DEFAULTS["track_gate_px"],
                          gate_vel=DEFAULTS["track_gate_vel"], max_gap=3,
                          min_h=DEFAULTS["peak_min_height"],
                          prom=DEFAULTS["peak_prominence"], max_peaks=30, min_len=8)
        assert len(trs) >= 2, "a rendered corridor must give trackable vertical structure"

    def test_forward_motion_leaves_the_shift_gate_at_zero(self):
        # expansion is symmetric about the focus of expansion, so the yaw proxy must not
        # fire on a straight walk — otherwise every door would be rejected as a turn.
        prof = frame_profiles(_door_scene(n=60), fps=FPS)
        assert float(np.median(np.abs(prof["shift_px"]))) <= 1.5

    def test_yaw_moves_the_shift_gate(self):
        cams = _walk(60, yaw=lambda s: -40.0 * s)
        prof = frame_profiles(_render(_corridor(), cams), fps=FPS)
        assert float(np.median(np.abs(prof["shift_px"]))) > 2.0


# ======================================================================================
class TestDoorwayPass:
    def test_schema_and_shape(self, one_door):
        assert one_door["schema"] == SCHEMA
        for k in ("doors", "candidates", "profile", "info"):
            assert k in one_door
        assert one_door["info"]["n_frames"] == 110

    def test_detects_the_crossing(self, one_door):
        assert len(one_door["doors"]) == 1, one_door["info"].get("warn")
        d = one_door["doors"][0]
        assert d["time_s"] == pytest.approx(3.2, abs=0.30)   # zd 3.2 m at 1.0 m/s
        assert d["confidence"] >= DEFAULTS["min_confidence"]
        assert d["accepted"] and not d["reject"]

    def test_reports_its_evidence(self, one_door):
        d = one_door["doors"][0]
        assert d["lintel"]["found"] is True
        assert d["scores"]["lintel"] > 0.0
        assert 0.2 <= d["sym_frac"] <= 0.8
        assert min(d["r2"]) >= DEFAULTS["min_r2"]
        assert d["n_pair_frames"] >= DEFAULTS["lintel_min_frames"]

    def test_measures_the_opening_width(self, one_door):
        """|X|/v in seconds of walking; at 1.0 m/s a 0.90 m opening reads 0.90.

        The nested readings are the scale sanity check: ONE crossing yields the clear
        opening (0.90 m, the door-leaf standard) AND the partition's junction with the
        corridor walls (1.82 m, the corridor). Both are physically right; the module
        reports the constriction and keeps the rest."""
        d = one_door["doors"][0]
        assert d["width_px_s"] == pytest.approx(0.90, rel=0.25)
        widths = [n["width_px_s"] for n in d["nested"]]
        assert any(w == pytest.approx(1.82, rel=0.15) for w in widths), widths
        assert d["width_px_s"] < min(widths)

    def test_two_doorways(self, two_doors):
        t = sorted(d["time_s"] for d in two_doors["doors"])
        assert len(t) == 2, (t, two_doors["info"].get("warn"))
        assert t[0] == pytest.approx(3.2, abs=0.35)
        assert t[1] == pytest.approx(9.0, abs=0.45)


# ======================================================================================
class TestFalsePositives:
    """Every corridor thing that is not a walked opening."""

    @pytest.mark.parametrize("name", ["column", "side_window", "wall_poster",
                                      "facing_posters", "dead_end", "turn"])
    def test_no_detection(self, adversarial, name):
        r = adversarial[name]
        assert r["doors"] == [], (name, [(c["time_s"], c["confidence"])
                                         for c in r["doors"]])

    @pytest.mark.parametrize("name", ["column", "side_window", "wall_poster"])
    def test_one_sided_things_never_even_pair(self, adversarial, name):
        """Passed on ONE side -> both edges carry the same sign of K -> gate (a) means
        no candidate PAIR is formed at that time at all."""
        r = adversarial[name]
        near = [c for c in r["candidates"] if abs(c["time_s"] - 3.2) < 0.8]
        assert all(c["confidence"] < DEFAULTS["min_confidence"] for c in near)

    def test_facing_posters_die_on_the_missing_lintel(self, adversarial):
        """Two opposite panels DO straddle the walk and DO share a crossing time — only
        the absence of a member spanning between them separates them from a doorway."""
        r = adversarial["facing_posters"]
        near = [c for c in r["candidates"] if abs(c["time_s"] - 3.2) < 1.0]
        assert near, "the fixture must reach the pairing stage, else it proves nothing"
        for c in near:
            assert c["confidence"] < DEFAULTS["min_confidence"]
            assert (any(x.startswith("no_lintel") or x == "lintel_time_disagrees"
                        for x in c["reject"])
                    or c["scores"]["lintel"] < 0.35), c

    def test_dead_end_never_completes_the_divergence(self, adversarial):
        """The corridor corners at the end wall DO diverge out of the frame — an edge
        0.91 m off the path leaves a 65 deg frame 1.43 m before its plane — so the walk
        having ARRIVED is a separate question from the divergence having completed, and
        it is the expansion rate in the gap that answers it."""
        r = adversarial["dead_end"]
        assert any("walk_stopped_before_crossing" in c["reject"]
                   for c in r["candidates"]), r["candidates"]
        for c in r["candidates"]:
            if c["arrival_ratio"] is not None and "walk_stopped_before_crossing" in c["reject"]:
                assert c["arrival_ratio"] < DEFAULTS["arrival_frac"]

    def test_turn_is_visible_to_the_yaw_gate(self, adversarial):
        """The per-candidate turn number cannot be asserted on this fixture: a 70 deg
        swing breaks eq. (1) outright, so the tracks that span it fail their Mobius fit
        and no pair is ever formed over the swing (the candidates that survive lie BEFORE
        it, and honestly report ~6 deg). What must hold is that the gate's observable —
        the frame-to-frame profile shift — does fire over the window, so that a slower
        drift, which the fit would tolerate, is still caught."""
        r = adversarial["turn"]
        assert all(not c["accepted"] for c in r["candidates"])
        cams = _walk(110, yaw=lambda s: -70.0 * np.clip((s - 1.0) / 1.6, 0.0, 1.0))
        prof = frame_profiles(_render(_add_door(_corridor(), 3.2), cams), fps=FPS)
        w = DEFAULTS["work_width"]
        fx = 0.5 * w / np.tan(np.radians(0.5 * DEFAULTS["hfov_deg"]))
        f0, f1 = int(1.0 * FPS), int(2.6 * FPS)
        swing = np.degrees(np.arctan(np.abs(prof["shift_px"][f0:f1 + 1]).sum() / fx))
        assert swing > DEFAULTS["turn_hard_deg"], swing
        for c in r["candidates"]:
            if c["turn_deg"] > DEFAULTS["turn_hard_deg"]:
                assert "turning" in c["reject"]

    def test_a_zero_run_is_diagnosable(self, adversarial):
        for name, r in adversarial.items():
            assert "warn" in r["info"], name
            assert "n_tracks" in r["info"] and "track_reject_hist" in r["info"]


# ======================================================================================
class TestRobustness:
    def test_sensor_noise(self):
        r = _detect(_door_scene(noise=6.0, seed=3))
        assert len(r["doors"]) == 1, r["info"].get("warn")
        assert r["doors"][0]["time_s"] == pytest.approx(3.2, abs=0.35)

    def test_illumination_ramp(self):
        # auto-exposure sweeping 0.55x -> 1.35x across the walk; the profile is normalised
        # by each frame's own median, so the peak bar must not move with it.
        g = np.linspace(0.55, 1.35, 110)
        r = _detect(_door_scene(gain=g, noise=3.0, seed=5))
        assert len(r["doors"]) == 1, r["info"].get("warn")
        assert r["doors"][0]["time_s"] == pytest.approx(3.2, abs=0.35)

    def test_heavier_noise_degrades_gracefully(self):
        r = _detect(_door_scene(noise=18.0, seed=7))
        assert len(r["doors"]) <= 1
        for d in r["doors"]:
            assert d["time_s"] == pytest.approx(3.2, abs=0.5)

    def test_slower_walk_same_time(self):
        # halve the speed: the crossing happens later in seconds, and the detector must
        # follow the geometry, not a tuned duration.
        r = _detect(_render(_add_door(_corridor(), 3.2), _walk(210, v=0.5)))
        assert [d["time_s"] for d in r["doors"]] == pytest.approx([6.4], abs=0.6)

    def test_camera_pitched_further_up(self):
        # 22 deg up-tilt: fewer jamb rows survive in the band. Either it still detects, or
        # it returns nothing with a diagnosis — never a wrong time.
        r = _detect(_render(_add_door(_corridor(), 3.2), _walk(110, pitch=22.0)))
        for d in r["doors"]:
            assert d["time_s"] == pytest.approx(3.2, abs=0.5)

    def test_camera_yawed_off_the_walk_direction(self):
        # carried 8 deg off the direction of travel: the focus of expansion moves off the
        # image centre, which the Mobius fit estimates instead of assuming.
        r = _detect(_render(_add_door(_corridor(), 3.2), _walk(110, yaw=8.0)))
        assert len(r["doors"]) == 1, r["info"].get("warn")
        assert r["doors"][0]["time_s"] == pytest.approx(3.2, abs=0.4)
        assert abs(r["doors"][0]["foe_px"] - 0.5 * DEFAULTS["work_width"]) > 8.0


# ======================================================================================
class TestConfidenceSeparation:
    """`min_confidence` must sit in an empty gap, not on top of a distribution."""

    def test_gap(self, one_door, two_doors, adversarial):
        true_conf = [d["confidence"] for d in one_door["doors"] + two_doors["doors"]]
        false_conf = [c["confidence"] for name, r in adversarial.items()
                      for c in r["candidates"]]
        assert true_conf, "no true detection to compare against"
        assert min(true_conf) > DEFAULTS["min_confidence"]
        if false_conf:
            assert max(false_conf) < DEFAULTS["min_confidence"]
            assert min(true_conf) - max(false_conf) > 0.10, (min(true_conf),
                                                            max(false_conf))


# ======================================================================================
class TestContract:
    """The result must drop into the wiring `door_detect` already feeds."""

    def test_door_times_and_arclengths(self, one_door):
        ts = door_times(one_door)
        assert len(ts) == 1
        # the same call build_coplay's --door-times path makes
        pose_t = np.linspace(0.0, 109 / FPS, 40)
        traj = np.stack([np.linspace(0.0, 3.6, 40), np.zeros(40)], axis=1)
        s = coarse_match.door_arclengths(traj, pose_t, ts)
        assert len(s) == 1
        assert 0.0 <= s[0] <= 3.6

    def test_min_confidence_can_only_raise_the_bar(self, one_door):
        assert door_times(one_door, min_confidence=0.99) == []
        assert len(door_times(one_door, min_confidence=0.0)) == len(one_door["doors"])

    def test_arclength_field_needs_the_poses(self, one_door):
        with pytest.raises(ValueError):
            door_s_values(one_door)

    def test_with_poses_it_fills_s_and_width(self):
        n = 110
        pose_t = np.linspace(0.0, (n - 1) / FPS, 40)
        traj = np.stack([np.zeros(40), np.linspace(0.0, 3.633, 40)], axis=1)
        r = _detect(_door_scene(n=n), pose_times=pose_t, traj_xz=traj, width_hint=1.82)
        assert len(r["doors"]) == 1, r["info"].get("warn")
        d = r["doors"][0]
        assert d["s"] is not None and 0.0 < d["s"] < 3.633
        assert d["s"] == pytest.approx(3.2, abs=0.35)      # 1 m/s walk, 1 recon unit = 1 m
        assert d["width_recon"] == pytest.approx(0.90, rel=0.30)
        assert DEFAULTS["width_ratio_band"][0] <= d["width_ratio"] <= DEFAULTS["width_ratio_band"][1]
        assert door_s_values(r) == [d["s"]]

    def test_events_feed_coarse_match(self):
        """The end of the wire: door arclengths become `door` events in recon_events."""
        traj = np.stack([np.zeros(40), np.linspace(0.0, 3.633, 40)], axis=1)
        ev = coarse_match.recon_events(traj, door_s=[3.2])
        assert any(e["kind"] == "door" for e in ev)


# ======================================================================================
class TestDegenerate:
    def test_unknown_parameter(self):
        with pytest.raises(ValueError):
            detect_doors(np.zeros((10, 8, 8), np.uint8), fps=FPS, params={"nope": 1})

    def test_fps_required_for_raw_frames(self):
        with pytest.raises(ValueError):
            detect_doors(np.zeros((10, 8, 8), np.uint8))

    def test_too_few_frames(self):
        r = detect_doors(np.zeros((4, 32, 32), np.uint8), fps=FPS)
        assert r["doors"] == [] and "fail" in r["info"]

    def test_blank_video_is_silent_not_wrong(self):
        r = detect_doors(np.full((60, 90, 160), 128, np.uint8), fps=FPS)
        assert r["doors"] == []
        assert r["info"]["n_tracks"] == 0
        assert "warn" in r["info"]

    def test_pose_length_mismatch(self):
        with pytest.raises(ValueError):
            detect_doors(np.zeros((20, 32, 32), np.uint8), fps=FPS,
                         pose_times=np.arange(5.0), traj_xz=np.zeros((4, 2)))

"""TDD for the wall-anchor integration (wall-scale-x3):
tools/build_coplay.py's model_corridor_widths()/wall_scale_anchor() and
tools/validate_wall_anchor.py's compute_metric_scale() 3-anchor fusion, plus the
validate CLI's exit-code contract (0/1/2/3).

Synthetic geometry is built at a KNOWN ground-truth scale (GT_SCALE=2.0: the
recon corridor is exactly half the model corridor's linear dimensions) so all
three anchors (ceiling-height, camera-height, corridor-width) are expected to
independently agree near 2.0 — a clean, deterministic check that the fusion
path actually wires together, not just that each anchor works in isolation
(that's covered by tests/test_metric_scale.py and tests/test_wall_anchor.py).

Phase 1b (wall source AXX->SXX): real-data re-verification found AXX
(architecture) is furniture-dominated — zero triangles with vertical span >
1.5m — so its "corridor width" candidates were actually furniture gaps; SXX
(structure) carries the real walls/glass partitions. model_corridor_widths()
now takes the raw (3*T,3) triangle soup (is_triangle_soup=True) and filters by
per-triangle vertical span before detecting wall-pair gaps, clamped to a
plausible [1.5, 6.0]m corridor-width range. TestModelCorridorWidthsSxx exercises
this directly with synthetic triangle-soup wall panels (make_corridor()'s
scattered-point clouds aren't triangle-structured, so the model side of the
fusion test below uses _wall_mesh_triangles() instead; the recon side is
unaffected — estimate_wall_scale still consumes a plain point cloud).
"""
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tools.build_coplay as bc
import tools.validate_wall_anchor as vwa
from test_wall_anchor import make_corridor

_REPO = Path(__file__).resolve().parent.parent
_UPLOAD = _REPO / "realtime" / "_uploads" / "upload_1781521406685.lbp2"
_GASAN_GLOB = str(_REPO / "models" / "Gasan_7F" / "*.dtdx")
_FXX_ONLY = str(_REPO / "models" / "Gasan_7F" / "G7F_FAB_FXX_7F-0_Central_1.dtdx")

GT_SCALE = 2.0
MODEL_WIDTH, MODEL_HEIGHT, MODEL_LEN = 2.4, 3.0, 12.0


def _encode_pose(center, forward, up) -> np.ndarray:
    """Inverse of tools/build_coplay.py::viewer_pose() — builds a 12-float
    camera-to-world pose array from a desired (center, forward, up)."""
    center = np.asarray(center, dtype=np.float64)
    f = np.asarray(forward, dtype=np.float64); f = f / np.linalg.norm(f)
    u = np.asarray(up, dtype=np.float64); u = u / np.linalg.norm(u)
    row0 = np.cross(u, -f); row0 = row0 / (np.linalg.norm(row0) + 1e-12)
    R = np.array([row0, u, -f])
    t = -R @ center
    return np.array([
        R[0, 0], -R[0, 1], -R[0, 2], t[0],
        -R[1, 0], R[1, 1], R[1, 2], -t[1],
        -R[2, 0], R[2, 1], R[2, 2], -t[2],
    ], dtype=np.float64)


def _synthetic_recon(width: float, height: float, length: float, eye: float, n_poses: int = 40, seed: int = 2):
    """A recon corridor point cloud + camera trajectory walking its centreline
    at constant eye height, gravity already aligned to +Y (up=[0,1,0] for every
    pose keeps _rot_a_to_b(...) at identity, so compute_metric_scale's internal
    gravity-align is a no-op — isolates the anchor-fusion math being tested).
    This is the RECON side (estimate_wall_scale consumes a plain point cloud,
    no triangle structure needed) — unaffected by the SXX triangle-soup change."""
    cloud = make_corridor(width=width, length=length, height=height, seed=seed)
    scan_pts = cloud.copy()
    scan_pts[:, 1] *= -1.0
    scan_pts[:, 2] *= -1.0  # compute_metric_scale flips these back internally
    zs = np.linspace(1.0, length - 1.0, n_poses)
    poses = np.array([_encode_pose([0.0, eye, z], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]) for z in zs])
    return poses, scan_pts


def _wall_mesh_triangles(x0: float, z_range: tuple, height: float, n_tris: int = 2000,
                         seed: int = 0, x_jitter: float = 0.02, min_span: float = 1.8) -> np.ndarray:
    """A synthetic wall as a valid (3*n_tris, 3) triangle soup (matching
    decode_geometry()'s non-indexed output format) at X=x0, spanning z_range and
    [0, height] in Y. Each triangle's own vertical span is >= min_span (real
    SXX wall triangles are typically tall single panels), but triangle Y-centers
    are randomized across the full height so the flattened point cloud actually
    populates model_corridor_widths()'s mid-band filter (a naive floor-only /
    ceiling-only rectangular panel puts every vertex at the two Y extremes and
    the mid-band filter finds nothing)."""
    rng = np.random.RandomState(seed)
    z = rng.uniform(z_range[0], z_range[1], n_tris)
    span = rng.uniform(min_span, height, n_tris)
    y_center = rng.uniform(span / 2, height - span / 2, n_tris)
    tris = []
    for i in range(n_tris):
        y0, y1 = y_center[i] - span[i] / 2, y_center[i] + span[i] / 2
        zc = z[i]
        tris.append([[x0, y0, zc - 0.05], [x0, y1, zc], [x0, (y0 + y1) / 2, zc + 0.05]])
    tris = np.array(tris, dtype=np.float64)
    tris[:, :, 0] += rng.normal(0, x_jitter, tris.shape[:2])
    return tris.reshape(-1, 3)


def _furniture_triangles(x0: float, z_range: tuple, y0: float = 0.4, span: float = 0.3,
                         n_tris: int = 500, seed: int = 9) -> np.ndarray:
    """Short (< 1.5m) triangles at a fixed low height — the furniture-scale
    noise that model_corridor_widths()'s vertical-span filter must reject
    (real-data finding: AXX is entirely made of these, no real walls)."""
    rng = np.random.RandomState(seed)
    z = rng.uniform(z_range[0], z_range[1], n_tris)
    tris = [[[x0, y0, zc - 0.02], [x0, y0 + span, zc], [x0, y0 + span * 0.5, zc + 0.02]] for zc in z]
    return np.array(tris, dtype=np.float64).reshape(-1, 3)


class TestThreeAnchorFusionPath(unittest.TestCase):
    def test_synthetic_model_and_recon_agree_on_ground_truth_scale(self):
        left = _wall_mesh_triangles(-MODEL_WIDTH / 2, (0, MODEL_LEN), MODEL_HEIGHT, seed=1)
        right = _wall_mesh_triangles(MODEL_WIDTH / 2, (0, MODEL_LEN), MODEL_HEIGHT, seed=2)
        wall_points = np.concatenate([left, right], axis=0)
        recon_eye = 1.5 / GT_SCALE  # assumed camera-carry height / GT_SCALE
        poses, scan_pts = _synthetic_recon(
            MODEL_WIDTH / GT_SCALE, MODEL_HEIGHT / GT_SCALE, MODEL_LEN / GT_SCALE, recon_eye,
        )
        scale = vwa.compute_metric_scale(poses, scan_pts, MODEL_HEIGHT, wall_points=wall_points)

        self.assertIsNotNone(scale["s_wall"])
        for key in ("s_vert", "s_cam", "s_wall"):
            self.assertAlmostEqual(scale[key], GT_SCALE, delta=GT_SCALE * 0.1)
        self.assertTrue(scale["s_v_info"]["agree"])
        # axis-split: s_h comes from the wall anchor (not a 3-way fuse with s_v);
        # s_v is the ceiling+camera-only fusion. Both land near ground truth here
        # since all three raw anchors independently agree.
        self.assertEqual(scale["s_h"], scale["s_wall"])
        self.assertAlmostEqual(scale["s_h"], GT_SCALE, delta=GT_SCALE * 0.1)
        self.assertAlmostEqual(scale["s_v"], GT_SCALE, delta=GT_SCALE * 0.1)
        self.assertIsNone(scale["fallback_reason"])
        # path_m is already axis-split metric (no separate scalar multiply needed)
        self.assertGreater(scale["path_m"], 0.0)


class TestModelCorridorWidthsSxx(unittest.TestCase):
    """Directly exercises the Phase-1b change: AXX->SXX source, triangle-soup
    vertical-span filtering, and the [1.5, 6.0]m width clamp."""

    def test_furniture_scale_triangles_are_filtered_out(self):
        left = _wall_mesh_triangles(-1.2, (0, 12.0), 3.0, seed=1)
        right = _wall_mesh_triangles(1.2, (0, 12.0), 3.0, seed=2)
        furniture = _furniture_triangles(0.3, (0, 12.0), seed=9)  # short, would corrupt the gap if not dropped
        tris = np.concatenate([left, right, furniture], axis=0)
        widths, info = bc.model_corridor_widths(tris, is_triangle_soup=True)
        self.assertEqual(len(widths), 1)
        self.assertAlmostEqual(widths[0], 2.4, delta=0.15)
        self.assertIn("top_candidates", info)

    def test_width_range_clamp_excludes_out_of_range_gap(self):
        left = _wall_mesh_triangles(-1.2, (0, 12.0), 3.0, seed=1)
        right = _wall_mesh_triangles(1.2, (0, 12.0), 3.0, seed=2)
        far = _wall_mesh_triangles(6.7, (0, 12.0), 3.0, seed=3)   # gaps ~2.4 and ~5.5
        tris = np.concatenate([left, right, far], axis=0)
        default_widths, _ = bc.model_corridor_widths(tris, is_triangle_soup=True)
        self.assertEqual(len(default_widths), 2)  # both in-range under the [1.5,6.0] default
        narrow_widths, narrow_info = bc.model_corridor_widths(
            tris, is_triangle_soup=True, width_range=(1.5, 3.0))
        self.assertEqual(len(narrow_widths), 1)
        self.assertAlmostEqual(narrow_widths[0], 2.4, delta=0.2)
        self.assertLessEqual(len(narrow_info["top_candidates"]), 5)

    def test_non_triangle_length_fails_cleanly(self):
        pts = np.random.RandomState(0).uniform(-1, 1, (17, 3))  # not divisible by 3
        widths, info = bc.model_corridor_widths(pts, is_triangle_soup=True)
        self.assertEqual(widths, [])
        self.assertIn("fail", info)


class TestWallFallback(unittest.TestCase):
    def test_no_walls_in_model_falls_back_to_two_anchors(self):
        rng = np.random.RandomState(3)
        open_space = np.column_stack([
            rng.uniform(-2.0, 2.0, 30000),
            rng.uniform(0.0, MODEL_HEIGHT, 30000),
            rng.uniform(0.0, 10.0, 30000),
        ])
        recon_eye = 1.5 / GT_SCALE
        poses, scan_pts = _synthetic_recon(
            MODEL_HEIGHT / GT_SCALE, MODEL_HEIGHT / GT_SCALE, 6.0, recon_eye, n_poses=20,
        )
        scale = vwa.compute_metric_scale(poses, scan_pts, MODEL_HEIGHT, wall_points=open_space)

        self.assertIsNone(scale["s_wall"])
        self.assertEqual(scale["s_v_info"]["n"], 2)
        self.assertIn("fail", scale["scale_info"]["wall_anchor"])
        self.assertIsNotNone(scale["fallback_reason"])
        # s_h falls back to s_v (fallback, not broken) — both land near ground truth
        self.assertEqual(scale["s_h"], scale["s_v"])
        self.assertAlmostEqual(scale["s_h"], GT_SCALE, delta=GT_SCALE * 0.15)

    def test_no_wall_points_provided_matches_prior_two_anchor_behavior(self):
        """wall_points=None (Phase-1 callers) must be bit-for-bit unaffected —
        the wall anchor is never computed, not merely absent from the result.
        s_h==s_v (isotropic axis-split) so path_m is numerically identical to
        the pre-axis-split 2-anchor value."""
        recon_eye = 1.5 / GT_SCALE
        poses, scan_pts = _synthetic_recon(
            MODEL_HEIGHT / GT_SCALE, MODEL_HEIGHT / GT_SCALE, 6.0, recon_eye, n_poses=20,
        )
        with_none = vwa.compute_metric_scale(poses, scan_pts, MODEL_HEIGHT, wall_points=None)
        without_kw = vwa.compute_metric_scale(poses, scan_pts, MODEL_HEIGHT)
        self.assertIsNone(with_none["s_wall"])
        self.assertEqual(with_none["s_h"], with_none["s_v"])
        self.assertEqual(with_none["path_m"], without_kw["path_m"])
        self.assertEqual(with_none["s_v_info"]["n"], 2)


def _write_coplay_fixture(tmp_dir: Path, rawp: list, duration: float) -> Path:
    """Minimal stand-in for a tools/build_coplay.py TEMPLATE output — just needs
    the `RAWP=[...], META={...};` substring check_coplay_geometry.py parses."""
    html = ("<html><body><script>const MESHES=[], MODELS=[], RAWP="
            + json.dumps(rawp) + ", META=" + json.dumps({"duration": duration}) + ";</script></body></html>")
    p = tmp_dir / "fixture.coplay.html"
    p.write_text(html, encoding="utf-8")
    return p


def _l_shaped_poses(corner_z: float = 3.0, end_xz: tuple = (2.0, 1.0), n_per_leg: int = 20) -> list:
    """A clean two-straight-segment L path: (0,10) -> (0,corner_z) -> end_xz."""
    leg1 = np.column_stack([np.zeros(n_per_leg), np.linspace(10.0, corner_z, n_per_leg)])
    leg2 = np.column_stack([np.linspace(0.0, end_xz[0], n_per_leg), np.linspace(corner_z, end_xz[1], n_per_leg)])
    xz = np.vstack([leg1, leg2])
    return [{"c": [float(x), 1.5, float(z)], "f": [0.0, 0.0, 1.0], "u": [0.0, 1.0, 0.0]} for x, z in xz]


class TestCheckCoplayGeometry(unittest.TestCase):
    """tools/check_coplay_geometry.py — exercised via subprocess (real CLI),
    both against a synthetic well-behaved L path (proves the gate isn't
    trivially always-fail) and the real red-baseline served HTML."""

    def _run(self, html_path, *extra_args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "tools/check_coplay_geometry.py", str(html_path), *extra_args],
            cwd=_REPO, capture_output=True, text=True, timeout=60,
        )

    def test_well_placed_l_path_passes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            html = _write_coplay_fixture(Path(td), _l_shaped_poses(corner_z=3.0, end_xz=(2.0, 1.0)), duration=20.0)
            proc = self._run(html, "--turn-z-max", "4", "--end-x", "-8,4", "--end-z-max", "2.5")
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("verdict=PASS", proc.stdout)

    def test_turn_point_too_deep_fails(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            # same shape, corner pushed to Z=9 (mirrors the real red baseline's ~7-9 range)
            html = _write_coplay_fixture(Path(td), _l_shaped_poses(corner_z=9.0, end_xz=(2.0, 7.5)), duration=20.0)
            proc = self._run(html, "--turn-z-max", "4", "--end-x", "-8,4", "--end-z-max", "2.5")
            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
            self.assertIn("verdict=FAIL", proc.stdout)

    def test_missing_html_exits_2(self):
        proc = self._run(_REPO / "reports" / "coplay" / "does_not_exist.html")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)

    @unittest.skipUnless(
        (_REPO / "realtime" / "_uploads" / "upload_1781521406685.coplay.html").exists(),
        "served coplay html fixture not present in this worktree",
    )
    def test_current_served_html_is_red_baseline_exit_1(self):
        """PM's re-verification finding: the currently-served html's turn point
        sits deep in an unrelated zone (Z~7-9, not the real ~3 lounge entrance).
        This documents that as a reproducible red baseline, not a one-off claim."""
        served = _REPO / "realtime" / "_uploads" / "upload_1781521406685.coplay.html"
        proc = self._run(served, "--turn-z-max", "4", "--end-x", "-8,4", "--end-z-max", "2.5")
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("verdict=FAIL", proc.stdout)


@unittest.skipUnless(_UPLOAD.exists(), "upload_1781521406685 fixture not present in this worktree")
class TestValidateCliExitContract(unittest.TestCase):
    """Exercises the actual CLI (subprocess) against the real fixture, not
    compute_metric_scale directly, so this is an end-to-end contract check."""

    def _run(self, *extra_args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "tools/validate_wall_anchor.py", str(_UPLOAD), *extra_args],
            cwd=_REPO, capture_output=True, text=True, timeout=120,
        )

    def test_no_wall_anchor_always_exits_0(self):
        proc = self._run("--no-wall-anchor")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("verdict=", proc.stdout)

    def test_default_mode_without_dtdx_exits_2(self):
        proc = self._run()
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)

    @unittest.skipUnless(Path(_FXX_ONLY).exists(), "Gasan_7F FXX fixture not present")
    def test_default_mode_without_sxx_exits_3(self):
        proc = self._run("--dtdx", _FXX_ONLY)
        self.assertEqual(proc.returncode, 3, proc.stdout + proc.stderr)
        self.assertIn("wall anchor not available", proc.stdout)

    @unittest.skipUnless(bool(__import__("glob").glob(_GASAN_GLOB)), "Gasan_7F fixture not present")
    def test_default_mode_with_full_model_never_crashes(self):
        proc = self._run("--dtdx", _GASAN_GLOB)
        self.assertIn(proc.returncode, (0, 1, 2), proc.stdout + proc.stderr)
        self.assertIn("verdict=", proc.stdout)

    @unittest.skipUnless(bool(__import__("glob").glob(_GASAN_GLOB)), "Gasan_7F fixture not present")
    def test_default_mode_real_data_hits_risk1_anchor_band_tag(self):
        """Documents the PM-anticipated 'risk 1' as a reproducible measured
        result (not manipulated): for upload_1781521406685, the recon-side
        corridor gap never straddles the camera trajectory at all (fails before
        _choose_scale's candidate-width mapping even runs), so s_h falls back to
        s_v (~1.75) — outside S_H_BAND[3.2,4.1] — while s_v itself is in-band.
        exit 2. If anchor-core's straddle detection changes, this test should be
        revisited rather than silently loosened."""
        proc = self._run("--dtdx", _GASAN_GLOB)
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("s_v ok=True", proc.stdout)
        self.assertIn("s_h ok=False", proc.stdout)
        self.assertIn("wall_anchor:", proc.stdout)


if __name__ == "__main__":
    unittest.main()

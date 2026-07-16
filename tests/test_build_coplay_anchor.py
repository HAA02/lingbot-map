"""TDD for the wall-anchor integration (Phase 2, wall-scale-x3):
tools/build_coplay.py's model_corridor_widths()/wall_scale_anchor() and
tools/validate_wall_anchor.py's compute_metric_scale() 3-anchor fusion, plus the
validate CLI's exit-code contract (0/1/2/3).

Synthetic geometry is built at a KNOWN ground-truth scale (GT_SCALE=2.0: the
recon corridor is exactly half the model corridor's linear dimensions) so all
three anchors (ceiling-height, camera-height, corridor-width) are expected to
independently agree near 2.0 — a clean, deterministic check that the fusion
path actually wires together, not just that each anchor works in isolation
(that's covered by tests/test_metric_scale.py and tests/test_wall_anchor.py).
"""
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
    gravity-align is a no-op — isolates the anchor-fusion math being tested)."""
    cloud = make_corridor(width=width, length=length, height=height, seed=seed)
    scan_pts = cloud.copy()
    scan_pts[:, 1] *= -1.0
    scan_pts[:, 2] *= -1.0  # compute_metric_scale flips these back internally
    zs = np.linspace(1.0, length - 1.0, n_poses)
    poses = np.array([_encode_pose([0.0, eye, z], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]) for z in zs])
    return poses, scan_pts


class TestThreeAnchorFusionPath(unittest.TestCase):
    def test_synthetic_model_and_recon_agree_on_ground_truth_scale(self):
        axx_points = make_corridor(width=MODEL_WIDTH, length=MODEL_LEN, height=MODEL_HEIGHT, seed=1)
        recon_eye = 1.5 / GT_SCALE  # assumed camera-carry height / GT_SCALE
        poses, scan_pts = _synthetic_recon(
            MODEL_WIDTH / GT_SCALE, MODEL_HEIGHT / GT_SCALE, MODEL_LEN / GT_SCALE, recon_eye,
        )
        scale = vwa.compute_metric_scale(poses, scan_pts, MODEL_HEIGHT, axx_points=axx_points)

        self.assertIsNotNone(scale["s_wall"])
        self.assertEqual(scale["scale_info"]["n"], 3)
        for key in ("s_vert", "s_cam", "s_wall"):
            self.assertAlmostEqual(scale[key], GT_SCALE, delta=GT_SCALE * 0.1)
        self.assertTrue(scale["scale_info"]["agree"])
        self.assertAlmostEqual(scale["s_m"], GT_SCALE, delta=GT_SCALE * 0.1)


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
        scale = vwa.compute_metric_scale(poses, scan_pts, MODEL_HEIGHT, axx_points=open_space)

        self.assertIsNone(scale["s_wall"])
        self.assertEqual(scale["scale_info"]["n"], 2)  # None filtered out by fuse_scale_estimates
        self.assertIn("fail", scale["scale_info"]["wall_anchor"])
        # the 2 surviving anchors still land near ground truth (fallback, not broken)
        self.assertAlmostEqual(scale["s_m"], GT_SCALE, delta=GT_SCALE * 0.15)

    def test_no_axx_points_provided_matches_prior_two_anchor_behavior(self):
        """axx_points=None (Phase-1 callers) must be bit-for-bit unaffected —
        the wall anchor is never computed, not merely absent from the result."""
        recon_eye = 1.5 / GT_SCALE
        poses, scan_pts = _synthetic_recon(
            MODEL_HEIGHT / GT_SCALE, MODEL_HEIGHT / GT_SCALE, 6.0, recon_eye, n_poses=20,
        )
        with_none = vwa.compute_metric_scale(poses, scan_pts, MODEL_HEIGHT, axx_points=None)
        without_kw = vwa.compute_metric_scale(poses, scan_pts, MODEL_HEIGHT)
        self.assertIsNone(with_none["s_wall"])
        self.assertEqual(with_none["s_m"], without_kw["s_m"])
        self.assertEqual(with_none["scale_info"]["n"], 2)


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
    def test_default_mode_without_axx_exits_3(self):
        proc = self._run("--dtdx", _FXX_ONLY)
        self.assertEqual(proc.returncode, 3, proc.stdout + proc.stderr)
        self.assertIn("wall anchor not available", proc.stdout)

    @unittest.skipUnless(bool(__import__("glob").glob(_GASAN_GLOB)), "Gasan_7F fixture not present")
    def test_default_mode_with_full_model_exits_0_or_1_never_crashes(self):
        proc = self._run("--dtdx", _GASAN_GLOB)
        self.assertIn(proc.returncode, (0, 1), proc.stdout + proc.stderr)
        self.assertIn("verdict=", proc.stdout)
        self.assertIn("wall_anchor:", proc.stdout)


if __name__ == "__main__":
    unittest.main()

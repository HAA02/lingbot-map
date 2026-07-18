"""Tests for scan2bim.forward_scale (anisotropic heading-relative forward scale)."""
import numpy as np
import pytest

from scan2bim.forward_scale import (
    anisotropic_scale_tensor,
    apply_forward_scale,
    corridor_open_boundary,
    estimate_forward_scale,
)
from scan2bim.metric_scale import apply_axis_split_scale


# --- anisotropic_scale_tensor ---------------------------------------------------

def test_tensor_is_symmetric_2x2():
    A = anisotropic_scale_tensor(0.7, 3.0, 2.0)
    assert A.shape == (2, 2)
    assert np.allclose(A, A.T)


def test_tensor_isotropic_when_equal():
    # s_f == s_lateral -> exactly s*I, independent of theta (degrades to isotropic)
    for theta in (0.0, 0.3, -1.2, 2.5):
        A = anisotropic_scale_tensor(theta, 2.5, 2.5)
        assert np.allclose(A, 2.5 * np.eye(2))


def test_tensor_eigenvalues_are_the_two_scales():
    A = anisotropic_scale_tensor(0.9, 3.3, 1.8)
    eig = np.sort(np.linalg.eigvalsh(A))
    assert np.allclose(eig, [1.8, 3.3])


def test_tensor_stretches_along_theta_by_s_forward():
    theta, s_f, s_h = 0.6, 3.0, 2.0
    A = anisotropic_scale_tensor(theta, s_f, s_h)
    along = np.array([np.cos(theta), np.sin(theta)])        # unit heading
    across = np.array([-np.sin(theta), np.cos(theta)])      # lateral
    assert np.allclose(A @ along, s_f * along)              # forward -> s_f
    assert np.allclose(A @ across, s_h * across)            # lateral -> s_h


# --- apply_forward_scale --------------------------------------------------------

def _sample_poses(n=12, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(n, 3))
    f = rng.normal(size=(n, 3)); f /= np.linalg.norm(f, axis=1, keepdims=True)
    u = rng.normal(size=(n, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    pts = rng.normal(size=(40, 3))
    return {"c": c, "f": f, "u": u}, pts


def test_apply_forward_scale_isotropic_matches_axis_split():
    # A = s_h*I MUST reproduce metric_scale.apply_axis_split_scale exactly (the
    # isotropic path that place_rigid keeps byte-identical when the flag is off).
    poses, pts = _sample_poses()
    s_h, s_v = 1.9656, 1.7522
    A = anisotropic_scale_tensor(0.4, s_h, s_h)             # theta irrelevant when equal
    got_p, got_pts = apply_forward_scale(poses, pts, A, s_v)
    exp_p, exp_pts = apply_axis_split_scale(poses, pts, s_h=s_h, s_v=s_v)
    assert np.allclose(got_p["c"], exp_p["c"])
    assert np.allclose(got_p["f"], exp_p["f"])
    assert np.allclose(got_p["u"], exp_p["u"])
    assert np.allclose(got_pts, exp_pts)


def test_apply_forward_scale_center_scales_by_tensor():
    poses = {"c": np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]]),
             "f": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
             "u": np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])}
    pts = np.array([[2.0, 0.0, 1.0]])
    A = anisotropic_scale_tensor(0.0, 3.0, 2.0)             # theta=0 -> X*3, Z*2
    got, got_pts = apply_forward_scale(poses, pts, A, s_v=1.5)
    # centres: x*3, y*1.5, z*2
    assert np.allclose(got["c"], [[3.0, 3.0, 0.0], [0.0, 1.5, 6.0]])
    assert np.allclose(got_pts, [[6.0, 0.0, 2.0]])


def test_apply_forward_scale_keeps_forward_unit_and_perp_to_up():
    poses, pts = _sample_poses(seed=3)
    A = anisotropic_scale_tensor(1.1, 3.4, 1.9)
    got, _ = apply_forward_scale(poses, pts, A, s_v=1.6)
    assert np.allclose(np.linalg.norm(got["u"], axis=1), 1.0)
    assert np.allclose(np.linalg.norm(got["f"], axis=1), 1.0)
    assert np.allclose(np.sum(got["f"] * got["u"], axis=1), 0.0, atol=1e-9)


# --- corridor_open_boundary -----------------------------------------------------

def _corridor_segments(z_close=0.0, z_far=20.0, x_lo=-1.0, x_hi=1.0, seg=0.5):
    """Two parallel walls (along Z) at x=x_lo and x=x_hi spanning z in [z_close, z_far]."""
    zs = np.arange(z_far, z_close - 1e-9, -seg)
    segs = []
    for i in range(len(zs) - 1):
        segs.append([[x_lo, zs[i]], [x_lo, zs[i + 1]]])
        segs.append([[x_hi, zs[i]], [x_hi, zs[i + 1]]])
    return np.asarray(segs, dtype=np.float64)


def test_corridor_open_boundary_finds_wall_end():
    # corridor along +Z (origin at z=0), walls span z in [0, 20]; beyond z=20 it opens.
    segs = _corridor_segments(z_close=0.0, z_far=20.0)
    L_end, info = corridor_open_boundary(segs, axis_origin=(0.0, 0.0), axis_dir=(0.0, 1.0),
                                         s_range=(-2.0, 40.0), step=0.5, half_width=3.0)
    assert L_end is not None
    # boundary is the last s with a facing pair -> z ~ 20 (within one step)
    assert abs(L_end[1] - 20.0) <= 0.75
    assert abs(L_end[0] - 0.0) <= 1e-6


def test_corridor_open_boundary_fails_without_pair():
    # single wall only -> no facing pair
    segs = np.array([[[-1.0, 0.0], [-1.0, 10.0]]], dtype=np.float64)
    L_end, info = corridor_open_boundary(segs, (0.0, 0.0), (0.0, 1.0))
    assert L_end is None
    assert "fail" in info


# --- estimate_forward_scale -----------------------------------------------------

def test_estimate_forward_scale_ratio():
    # start (0,10) -> L_end (0,-2) along run_dir +Z: forward dist 12, arclen 3 -> s_f 4
    s_f, info = estimate_forward_scale(3.0, start_xz=(0.0, 10.0), L_end_xz=(0.0, -2.0),
                                       run_dir=(0.0, 1.0))
    assert s_f == pytest.approx(4.0)
    assert info["forward_dist"] == pytest.approx(12.0)


def test_estimate_forward_scale_projects_out_lateral():
    # a lateral offset in start must not inflate the forward distance (projection).
    s_f, info = estimate_forward_scale(4.0, start_xz=(5.0, 8.0), L_end_xz=(0.0, 0.0),
                                       run_dir=(0.0, 1.0))
    assert info["forward_dist"] == pytest.approx(8.0)   # only the Z component counts
    assert s_f == pytest.approx(2.0)


def test_estimate_forward_scale_band_annotation():
    s_f, info = estimate_forward_scale(3.0, (0.0, 10.0), (0.0, -2.0), (0.0, 1.0),
                                       s_h=2.0, band=(1.0, 3.0))
    assert info["band"] == [2.0, 6.0]
    assert info["in_band"] is True        # 2.0 < 4.0 <= 6.0


def test_estimate_forward_scale_rejects_zero_arclen():
    with pytest.raises(ValueError):
        estimate_forward_scale(0.0, (0.0, 0.0), (0.0, 1.0), (0.0, 1.0))

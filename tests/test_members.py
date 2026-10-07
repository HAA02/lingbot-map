"""Pipes and columns measured from XYZ. Gaussian PLY contributes means only."""
import struct
import unittest
from pathlib import Path

import numpy as np

from scan2bim.members import extract_members, extract_members_from_ply
from scan2bim.ply_xyz import read_ply_xyz


def _basis(axis):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    b1 = np.cross(axis, tmp)
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(axis, b1)
    return axis, b1, b2


def _cylinder(center, axis, radius, length, n, rng, noise=0.002):
    axis, b1, b2 = _basis(axis)
    t = rng.uniform(-length / 2.0, length / 2.0, n)
    th = rng.uniform(0.0, 2.0 * np.pi, n)
    pts = (np.asarray(center, dtype=np.float64)
           + t[:, None] * axis
           + (radius * np.cos(th))[:, None] * b1
           + (radius * np.sin(th))[:, None] * b2)
    pts += rng.normal(0.0, noise, pts.shape)
    return pts


def _scene():
    rng = np.random.default_rng(1)
    column = _cylinder([0.0, 0.0, 1.5], [0, 0, 1], 0.20, 3.0, 500, rng)
    pipe_a = _cylinder([2.0, 0.0, 2.7], [1, 0, 0], 0.04, 4.0, 400, rng)
    pipe_b = _cylinder([4.0, 1.0, 2.7], [0, 1, 0], 0.04, 2.0, 250, rng)
    clutter = rng.uniform([-1, -1, 0], [5, 3, 3], size=(80, 3))
    return np.vstack([column, pipe_a, pipe_b, clutter])


class TestExtractMembers(unittest.TestCase):
    def test_column_and_two_pipes_with_one_joint(self):
        out = extract_members(_scene(), seed=0)
        columns = [m for m in out["members"] if m["kind"] == "column"]
        pipes = [m for m in out["members"] if m["kind"] == "pipe"]
        self.assertEqual(len(columns), 1, out["members"])
        self.assertEqual(len(pipes), 2, out["members"])
        self.assertAlmostEqual(columns[0]["diameter_m"], 0.40, delta=0.03)
        self.assertAlmostEqual(columns[0]["length_m"], 3.0, delta=0.15)
        diams = sorted(p["diameter_m"] for p in pipes)
        self.assertAlmostEqual(diams[0], 0.08, delta=0.02)
        self.assertAlmostEqual(diams[1], 0.08, delta=0.02)
        lengths = sorted(p["length_m"] for p in pipes)
        self.assertAlmostEqual(lengths[0], 2.0, delta=0.2)
        self.assertAlmostEqual(lengths[1], 4.0, delta=0.25)
        pipe_ids = {p["id"] for p in pipes}
        joints = [c for c in out["connections"] if c["a"] in pipe_ids and c["b"] in pipe_ids]
        self.assertEqual(len(joints), 1, out["connections"])
        self.assertLess(joints[0]["gap_m"], 0.15)

    def test_wall_strip_is_not_a_pipe(self):
        rng = np.random.default_rng(2)
        xs = rng.uniform(0, 4, 600)
        zs = rng.uniform(0, 3, 600)
        wall = np.column_stack([xs, np.zeros(600), zs])
        wall += rng.normal(0, 0.002, wall.shape)
        out = extract_members(wall, seed=0)
        self.assertEqual(out["members"], [])

    def test_too_few_points(self):
        out = extract_members(np.zeros((5, 3)), seed=0)
        self.assertEqual(out["members"], [])
        self.assertEqual(out["connections"], [])


class TestPlyXyz(unittest.TestCase):
    def test_ascii_ply_keeps_xyz_and_ignores_gaussian_fields(self):
        path = Path("tests") / "_tmp_means.ply"
        try:
            path.write_text(
                "ply\nformat ascii 1.0\n"
                "element vertex 2\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property float opacity\nproperty float scale_0\n"
                "end_header\n"
                "1 2 3 0.9 0.01\n"
                "4 5 6 0.2 0.02\n",
                encoding="utf-8",
            )
            xyz = read_ply_xyz(str(path))
            np.testing.assert_allclose(xyz, [[1, 2, 3], [4, 5, 6]])
            out = extract_members_from_ply(str(path), seed=0)
            self.assertEqual(out["members"], [])
        finally:
            path.unlink(missing_ok=True)

    def test_binary_little_endian_xyz(self):
        path = Path("tests") / "_tmp_means_bin.ply"
        try:
            header = (
                "ply\nformat binary_little_endian 1.0\n"
                "element vertex 1\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\n"
                "end_header\n"
            ).encode("ascii")
            body = struct.pack("<fffB", 1.5, -2.0, 3.25, 255)
            path.write_bytes(header + body)
            xyz = read_ply_xyz(str(path))
            np.testing.assert_allclose(xyz, [[1.5, -2.0, 3.25]])
        finally:
            path.unlink(missing_ok=True)

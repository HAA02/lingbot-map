"""Tests for scan2bim.colmap_poses (COLMAP images.txt → product {c,f,u})."""
import math
import unittest

import numpy as np

from scan2bim.colmap_poses import poses_from_images_txt, quat_wxyz_to_rotmat


def _vec(pose_key):
    return np.asarray(pose_key, dtype=np.float64)


class TestColmapPoses(unittest.TestCase):
    def test_identity_quaternion_zero_t(self):
        text = (
            "# Identity camera at origin\n"
            "1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 identity.jpg\n"
            "\n"
        )
        poses = poses_from_images_txt(text)
        self.assertEqual(len(poses), 1)
        p = poses[0]
        self.assertEqual(p["image_id"], 1)
        self.assertEqual(p["name"], "identity.jpg")
        np.testing.assert_allclose(_vec(p["c"]), [0.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(_vec(p["f"]), [0.0, 0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(_vec(p["u"]), [0.0, -1.0, 0.0], atol=1e-6)
        for key in ("c", "f", "u"):
            self.assertTrue(all(type(x) is float for x in p[key]))

    def test_known_90deg_yaw(self):
        # Hardcoded known pair: 90° about +Y (right-hand). Not a general converter.
        # R = [[0, 0, 1],
        #      [0, 1, 0],
        #      [-1, 0, 0]]
        # q = (sqrt(2)/2, 0, sqrt(2)/2, 0)
        half = math.sqrt(2.0) / 2.0
        qw, qx, qy, qz = half, 0.0, half, 0.0
        R_expected = np.array(
            [[0.0, 0.0, 1.0],
             [0.0, 1.0, 0.0],
             [-1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        np.testing.assert_allclose(quat_wxyz_to_rotmat(qw, qx, qy, qz), R_expected, atol=1e-6)

        t = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        text = (
            f"7 {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 yaw90.jpg\n"
            "12.0 34.0 -1\n"
        )
        poses = poses_from_images_txt(text)
        self.assertEqual(len(poses), 1)
        R = R_expected
        c_exp = -R.T @ t
        f_exp = R.T @ np.array([0.0, 0.0, 1.0])
        u_exp = R.T @ np.array([0.0, -1.0, 0.0])
        f_exp = f_exp / np.linalg.norm(f_exp)
        u_exp = u_exp / np.linalg.norm(u_exp)
        p = poses[0]
        np.testing.assert_allclose(_vec(p["c"]), c_exp, atol=1e-6)
        np.testing.assert_allclose(_vec(p["f"]), f_exp, atol=1e-6)
        np.testing.assert_allclose(_vec(p["u"]), u_exp, atol=1e-6)
        self.assertEqual(p["name"], "yaw90.jpg")
        self.assertEqual(p["image_id"], 7)

    def test_image_id_order_even_if_file_is_unsorted(self):
        text = (
            "2 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 second.jpg\n"
            "\n"
            "1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 first.jpg\n"
            "\n"
        )
        poses = poses_from_images_txt(text)
        self.assertEqual([p["image_id"] for p in poses], [1, 2])
        self.assertEqual([p["name"] for p in poses], ["first.jpg", "second.jpg"])

    def test_comments_and_blank_points2d_ignored(self):
        text = (
            "# Image list with two lines of data per image:\n"
            "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
            "\n"
            "4 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 kept.jpg\n"
            "\n"
            "# trailing comment\n"
        )
        poses = poses_from_images_txt(text)
        self.assertEqual(len(poses), 1)
        self.assertEqual(poses[0]["name"], "kept.jpg")
        self.assertEqual(poses[0]["image_id"], 4)
        np.testing.assert_allclose(_vec(poses[0]["c"]), [0.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(_vec(poses[0]["f"]), [0.0, 0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(_vec(poses[0]["u"]), [0.0, -1.0, 0.0], atol=1e-6)

    def test_only_comments_returns_empty(self):
        text = (
            "# COLMAP images.txt\n"
            "# no images in this model\n"
            "\n"
            "# still nothing\n"
        )
        self.assertEqual(poses_from_images_txt(text), [])

    def test_malformed_extra_tokens_raise(self):
        text = "1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 name.jpg EXTRA\n\n"
        with self.assertRaises(ValueError):
            poses_from_images_txt(text)


if __name__ == "__main__":
    unittest.main()

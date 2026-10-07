"""COLMAP images.txt → product poses {c, f, u} in the reconstruction frame.

COLMAP stores a Hamilton quaternion (qw,qx,qy,qz) and translation t that map
world → camera: X_cam = R @ X_world + t. Camera axes are OpenCV: +X right,
+Y down, +Z forward. Product forward is camera +Z in world; product up is the
opposite of camera +Y (so +Y-down becomes up). No BIM / glTF axis conversion.
"""
from __future__ import annotations

import numpy as np


def quat_wxyz_to_rotmat(qw, qx, qy, qz) -> np.ndarray:
    """Hamilton (qw,qx,qy,qz) → 3x3 world-to-camera rotation (COLMAP qvec2rotmat)."""
    q = np.array([float(qw), float(qx), float(qy), float(qz)], dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0.0:
        raise ValueError("zero quaternion")
    w, x, y, z = q / n
    return np.array(
        [
            [1.0 - 2.0 * y * y - 2.0 * z * z, 2.0 * x * y - 2.0 * z * w, 2.0 * x * z + 2.0 * y * w],
            [2.0 * x * y + 2.0 * z * w, 1.0 - 2.0 * x * x - 2.0 * z * z, 2.0 * y * z - 2.0 * x * w],
            [2.0 * x * z - 2.0 * y * w, 2.0 * y * z + 2.0 * x * w, 1.0 - 2.0 * x * x - 2.0 * y * y],
        ],
        dtype=np.float64,
    )


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0.0:
        raise ValueError("zero vector")
    return v / n


def _pose_from_rt(R: np.ndarray, t: np.ndarray, *, name: str, image_id: int) -> dict:
    C = -R.T @ t
    f = _unit(R.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64))
    u = _unit(R.T @ np.array([0.0, -1.0, 0.0], dtype=np.float64))
    return {
        "c": [float(x) for x in C],
        "f": [float(x) for x in f],
        "u": [float(x) for x in u],
        "name": name,
        "image_id": int(image_id),
    }


def _parse_image_line(line: str) -> dict:
    parts = line.split()
    if len(parts) != 10:
        raise ValueError(f"expected 10 tokens (IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME), got {len(parts)}")
    image_id = int(parts[0])
    qw, qx, qy, qz = (float(parts[i]) for i in range(1, 5))
    tx, ty, tz = (float(parts[i]) for i in range(5, 8))
    # CAMERA_ID is required by the line format; unused for product pose.
    int(parts[8])
    name = parts[9]
    R = quat_wxyz_to_rotmat(qw, qx, qy, qz)
    t = np.array([tx, ty, tz], dtype=np.float64)
    return _pose_from_rt(R, t, name=name, image_id=image_id)


def poses_from_images_txt(text: str) -> list[dict]:
    """Parse COLMAP images.txt content. Skip comments and empty lines.

    Return poses in IMAGE_ID order. Each pose is
    {"c":[3 floats], "f":[3], "u":[3], "name": str, "image_id": int}.
    Floats are Python floats, not rounded.
    """
    lines = text.splitlines()
    poses: list[dict] = []
    i = 0
    n = len(lines)
    while i < n:
        stripped = lines[i].strip()
        i += 1
        if not stripped or stripped.startswith("#"):
            continue
        poses.append(_parse_image_line(stripped))
        # Second line of the image pair is POINTS2D (may be empty).
        if i < n:
            i += 1
    poses.sort(key=lambda p: p["image_id"])
    return poses

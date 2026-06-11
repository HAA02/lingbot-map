"""Project BIM objects into camera frames.

Given a camera (intrinsic K, c2w extrinsic) **in BIM coordinates**, compute
the 2D projection of each BIM object's sample points and decide whether
that object is a visible candidate in this frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .metadata import BimObject


@dataclass
class ProjectionResult:
    guid: str
    frame_number: int
    visible: bool
    projected_bbox: tuple[float, float, float, float] | None  # x1,y1,x2,y2
    center_uv: tuple[float, float] | None
    depth: float | None        # along camera Z, in BIM units
    view_angle_deg: float | None  # angle between view ray and object axis (if available)
    score: float               # rough visibility score in [0,1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "guid": self.guid,
            "frame_number": self.frame_number,
            "visible": bool(self.visible),
            "projected_bbox": list(self.projected_bbox) if self.projected_bbox else None,
            "center_uv": list(self.center_uv) if self.center_uv else None,
            "depth": float(self.depth) if self.depth is not None else None,
            "view_angle_deg": float(self.view_angle_deg) if self.view_angle_deg is not None else None,
            "score": float(self.score),
        }


def _c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    Rt = R.T
    w2c = np.zeros((3, 4), dtype=np.float64)
    w2c[:3, :3] = Rt
    w2c[:3, 3] = -Rt @ t
    return w2c


def project_points(
    points_world: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic_c2w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points to image plane.

    Returns:
        uv: (N,2) pixel coordinates (may be outside image)
        depth: (N,) camera-space Z (positive = in front of camera)
    """
    w2c = _c2w_to_w2c(extrinsic_c2w)
    pts_h = np.concatenate([points_world, np.ones((points_world.shape[0], 1))], axis=1)
    cam = (w2c @ pts_h.T).T  # (N,3) in camera frame
    z = cam[:, 2]
    safe_z = np.where(np.abs(z) < 1e-8, 1e-8, z)
    uvw = (intrinsic @ cam.T).T
    uv = uvw[:, :2] / safe_z[:, None]
    return uv, z


def project_object_to_frame(
    obj: BimObject,
    intrinsic: np.ndarray,
    extrinsic_c2w_bim: np.ndarray,
    image_size: tuple[int, int],  # (H, W)
    frame_number: int,
    *,
    margin_px: float = 0.0,
) -> ProjectionResult:
    H, W = image_size
    sample_pts = obj.sample_points()
    uv, depth = project_points(sample_pts, intrinsic, extrinsic_c2w_bim)

    in_front = depth > 0
    if not np.any(in_front):
        return ProjectionResult(obj.guid, frame_number, False, None, None, None, None, 0.0)

    uv_f = uv[in_front]
    d_f = depth[in_front]

    x_in = (uv_f[:, 0] >= -margin_px) & (uv_f[:, 0] < W + margin_px)
    y_in = (uv_f[:, 1] >= -margin_px) & (uv_f[:, 1] < H + margin_px)
    inside = x_in & y_in

    visible = bool(np.any(inside))
    if visible:
        u_min, v_min = uv_f[:, 0].min(), uv_f[:, 1].min()
        u_max, v_max = uv_f[:, 0].max(), uv_f[:, 1].max()
        bbox = (float(u_min), float(v_min), float(u_max), float(v_max))
    else:
        bbox = None

    rep = obj.representative_point()
    rep_uv, rep_z = project_points(rep[None, :], intrinsic, extrinsic_c2w_bim)
    center_uv = (float(rep_uv[0, 0]), float(rep_uv[0, 1])) if rep_z[0] > 0 else None
    center_depth = float(rep_z[0]) if rep_z[0] > 0 else None

    view_angle = None
    if obj.start is not None and obj.end is not None and center_depth is not None:
        axis = obj.end - obj.start
        if np.linalg.norm(axis) > 1e-6:
            axis = axis / np.linalg.norm(axis)
            cam_C = extrinsic_c2w_bim[:3, 3]
            view = rep - cam_C
            view = view / max(np.linalg.norm(view), 1e-8)
            cos = abs(float(np.dot(axis, view)))
            view_angle = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

    # crude score: fraction of samples in-frame and in front of camera.
    score = float(inside.sum() / sample_pts.shape[0]) if visible else 0.0

    return ProjectionResult(
        guid=obj.guid,
        frame_number=frame_number,
        visible=visible,
        projected_bbox=bbox,
        center_uv=center_uv,
        depth=center_depth,
        view_angle_deg=view_angle,
        score=score,
    )

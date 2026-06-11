"""Sim(3) alignment: lingbot_world -> bim_world.

Uses Umeyama's closed-form solution to compute the similarity transform
(scale, rotation, translation) from N>=3 correspondence pairs.

References:
  Umeyama, "Least-Squares Estimation of Transformation Parameters Between
  Two Point Patterns", IEEE PAMI 1991.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Sim3:
    """y = s * R @ x + t"""
    scale: float
    rotation: np.ndarray  # (3,3)
    translation: np.ndarray  # (3,)

    def apply(self, points: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(points)
        out = self.scale * (self.rotation @ pts.T).T + self.translation
        return out.reshape(points.shape) if points.ndim == 1 else out

    def apply_pose_c2w(self, extrinsic_c2w: np.ndarray) -> np.ndarray:
        """Transform a (3,4) c2w pose into the target frame.

        For a similarity (scale,R,t): the rotation of the pose is R_new = R @ R_cam,
        and the translation is s*R@C + t.
        Camera intrinsics are unaffected; the scale only modifies the metric
        baseline (interpret subsequent depths as scaled by `scale`).
        """
        R_cam = extrinsic_c2w[:3, :3]
        C_cam = extrinsic_c2w[:3, 3]
        R_new = self.rotation @ R_cam
        C_new = self.scale * (self.rotation @ C_cam) + self.translation
        out = np.zeros((3, 4), dtype=np.float64)
        out[:3, :3] = R_new
        out[:3, 3] = C_new
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": float(self.scale),
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
        }

    @classmethod
    def identity(cls) -> "Sim3":
        return cls(1.0, np.eye(3), np.zeros(3))


@dataclass
class AlignmentResult:
    transform: Sim3
    rmse: float
    n_correspondences: int
    per_point_residuals: np.ndarray
    scale_residual: float
    quality: str  # "green" | "yellow" | "review"
    source: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transform": self.transform.to_dict(),
            "rmse": float(self.rmse),
            "n_correspondences": int(self.n_correspondences),
            "per_point_residuals": self.per_point_residuals.tolist(),
            "scale_residual": float(self.scale_residual),
            "quality": self.quality,
            "source": self.source,
        }


def solve_sim3_umeyama(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    *,
    estimate_scale: bool = True,
    rmse_green: float = 0.5,
    rmse_yellow: float = 1.0,
) -> AlignmentResult:
    """Solve y = s * R @ x + t mapping src -> dst.

    src_points, dst_points: (N,3) arrays of corresponding points.
    """
    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"src/dst must be (N,3) and match; got {src.shape}, {dst.shape}")
    n = src.shape[0]
    if n < 3:
        raise ValueError(f"need >=3 correspondences, got {n}")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt

    src_var = (src_c ** 2).sum() / n
    if estimate_scale and src_var > 1e-12:
        scale = float((D * np.diag(S)).sum() / src_var)
    else:
        scale = 1.0

    t = dst_mean - scale * R @ src_mean

    transform = Sim3(scale=scale, rotation=R, translation=t)
    pred = transform.apply(src)
    residuals = np.linalg.norm(pred - dst, axis=1)
    rmse = float(np.sqrt((residuals ** 2).mean()))

    # scale residual: how anisotropic the residuals are relative to RMSE.
    scale_residual = float(residuals.std()) if n > 1 else 0.0

    if rmse <= rmse_green:
        quality = "green"
    elif rmse <= rmse_yellow:
        quality = "yellow"
    else:
        quality = "review"

    return AlignmentResult(
        transform=transform,
        rmse=rmse,
        n_correspondences=n,
        per_point_residuals=residuals,
        scale_residual=scale_residual,
        quality=quality,
    )


def load_correspondences(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load correspondence file.

    Format:
    {
      "pairs": [
        {"lingbot": [x,y,z], "bim": [x,y,z], "label": "optional"},
        ...
      ]
    }
    """
    with open(path) as f:
        data = json.load(f)
    pairs = data["pairs"]
    src = np.array([p["lingbot"] for p in pairs], dtype=np.float64)
    dst = np.array([p["bim"] for p in pairs], dtype=np.float64)
    return src, dst

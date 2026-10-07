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


# --- Gasan coverage ops: start-hint 2-click + correspondence Sim3 gates ---

START_DIR_MIN_SEP_M = 0.3
RMSE_GREEN_M = 0.10
RMSE_YELLOW_M = 0.25
_YAW_REFINE_DEG = (-8.0, -4.0, 0.0, 4.0, 8.0)


def start_direction_ok(p0, p1, min_sep_m: float = START_DIR_MIN_SEP_M) -> bool:
    """Second click of start-hint must be far enough to define a walk heading."""
    a = np.asarray(p0, dtype=np.float64).reshape(-1)
    b = np.asarray(p1, dtype=np.float64).reshape(-1)
    if a.size < 2 or b.size < 2:
        return False
    return float(np.hypot(b[0] - a[0], b[1] - a[1])) >= float(min_sep_m)


def yaw_candidates_from_user_direction(
    direction_az_deg: float,
    scan_axis_deg: float,
    *,
    refine_deg: tuple[float, ...] = _YAW_REFINE_DEG,
) -> list[float]:
    """Map user walk heading + scan PCA axis onto a small yaw set (no 90° flood).

    Axes are unsigned (mod 180). Flip ±180 covers PCA sign ambiguity.
    """
    m = float(direction_az_deg) % 180.0
    sax = float(scan_axis_deg) % 180.0
    cand: set[float] = set()
    for flip in (0.0, 180.0):
        base = (m - sax + flip) % 360.0
        for dd in refine_deg:
            cand.add(round((base + dd) % 360.0, 1))
    return sorted(cand)


def correspondence_spread(points: np.ndarray) -> dict:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return {"extent_m": 0.0, "rms_radius_m": 0.0, "rank": 0, "singular_values": []}
    centered = pts - pts.mean(axis=0)
    extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    rms = float(np.sqrt((centered ** 2).sum(axis=1).mean())) if pts.shape[0] else 0.0
    if pts.shape[0] >= 2:
        singular = np.linalg.svd(centered, compute_uv=False)
        rel = singular / max(float(singular[0]), 1e-9)
        rank = int((rel > 0.08).sum())
    else:
        singular = np.zeros(0, dtype=np.float64)
        rank = 0
    return {
        "extent_m": extent,
        "rms_radius_m": rms,
        "rank": rank,
        "singular_values": singular.astype(float).tolist(),
    }


def correspondence_spread_ok(scan_spread: dict, model_spread: dict) -> bool:
    return (
        scan_spread["extent_m"] >= 0.15
        and model_spread["extent_m"] >= 0.50
        and scan_spread["rank"] >= 2
        and model_spread["rank"] >= 2
    )


def alignment_permits_coverage_analysis(alignment: dict | None) -> bool:
    """Invariant: red (or missing) alignment never feeds coverage / observed."""
    return isinstance(alignment, dict) and alignment.get("quality") in ("green", "yellow")


def auto_geometric_alignment_record(al: dict, quality: str) -> dict:
    """Persist the searched Sim3, including scale used to build translation."""
    return {
        "quality": quality,
        "stable": False,
        "rmse_m": None,
        "scale": float(al["scale"]),
        "rotation": al["rotation"],
        "translation": al["translation"],
        "n": 0,
        "note": "auto_geometric_gravity_aligned",
        "pairs": [],
    }


def solve_scan_to_model_alignment(pairs: list[dict]) -> dict:
    """Sim3 from scan/model pairs. <4 pairs or collinear/clustered → quality red."""
    if len(pairs) < 4:
        return {
            "quality": "red",
            "stable": False,
            "rmse_m": None,
            "scale": 1.0,
            "rotation": np.eye(3).tolist(),
            "translation": [0, 0, 0],
            "n": len(pairs),
            "note": "need >=4 correspondences; using identity for preview",
        }
    src = np.asarray([p["scan"] for p in pairs], dtype=np.float64)
    dst = np.asarray([p["model"] for p in pairs], dtype=np.float64)
    src_spread = correspondence_spread(src)
    dst_spread = correspondence_spread(dst)
    spread_ok = correspondence_spread_ok(src_spread, dst_spread)
    result = solve_sim3_umeyama(src, dst, rmse_green=RMSE_GREEN_M, rmse_yellow=RMSE_YELLOW_M)
    loo_errors = []
    if len(pairs) >= 5:
        for i in range(len(pairs)):
            keep = [j for j in range(len(pairs)) if j != i]
            r = solve_sim3_umeyama(src[keep], dst[keep], rmse_green=RMSE_GREEN_M, rmse_yellow=RMSE_YELLOW_M)
            pred = r.transform.apply(src[i])
            loo_errors.append(float(np.linalg.norm(pred - dst[i])))
    max_loo = max(loo_errors) if loo_errors else float(result.rmse)
    quality = "red" if result.quality == "review" else result.quality
    stable = bool(max_loo <= max(0.25, float(result.rmse) * 2.5))
    if not stable and quality == "green":
        quality = "yellow"
    if not stable and quality == "yellow":
        quality = "red"
    spread_note = None
    if not spread_ok:
        quality = "red"
        stable = False
        spread_note = "correspondence points are too clustered or near-collinear; distribute points across the scan/model area"
    return {
        "quality": quality,
        "stable": stable,
        "rmse_m": float(result.rmse),
        "max_leave_one_out_rmse_m": max_loo,
        "correspondence_spread": {
            "ok": spread_ok,
            "scan": src_spread,
            "model": dst_spread,
            "note": spread_note,
        },
        "scale": float(result.transform.scale),
        "rotation": result.transform.rotation.tolist(),
        "translation": result.transform.translation.tolist(),
        "n": int(result.n_correspondences),
        "per_point_residuals": result.per_point_residuals.tolist(),
    }

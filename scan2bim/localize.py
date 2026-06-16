"""FR-2.3 stage-2: automatic per-frame camera localization in the MODEL frame via
object-anchor PnP — removes the manual ground-truth path.

Detected anchors (OWL-ViT, scan2bim.detect) are matched to model anchors
(scan2bim.anchors) of the same type → 2D-3D correspondences → cv2.solvePnPRansac
→ camera pose (world-to-camera R,t) in model coordinates. Repeated geometry
(parallel pipes) is disambiguated by distinctive objects (AC, lights, columns).

Bootstrapped: a rough per-frame pose prior (from the reconstruction trajectory)
projects model anchors into the frame; each detection pairs to the nearest
projected anchor of its type; PnP refines. Conventions = OpenCV: X_cam = R·X + t,
pixel = K·X_cam / z, camera center in world = −Rᵀ t.
"""
from __future__ import annotations

import numpy as np

# detect.DEFAULT_PROMPTS label -> anchors.classify_anchor type
LABEL2TYPE = {
    "ceiling air conditioner": "ac", "ceiling light": "light",
    "structural column": "column", "red pipe": "pipe", "duct": "duct",
}


def intrinsics(w: int, h: int, hfov_deg: float = 69.0) -> np.ndarray:
    """Pinhole K from horizontal FOV (monocular phone footage, no stored intrinsics)."""
    f = (w / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    return np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]], float)


def project(K, R, t, P):
    """World points P (N,3) → (pixels (N,2), depth (N,)). R,t world-to-camera."""
    P = np.atleast_2d(np.asarray(P, float))
    cam = P @ np.asarray(R, float).T + np.asarray(t, float)   # X_cam = R X + t
    z = cam[:, 2]
    uv = cam @ np.asarray(K, float).T
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    return uv[:, :2] / safe[:, None], z


def camera_center(R, t) -> np.ndarray:
    return -np.asarray(R, float).T @ np.asarray(t, float)


def match_by_projection(dets, anchors, K, R, t, *, max_px=90.0, img_wh=None):
    """Pair detections to model anchors via a pose prior.

    dets:    [{label, cx, cy, score}]  (scan2bim.detect output)
    anchors: [{type, center[x,y,z]}]   (scan2bim.anchors, typed)
    Returns [(P3d(3,), pix2d(2,), type)] for same-type nearest-projection pairs.
    """
    proj: dict = {}
    for a in anchors:
        uv, z = project(K, R, t, [a["center"]])
        if z[0] <= 0:
            continue
        u, v = uv[0]
        if img_wh and not (0 <= u <= img_wh[0] and 0 <= v <= img_wh[1]):
            continue
        proj.setdefault(a["type"], []).append((np.array([u, v]), np.asarray(a["center"], float)))
    corr = []
    for d in dets:
        typ = LABEL2TYPE.get(d["label"])
        cand = proj.get(typ)
        if not cand:
            continue
        dp = np.array([float(d["cx"]), float(d["cy"])])
        dist = [float(np.linalg.norm(dp - p)) for p, _ in cand]
        j = int(np.argmin(dist))
        if dist[j] <= max_px:
            corr.append((cand[j][1], dp, typ))
    return corr


def solve_pnp(K, pts3d, pts2d, *, ransac_px=14.0, prior=None):
    """cv2.solvePnPRansac → (R(3,3), t(3,), inlier_idx). None if <4 pts or fail."""
    import cv2
    obj = np.asarray(pts3d, np.float64).reshape(-1, 1, 3)
    img = np.asarray(pts2d, np.float64).reshape(-1, 1, 2)
    if len(obj) < 4:
        return None
    kw = dict(reprojectionError=float(ransac_px), iterationsCount=300, flags=cv2.SOLVEPNP_EPNP)
    rv0 = tv0 = None
    if prior is not None:
        Rp, tp = prior
        rv0, _ = cv2.Rodrigues(np.asarray(Rp, np.float64))
        tv0 = np.asarray(tp, np.float64).reshape(3, 1)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, np.asarray(K, np.float64), None, **kw)
    if not ok or inl is None or len(inl) < 4:
        return None
    # non-linear refine on the inliers
    inl = inl.ravel()
    cv2.solvePnPRefineLM(obj[inl], img[inl], np.asarray(K, np.float64), None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.ravel(), inl


def localize_frame(K, dets, anchors, prior_R, prior_t, *, max_px=90.0, ransac_px=14.0, img_wh=None):
    """One frame: prior pose → match → PnP. Returns {R,t,center,n_corr,n_inlier,
    rmse} or None if not localizable (too few anchor matches)."""
    corr = match_by_projection(dets, anchors, K, prior_R, prior_t, max_px=max_px, img_wh=img_wh)
    if len(corr) < 4:
        return None
    P3 = np.array([c[0] for c in corr]); P2 = np.array([c[1] for c in corr])
    res = solve_pnp(K, P3, P2, ransac_px=ransac_px, prior=(prior_R, prior_t))
    if res is None:
        return None
    R, t, inl = res
    uv, z = project(K, R, t, P3[inl])
    rmse = float(np.sqrt(((uv - P2[inl]) ** 2).sum(1).mean()))
    return {"R": R, "t": t, "center": camera_center(R, t),
            "n_corr": len(corr), "n_inlier": int(len(inl)), "rmse": rmse}

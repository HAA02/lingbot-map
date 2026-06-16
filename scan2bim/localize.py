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


def pose_from_lookat(center, forward, up):
    """(camera center, forward=look dir, up) in world → world-to-camera (R, t).
    OpenCV: camera looks +Z_cam. Used to turn co-play trajectory poses {c,f,u}
    into PnP priors. center = -Rᵀ t; det(R)=+1."""
    c = np.asarray(center, float); f = np.asarray(forward, float); u = np.asarray(up, float)
    z = f / (np.linalg.norm(f) + 1e-12)
    x = np.cross(u, z); x /= (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    R = np.array([x, y, z])           # rows = camera axes in world
    return R, -R @ c


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


def localize_trajectory(K, prior_poses, dets_by_frame, anchors, *, min_inliers=5,
                        max_px=120.0, ransac_px=14.0, max_rmse=25.0, smooth_win=5):
    """Per-frame object-anchor PnP over a whole sequence → refined model-frame poses.

    prior_poses: [(R,t), ...] world-to-camera priors (from the trajectory).
    dets_by_frame: [[det,...], ...] aligned with prior_poses.
    Frames with enough confident anchor matches get a PnP pose; the rest have their
    camera CENTER linearly interpolated from the PnP frames (orientation kept from
    prior). Centers are moving-average smoothed. Returns (per-frame dicts, summary).
    """
    n = len(prior_poses)
    pnp = [None] * n
    for i in range(n):
        Rp, tp = prior_poses[i]
        out = localize_frame(K, dets_by_frame[i], anchors, Rp, tp, max_px=max_px, ransac_px=ransac_px)
        if out and out["n_inlier"] >= min_inliers and out["rmse"] <= max_rmse:
            pnp[i] = out
    loc = [i for i in range(n) if pnp[i] is not None]
    # temporal-consistency outlier rejection: planar ceilings make some PnP poses
    # jump (low pixel-rmse but wrong depth). Demote frames whose center deviates
    # from the local median of PnP centers → they get interpolated instead.
    if len(loc) >= 5:
        pc = np.array([pnp[i]["center"] for i in loc])
        med = np.array([np.median(pc[max(0, j - 2):j + 3], axis=0) for j in range(len(loc))])
        for j, i in enumerate(loc):
            if np.linalg.norm(pc[j] - med[j]) > 0.5:
                pnp[i] = None
        loc = [i for i in range(n) if pnp[i] is not None]
    centers = np.array([camera_center(*prior_poses[i]) for i in range(n)], float)  # fallback = prior
    if loc:
        pc = np.array([pnp[i]["center"] for i in loc])
        for k in range(3):
            centers[:, k] = np.interp(np.arange(n), loc, pc[:, k])
    if smooth_win > 1 and n >= smooth_win:
        ker = np.ones(smooth_win) / smooth_win
        pad = smooth_win // 2
        for k in range(3):
            ext = np.concatenate([np.repeat(centers[0, k], pad), centers[:, k], np.repeat(centers[-1, k], pad)])
            centers[:, k] = np.convolve(ext, ker, "valid")[:n]
    out_list = []
    for i in range(n):
        R = pnp[i]["R"] if pnp[i] else prior_poses[i][0]
        out_list.append({"R": R, "t": -R @ centers[i], "center": centers[i],
                         "status": "pnp" if pnp[i] else "interp",
                         "n_inlier": pnp[i]["n_inlier"] if pnp[i] else 0})
    return out_list, {"n_frames": n, "n_pnp": len(loc), "pnp_frac": round(len(loc) / max(n, 1), 3)}

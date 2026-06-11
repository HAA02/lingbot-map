"""Tier 3 — Multi-window registration via Procrustes (umeyama) on overlap frames.

When a video is too long to fit in one lingbot inference window (typical limit 32–64
frames), we process it in overlapping windows. Each window produces its own world
frame; this module aligns successive windows into a common frame using the
overlapping camera centers as paired correspondences.
"""
from __future__ import annotations

import numpy as np
from lingbot_map.utils.geometry import umeyama


def procrustes_align(src_centers: np.ndarray, dst_centers: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Find (s, R, t) such that s*R @ src + t ≈ dst.

    Args:
        src_centers, dst_centers: (K, 3) point sets in correspondence.
    Returns:
        s: scale, R: (3,3) rotation, t: (3,) translation.
    """
    assert src_centers.shape == dst_centers.shape
    assert src_centers.shape[1] == 3
    # umeyama wants (m, n) = (3, K)
    s, R, t = umeyama(src_centers.T.astype(np.float64),
                      dst_centers.T.astype(np.float64))
    return float(s), R.astype(np.float32), t.flatten().astype(np.float32)


def apply_similarity_to_points(xyz: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """xyz' = s*R @ xyz + t. Input shape (M, 3)."""
    return (s * (xyz @ R.T) + t[None, :]).astype(np.float32)


def apply_similarity_to_c2w(c2w: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Transform a batch of (N, 3, 4) c2w matrices by similarity (s, R, t).

    For a c2w that places the camera in source-world coords, the new c2w'
    placing it in target-world coords is:
        R_new = R @ R_c2w
        t_new = s*R @ t_c2w + t
    Scale is applied only to the translation (camera position), not the rotation,
    so that the camera orientation in the new frame is consistent.
    """
    out = c2w.copy().astype(np.float32)
    Rc = c2w[:, :3, :3]                 # (N, 3, 3)
    tc = c2w[:, :3, 3]                  # (N, 3)
    out[:, :3, :3] = R @ Rc             # (N, 3, 3) broadcast: (3,3) @ (N,3,3)
    out[:, :3, 3]  = s * (tc @ R.T) + t  # (N, 3)
    return out


def apply_similarity_to_w2c(w2c: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """w2c is the inverse of c2w. Easier to compute c2w → transform → invert."""
    from lingbot_map.utils.geometry import closed_form_inverse_se3
    # 3x4 → 4x4 for inverse
    N = w2c.shape[0]
    w2c44 = np.tile(np.eye(4, dtype=np.float32)[None], (N, 1, 1))
    w2c44[:, :3, :4] = w2c
    c2w44 = closed_form_inverse_se3(w2c44)
    c2w_new = apply_similarity_to_c2w(c2w44[:, :3, :], s, R, t)
    # back to w2c
    c2w44_new = np.tile(np.eye(4, dtype=np.float32)[None], (N, 1, 1))
    c2w44_new[:, :3, :4] = c2w_new
    return closed_form_inverse_se3(c2w44_new)[:, :3, :].astype(np.float32)


def icp_refine_rigid(
    src: np.ndarray, dst: np.ndarray, *,
    max_iter: int = 12, reject_pct: float = 70.0, tol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Point-to-point ICP with reciprocal-NN matching and percentile-based outlier
    rejection. Refines a rigid (R, t) that aligns `src` → `dst`. Scale is held
    at 1 (use after a Sim(3) seed has already absorbed the scale).

    Returns (R_total, t_total, final_rmse). Both src and dst are (M, 3) arrays.
    """
    from scipy.spatial import cKDTree
    if src.shape[0] < 3 or dst.shape[0] < 3:
        return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), float("inf")
    R_total = np.eye(3, dtype=np.float64)
    t_total = np.zeros(3, dtype=np.float64)
    cur = src.astype(np.float64)
    prev_rmse = float("inf")
    dst_tree = cKDTree(dst)
    for _ in range(max_iter):
        d, idx = dst_tree.query(cur, k=1)
        thresh = np.percentile(d, reject_pct)
        mask = d < thresh
        if mask.sum() < 3:
            break
        src_c = cur[mask]
        dst_c = dst[idx[mask]]
        src_mean = src_c.mean(0); dst_mean = dst_c.mean(0)
        H = (src_c - src_mean).T @ (dst_c - dst_mean)
        U, _, Vt = np.linalg.svd(H)
        if np.linalg.det(Vt.T @ U.T) < 0:
            Vt[-1] *= -1
        R_step = Vt.T @ U.T
        t_step = dst_mean - R_step @ src_mean
        cur = (R_step @ cur.T).T + t_step
        # Compose into running transform
        R_total = R_step @ R_total
        t_total = R_step @ t_total + t_step
        rmse = float(np.sqrt((d[mask] ** 2).mean()))
        if abs(prev_rmse - rmse) < tol:
            break
        prev_rmse = rmse
    return R_total.astype(np.float32), t_total.astype(np.float32), float(prev_rmse)


def chain_windows(windows: list[dict], overlap: int, *, use_icp: bool = True) -> dict:
    """Align a list of per-window result dicts into a common frame.

    Each window dict must contain:
        c2w:    (N_k, 3, 4) cam-to-world in window-local frame
        w2c:    (N_k, 3, 4) world-to-cam
        xyz:    (M_k, 3)    world-frame points
        rgb:    (M_k, 3)    uint8 colors
        conf:   (M_k,)      per-point confidence
        source: (M_k,)      per-point source frame id
        frame_ids: (N_k,)   source frame ids for each pose/depth frame
        depth:  (N_k, H, W) per-frame depth in camera Z (not transformed by alignment)
        thumbs: list of (H, W, 3) RGB images (not transformed)
        K:      (N_k, 3, 3) intrinsics (not transformed)

    `overlap` is the number of frames shared between adjacent windows:
    last `overlap` poses of window K are physically the same as first `overlap`
    of window K+1.

    Returns a single merged dict in the first window's frame, with `frame_count`
    summing the deduplicated per-window pose counts.
    """
    if len(windows) == 0:
        raise ValueError("no windows")
    if len(windows) == 1:
        out = dict(windows[0])
        out["frame_count"] = windows[0]["c2w"].shape[0]
        return out

    aligned = [windows[0]]
    cum_s, cum_R, cum_t = 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    for k in range(1, len(windows)):
        ref = aligned[-1]
        new = windows[k]
        # 1) Procrustes on overlap cam centers (Sim3, absorbs scale + coarse pose).
        src = new["c2w"][:overlap, :3, 3]
        dst = ref["c2w"][-overlap:, :3, 3]
        s_k, R_k, t_k = procrustes_align(src, dst)
        # Apply Procrustes to new window
        xyz_new = apply_similarity_to_points(new["xyz"], s_k, R_k, t_k)
        c2w_new = apply_similarity_to_c2w(new["c2w"], s_k, R_k, t_k)
        w2c_new = apply_similarity_to_w2c(new["w2c"], s_k, R_k, t_k)
        # Depth is a metric distance in the same local world scale as the
        # window poses.  If a Sim(3) scale aligns the window into the reference
        # frame, TSDF integration must see depth in that reference scale too;
        # otherwise camera poses and observed surfaces disagree and duplicate
        # shells survive in the reconstructed mesh.
        depth_new = new["depth"] * s_k

        # 2) ICP refinement on overlap-region surface points (rigid).
        # Take points belonging to overlap frames of each window for surface-level
        # alignment that goes beyond cam-center fit.
        icp_R, icp_t, icp_rmse = np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), float("nan")
        if use_icp:
            # Subsample to keep ICP fast: up to 8000 points from each overlap region.
            # Heuristic: points from a window with conf in top-50%.
            try:
                ref_frame_ids = ref.get("frame_ids")
                new_frame_ids = new.get("frame_ids")
                if ref_frame_ids is not None and new_frame_ids is not None:
                    ref_ids = np.asarray(ref_frame_ids[-overlap:], dtype=np.uint32)
                    new_ids = np.asarray(new_frame_ids[:overlap], dtype=np.uint32)
                    ref_mask = np.isin(ref.get("source"), ref_ids)
                    new_mask = np.isin(new.get("source"), new_ids)
                    ref_pts = ref["xyz"][ref_mask]
                    new_pts = xyz_new[new_mask]
                else:
                    ref_pts = ref["xyz"]
                    new_pts = xyz_new
                if ref_pts.shape[0] < 3 or new_pts.shape[0] < 3:
                    raise ValueError("not enough overlap points for ICP")
                if ref_pts.shape[0] > 8000:
                    idx = np.random.default_rng(k * 13).choice(ref_pts.shape[0], 8000, replace=False)
                    ref_pts = ref_pts[idx]
                if new_pts.shape[0] > 8000:
                    idx = np.random.default_rng(k * 17).choice(new_pts.shape[0], 8000, replace=False)
                    new_pts = new_pts[idx]
                icp_R, icp_t, icp_rmse = icp_refine_rigid(new_pts, ref_pts, max_iter=15, reject_pct=70.0)
                # Apply ICP correction to xyz and c2w
                xyz_new = (xyz_new @ icp_R.T) + icp_t
                # c2w: rotation gets premultiplied; translation = R @ t_old + t_step
                R_old = c2w_new[:, :3, :3]
                t_old = c2w_new[:, :3, 3]
                c2w_new = c2w_new.copy()
                c2w_new[:, :3, :3] = icp_R @ R_old
                c2w_new[:, :3, 3]  = (t_old @ icp_R.T) + icp_t
                # w2c: recompute via inverse
                from lingbot_map.utils.geometry import closed_form_inverse_se3
                N = c2w_new.shape[0]
                c44 = np.tile(np.eye(4, dtype=np.float32)[None], (N, 1, 1))
                c44[:, :3, :4] = c2w_new
                w2c_new = closed_form_inverse_se3(c44)[:, :3, :].astype(np.float32)
            except Exception:
                pass

        new_aligned = {
            "c2w":  c2w_new,
            "w2c":  w2c_new,
            "xyz":  xyz_new,
            "rgb":  new["rgb"],
            "conf": new["conf"],
            "source": new.get("source", np.zeros(new["xyz"].shape[0], dtype=np.uint32)),
            "frame_ids": new.get("frame_ids"),
            "depth":  depth_new.astype(np.float32, copy=False),
            "thumbs": new["thumbs"],
            "K":      new["K"],
            "scale_to_ref": s_k,
            "icp_rmse": icp_rmse,
        }
        aligned.append(new_aligned)

    # Concatenate, dropping the overlap from each non-first window to avoid
    # duplicate camera poses (we keep them as distinct so TSDF gets all observations,
    # but here we drop dupes for clarity in the merged result).
    def cat_drop_overlap(field):
        if len(aligned) == 1:
            return aligned[0][field]
        parts = [aligned[0][field]]
        for k in range(1, len(aligned)):
            parts.append(aligned[k][field][overlap:])
        return np.concatenate(parts, axis=0)

    # Per-frame fields (c2w, w2c, depth, K) → drop overlap to avoid duplicates
    merged = {
        "c2w":   cat_drop_overlap("c2w"),
        "w2c":   cat_drop_overlap("w2c"),
        "depth": cat_drop_overlap("depth"),
        "K":     cat_drop_overlap("K"),
        "frame_ids": cat_drop_overlap("frame_ids") if "frame_ids" in aligned[0] else None,
        # Per-point fields (xyz, rgb, conf) → keep all (overlap regions produce
        # extra observations of the same surface, which TSDF / Tier1 fusion will average).
        "xyz":  np.concatenate([a["xyz"]  for a in aligned], axis=0),
        "rgb":  np.concatenate([a["rgb"]  for a in aligned], axis=0),
        "conf": np.concatenate([a["conf"] for a in aligned], axis=0),
        "source": np.concatenate([
            a.get("source", np.zeros(a["xyz"].shape[0], dtype=np.uint32))
            for a in aligned
        ], axis=0).astype(np.uint32),
    }
    # thumbs is a list — same drop-overlap logic
    thumbs = list(aligned[0]["thumbs"])
    for k in range(1, len(aligned)):
        thumbs.extend(aligned[k]["thumbs"][overlap:])
    merged["thumbs"] = thumbs
    merged["frame_count"] = merged["c2w"].shape[0]
    merged["n_windows"] = len(windows)
    merged["scales_to_ref"] = [1.0] + [a["scale_to_ref"] for a in aligned[1:]]
    return merged

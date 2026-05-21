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


def chain_windows(windows: list[dict], overlap: int) -> dict:
    """Align a list of per-window result dicts into a common frame.

    Each window dict must contain:
        c2w:    (N_k, 3, 4) cam-to-world in window-local frame
        w2c:    (N_k, 3, 4) world-to-cam
        xyz:    (M_k, 3)    world-frame points
        rgb:    (M_k, 3)    uint8 colors
        conf:   (M_k,)      per-point confidence
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
        # Overlap correspondences
        src = new["c2w"][:overlap, :3, 3]   # first overlap of new window (in its local frame)
        dst = ref["c2w"][-overlap:, :3, 3]  # last overlap of previous (already in cumulative frame)
        s_k, R_k, t_k = procrustes_align(src, dst)
        # Compose with running cumulative (we already have ref in target frame)
        # We just need the transform from new-local → ref's (already-aligned) frame.
        new_aligned = {
            "c2w":  apply_similarity_to_c2w(new["c2w"], s_k, R_k, t_k),
            "w2c":  apply_similarity_to_w2c(new["w2c"], s_k, R_k, t_k),
            "xyz":  apply_similarity_to_points(new["xyz"], s_k, R_k, t_k),
            "rgb":  new["rgb"],
            "conf": new["conf"],
            # Depth values are in camera frame, unchanged by world-space similarity.
            "depth":  new["depth"],
            "thumbs": new["thumbs"],
            "K":      new["K"],
            "scale_to_ref": s_k,
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
        # Per-point fields (xyz, rgb, conf) → keep all (overlap regions produce
        # extra observations of the same surface, which TSDF / Tier1 fusion will average).
        "xyz":  np.concatenate([a["xyz"]  for a in aligned], axis=0),
        "rgb":  np.concatenate([a["rgb"]  for a in aligned], axis=0),
        "conf": np.concatenate([a["conf"] for a in aligned], axis=0),
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

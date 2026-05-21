"""Lightweight TSDF (Truncated Signed Distance Function) volume + Marching Cubes.

Pure numpy + scikit-image — no Open3D needed (Open3D doesn't yet support Python 3.14).

Pipeline:
    1. Build a TSDF voxel grid in world space (bounds inferred from camera trajectory + max depth).
    2. For each frame: project voxel centers into the camera, sample observed depth at
       the projected pixel, compute signed distance d = depth_obs - voxel_z (in camera frame).
       Truncate to [-trunc, +trunc] and accumulate weighted average per voxel.
    3. Extract zero-isosurface via Marching Cubes → triangle mesh.
    4. Sample colors per vertex by re-projecting into the best-visible frame.

Performance notes:
    - For a 5×5×5 m volume at 2 cm voxels: 250³ ≈ 15M voxels. Per-frame projection is
      one big matmul → fast even in numpy.
    - 32 frames × 15M voxels × ~50 fp32 ops = ~24 GFLOPS total. Few seconds on CPU.
"""
from __future__ import annotations

import numpy as np
from skimage.measure import marching_cubes


def build_tsdf_volume(
    depth_maps: np.ndarray,       # (N, H, W) float32 — per-frame depth in meters
    w2c: np.ndarray,              # (N, 3, 4) float32 — world→camera (OpenCV)
    K: np.ndarray,                # (N, 3, 3) float32 — camera intrinsics
    confs: np.ndarray | None,     # (N, H, W) float32 — per-pixel confidence (weight)
    voxel_size: float = 0.02,
    trunc: float = 0.08,
    conf_floor: float = 1.0,      # ignore observations below this confidence
    pad: float = 0.3,             # extra world-space padding around camera trajectory
):
    """Fuse per-frame depth maps into a TSDF volume.

    Returns a dict with:
        tsdf:     (Dx, Dy, Dz) float32, truncated signed distance ∈ [-1, 1]
        weights:  (Dx, Dy, Dz) float32, accumulated observation weight
        origin:   (3,) float32, world-space coord of voxel (0,0,0)
        voxel_size: scalar
    """
    N, H, W = depth_maps.shape
    assert w2c.shape == (N, 3, 4)
    assert K.shape == (N, 3, 3)

    # ── 1. Decide grid bounds from camera positions + depth fan ─────────────
    # c2w translation = camera center in world
    c2w = _invert_w2c_batch(w2c)
    cam_centers = c2w[:, :, 3]  # (N, 3)

    # Estimate scene span as cam motion + max trunc-bounded depth in each direction.
    # Cheap: just use cam_span + a fixed depth margin. (TSDF auto-clips beyond grid.)
    max_depth = np.percentile(depth_maps[depth_maps > 0.05], 95) if (depth_maps > 0.05).any() else 3.0
    margin = max_depth + pad
    mn = cam_centers.min(axis=0) - margin
    mx = cam_centers.max(axis=0) + margin

    dims = np.ceil((mx - mn) / voxel_size).astype(np.int32)
    # Safety cap: avoid runaway memory (e.g., bad depth gives 10m fan in a 0.5m scene).
    dims = np.clip(dims, 16, 350)
    Dx, Dy, Dz = int(dims[0]), int(dims[1]), int(dims[2])
    origin = mn.astype(np.float32)

    tsdf = np.zeros((Dx, Dy, Dz), dtype=np.float32)
    weights = np.zeros((Dx, Dy, Dz), dtype=np.float32)

    # Precompute voxel-center world coords. ~15M floats per axis = ~180 MB total for big grids;
    # do this as broadcasted arange to save memory.
    xs = origin[0] + (np.arange(Dx, dtype=np.float32) + 0.5) * voxel_size
    ys = origin[1] + (np.arange(Dy, dtype=np.float32) + 0.5) * voxel_size
    zs = origin[2] + (np.arange(Dz, dtype=np.float32) + 0.5) * voxel_size
    # Flatten to (M, 3) for matmul
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    voxels_world = np.stack([XX, YY, ZZ], axis=-1).reshape(-1, 3)  # (M, 3)
    M = voxels_world.shape[0]
    voxels_world_h = np.concatenate([voxels_world, np.ones((M, 1), dtype=np.float32)], axis=1)  # (M, 4)

    # ── 2. Per-frame integration ────────────────────────────────────────────
    for i in range(N):
        Ri = w2c[i]                # (3, 4) world→cam
        Ki = K[i]                  # (3, 3)
        di = depth_maps[i]         # (H, W)
        ci = confs[i] if confs is not None else None  # (H, W)

        # Voxel centers in camera space
        cam = voxels_world_h @ Ri.T  # (M, 3)
        cam_z = cam[:, 2]
        valid_z = cam_z > 1e-3
        # Project to pixels
        # u = fx*x/z + cx ; v = fy*y/z + cy
        u = (Ki[0, 0] * cam[:, 0] + Ki[0, 2] * cam_z) / np.where(valid_z, cam_z, 1.0)
        v = (Ki[1, 1] * cam[:, 1] + Ki[1, 2] * cam_z) / np.where(valid_z, cam_z, 1.0)
        # Wait — that's wrong; should be u = fx*(x/z) + cx
        u = Ki[0, 0] * (cam[:, 0] / np.where(valid_z, cam_z, 1.0)) + Ki[0, 2]
        v = Ki[1, 1] * (cam[:, 1] / np.where(valid_z, cam_z, 1.0)) + Ki[1, 2]
        ui = np.floor(u).astype(np.int32)
        vi = np.floor(v).astype(np.int32)
        in_bounds = valid_z & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

        if not in_bounds.any():
            continue

        # Sample depth + confidence at projected pixel
        sel = np.where(in_bounds)[0]
        d_obs = di[vi[sel], ui[sel]]                 # observed depth at that pixel
        c_obs = ci[vi[sel], ui[sel]] if ci is not None else np.ones(sel.size, np.float32)
        z_vox = cam_z[sel]

        # Signed distance: positive = voxel in front of surface (toward camera);
        # negative = voxel behind surface. Standard TSDF convention.
        sdf = d_obs - z_vox
        # Validity: observed depth > 0, conf high enough, within truncation band
        good = (d_obs > 0.05) & (c_obs > conf_floor) & (np.abs(sdf) < trunc * 2.0)
        if not good.any():
            continue
        sel = sel[good]
        sdf = sdf[good]
        c_obs = c_obs[good]

        # Truncate & normalize: TSDF stored in [-1, 1]
        tsdf_obs = np.clip(sdf / trunc, -1.0, 1.0).astype(np.float32)
        w_obs = c_obs.astype(np.float32)

        # Voxel indices for accumulation
        # Map flat index back to (ix, iy, iz)
        ix = sel // (Dy * Dz)
        iy = (sel % (Dy * Dz)) // Dz
        iz = sel % Dz

        # Online weighted average:  T' = (W*T + w*t) / (W + w)
        W_prev = weights[ix, iy, iz]
        T_prev = tsdf[ix, iy, iz]
        W_new = W_prev + w_obs
        T_new = (W_prev * T_prev + w_obs * tsdf_obs) / np.maximum(W_new, 1e-6)
        tsdf[ix, iy, iz] = T_new
        weights[ix, iy, iz] = W_new

    return {
        "tsdf": tsdf,
        "weights": weights,
        "origin": origin,
        "voxel_size": float(voxel_size),
    }


def extract_mesh(volume: dict, min_weight: float = 1.0):
    """Marching cubes on the TSDF zero-isosurface.

    Returns:
        verts (V, 3) float32 world coords, faces (F, 3) int32, normals (V, 3) float32.
        Empty arrays if no surface found.
    """
    tsdf = volume["tsdf"]
    w = volume["weights"]
    origin = volume["origin"]
    vs = volume["voxel_size"]

    # Mask out unobserved voxels (weight too low → treat as far-positive so MC won't cross)
    field = np.where(w >= min_weight, tsdf, 1.0).astype(np.float32)
    # Ensure both signs present
    if not (field.min() < 0 and field.max() > 0):
        return (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32),
                np.zeros((0, 3), np.float32))

    try:
        verts, faces, normals, _ = marching_cubes(field, level=0.0, spacing=(vs, vs, vs))
    except (ValueError, RuntimeError):
        return (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32),
                np.zeros((0, 3), np.float32))

    verts = (verts + origin).astype(np.float32)
    return verts, faces.astype(np.int32), normals.astype(np.float32)


def color_vertices(verts: np.ndarray, images: np.ndarray, w2c: np.ndarray, K: np.ndarray):
    """For each mesh vertex, sample color from the frame where it is most visible.

    Visibility heuristic: project vertex into each camera; pick the frame with smallest
    cam-frame |z| ratio against the camera center direction (i.e., closest visible).
    Simple but works for short clips.

    Args:
        verts:  (V, 3) world coords
        images: (N, H, W, 3) uint8 (already in OpenCV pixel order, RGB)
        w2c:    (N, 3, 4)
        K:      (N, 3, 3)
    Returns:
        colors: (V, 3) uint8
    """
    if verts.shape[0] == 0:
        return np.zeros((0, 3), np.uint8)
    V = verts.shape[0]
    N, H, W, _ = images.shape

    best_z = np.full(V, np.inf, dtype=np.float32)
    best_uv = np.full((V, 2), -1, dtype=np.int32)
    best_idx = np.full(V, -1, dtype=np.int32)

    verts_h = np.concatenate([verts, np.ones((V, 1), dtype=np.float32)], axis=1)
    for i in range(N):
        cam = verts_h @ w2c[i].T  # (V, 3)
        cz = cam[:, 2]
        valid_z = cz > 0.05
        u = K[i, 0, 0] * (cam[:, 0] / np.where(valid_z, cz, 1.0)) + K[i, 0, 2]
        v = K[i, 1, 1] * (cam[:, 1] / np.where(valid_z, cz, 1.0)) + K[i, 1, 2]
        ui = np.floor(u).astype(np.int32)
        vi = np.floor(v).astype(np.int32)
        in_bounds = valid_z & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        # Pick this frame if vertex is closer than current best
        candidate = in_bounds & (cz < best_z)
        idx = np.where(candidate)[0]
        best_z[idx] = cz[idx]
        best_uv[idx, 0] = ui[idx]
        best_uv[idx, 1] = vi[idx]
        best_idx[idx] = i

    colors = np.zeros((V, 3), dtype=np.uint8)
    for i in range(N):
        sel = np.where(best_idx == i)[0]
        if sel.size == 0:
            continue
        colors[sel] = images[i, best_uv[sel, 1], best_uv[sel, 0]]
    # Vertices that no camera saw: gray
    nosee = best_idx < 0
    colors[nosee] = 128
    return colors


def _invert_w2c_batch(w2c: np.ndarray) -> np.ndarray:
    """(N, 3, 4) world→cam → (N, 3, 4) cam→world."""
    R = w2c[:, :3, :3]
    t = w2c[:, :3, 3:]
    Rt = np.transpose(R, (0, 2, 1))
    tinv = -np.matmul(Rt, t)
    out = np.zeros_like(w2c)
    out[:, :3, :3] = Rt
    out[:, :3, 3:] = tinv
    return out

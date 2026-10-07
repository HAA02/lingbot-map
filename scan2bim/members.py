"""Separate pipes and columns from a point cloud and measure them.

Input is XYZ only: a scan cloud, or the means of a Gaussian splat PLY.
Splat scale and rotation are not used. Diameter is the radius of the points
around the fitted axis. A full-around cloud is what the checks use; a scan
that sees only one side of a pipe pulls the axis toward that side.

Members stay in the cloud's own frame. This does not align anything to a BIM.
"""
from __future__ import annotations

import numpy as np

from scan2bim.ply_xyz import read_ply_xyz


def extract_members_from_ply(path: str, **kwargs) -> dict:
    return extract_members(read_ply_xyz(path), **kwargs)


def extract_members(
    points,
    *,
    up_axis: int = 2,
    min_length_m: float = 0.4,
    max_members: int = 8,
    seed: int = 0,
    connect_gap_m: float = 0.15,
) -> dict:
    """Label columns and pipes, with diameter, length, endpoints, and joints.

    up_axis is the vertical index (2 = Z-up, 1 = Y-up).
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be (N, 3)")
    up = np.zeros(3, dtype=np.float64)
    up[int(up_axis)] = 1.0
    members: list[dict] = []
    remaining = pts
    for member, keep in _columns(remaining, up, min_length_m):
        member["id"] = f"m{len(members)}"
        members.append(member)
        remaining = remaining[~keep]
    rng = np.random.default_rng(seed)
    for _ in range(max(0, int(max_members) - len(members))):
        if len(remaining) < 40:
            break
        fit = _ransac_pipe(remaining, rng, up, min_length_m=min_length_m)
        if fit is None:
            break
        member, keep = fit
        member["id"] = f"m{len(members)}"
        members.append(member)
        remaining = remaining[~keep]
    members = _merge_collinear(members)
    return {
        "up_axis": int(up_axis),
        "members": members,
        "connections": _connect(members, connect_gap_m),
    }


def _plane_basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tmp = np.array([1.0, 0.0, 0.0]) if abs(float(up[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    b1 = np.cross(up, tmp)
    b1 /= np.linalg.norm(b1) + 1e-12
    b2 = np.cross(up, b1)
    return b1, b2


def _columns(pts: np.ndarray, up: np.ndarray, min_length_m: float):
    """Vertical shells: tall Z span and a round footprint in the horizontal plane."""
    if len(pts) < 40:
        return
    along = pts @ up
    b1, b2 = _plane_basis(up)
    uv = np.column_stack([pts @ b1, pts @ b2])
    cell = 0.08
    keys = np.floor(uv / cell).astype(np.int64)
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, key in enumerate(map(tuple, keys)):
        buckets.setdefault(key, []).append(i)
    tall = []
    for key, idxs in buckets.items():
        if len(idxs) < 25:
            continue
        span = float(along[idxs].max() - along[idxs].min())
        if span >= min_length_m:
            tall.append(key)
    if not tall:
        return
    tall_set = set(tall)
    seen: set[tuple[int, int]] = set()
    for key in tall:
        if key in seen:
            continue
        stack = [key]
        seen.add(key)
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            cx, cy = cur
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nxt = (cx + dx, cy + dy)
                    if nxt in tall_set and nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
        idxs = [i for k in comp for i in buckets[k]]
        if len(idxs) < 40:
            continue
        footprint = uv[idxs]
        center2 = footprint.mean(axis=0)
        radial = np.linalg.norm(footprint - center2, axis=1)
        radius = float(np.median(radial))
        if not (0.05 <= radius <= 1.0):
            continue
        cov = np.cov((footprint - center2).T)
        w = np.sort(np.clip(np.linalg.eigvalsh(cov), 0.0, None))
        if w[-2] <= 1e-8 or float(w[-1] / w[-2]) > 2.5:
            continue  # a wall is a long footprint, not a round shaft
        keep = np.zeros(len(pts), dtype=bool)
        keep[idxs] = True
        center3 = pts[idxs].mean(axis=0)
        t = (pts[idxs] - center3) @ up
        # Axis through the horizontal center of the shell, not a surface point.
        axis_point = (center2[0] * b1 + center2[1] * b2) + up * float(center3 @ up)
        start = axis_point + up * float(t.min())
        end = axis_point + up * float(t.max())
        yield _member("column", radius, start, end, up, int(keep.sum())), keep


def _ransac_pipe(pts, rng, up, *, min_length_m: float):
    n = len(pts)
    best = None
    for _ in range(50):
        seed = pts[int(rng.integers(0, n))]
        nb = pts[np.linalg.norm(pts - seed, axis=1) < 0.12]
        if len(nb) < 20:
            continue
        mu = nb.mean(axis=0)
        _u, s, vt = np.linalg.svd(nb - mu, full_matrices=False)
        if float(s[0]) < 2.0 * float(s[1]):
            continue
        direction = vt[0]
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        if abs(float(direction @ up)) >= np.cos(np.deg2rad(30.0)):
            continue
        radial_nb = _radial(nb, mu, direction)
        radius = float(np.median(radial_nb))
        if not (0.015 <= radius <= 0.5):
            continue
        if float(np.std(radial_nb)) > 0.012:
            continue
        radial = _radial(pts, mu, direction)
        inl = np.abs(radial - radius) < 0.012
        if int(inl.sum()) < 80:
            continue
        if not _round_enough(pts[inl], mu, direction):
            continue
        span = (pts[inl] - mu) @ direction
        length = float(span.max() - span.min())
        if length < min_length_m:
            continue
        score = int(inl.sum())
        if best is None or score > best[0]:
            best = (score, direction, mu, inl)
    if best is None:
        return None
    _score, direction, _mu, inl = best
    center = pts[inl].mean(axis=0)
    _u, s, vt = np.linalg.svd(pts[inl] - center, full_matrices=False)
    direction = vt[0]
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    if direction @ up < 0:
        direction = -direction
    along = (pts[inl] - center) @ direction
    radius = float(np.median(_radial(pts[inl], center, direction)))
    start = center + direction * float(along.min())
    end = center + direction * float(along.max())
    return _member("pipe", radius, start, end, direction, int(inl.sum())), inl


def _member(kind, radius, start, end, axis, inliers) -> dict:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    return {
        "kind": kind,
        "diameter_m": round(2.0 * float(radius), 4),
        "radius_m": round(float(radius), 4),
        "length_m": round(float(np.linalg.norm(end - start)), 4),
        "start": [round(float(x), 4) for x in start],
        "end": [round(float(x), 4) for x in end],
        "axis": [round(float(x), 4) for x in axis],
        "inliers": int(inliers),
    }


def _radial(pts, center, direction) -> np.ndarray:
    rel = pts - center
    along = rel @ direction
    perp = rel - np.outer(along, direction)
    return np.linalg.norm(perp, axis=1)


def _round_enough(pts, center, direction) -> bool:
    rel = pts - center
    along = rel @ direction
    perp = rel - np.outer(along, direction)
    cov = perp.T @ perp
    w = np.sort(np.clip(np.linalg.eigvalsh(cov), 0.0, None))
    if w[-2] <= 1e-10:
        return False
    return float(w[-1] / w[-2]) <= 6.0


def _merge_collinear(members: list[dict]) -> list[dict]:
    """Join fragments of one straight member that RANSAC split apart."""
    items = [dict(m) for m in members]
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(items):
            for j in range(i + 1, len(items)):
                b = items[j]
                if not _same_shaft(a, b):
                    continue
                items[i] = _union_shaft(a, b)
                del items[j]
                changed = True
                break
            if changed:
                break
    for i, m in enumerate(items):
        m["id"] = f"m{i}"
    return items


def _same_shaft(a: dict, b: dict) -> bool:
    if a["kind"] != b["kind"]:
        return False
    aa = np.asarray(a["axis"], dtype=np.float64)
    ba = np.asarray(b["axis"], dtype=np.float64)
    if abs(float(aa @ ba)) < 0.98:
        return False
    ra, rb = float(a["radius_m"]), float(b["radius_m"])
    if abs(ra - rb) > 0.25 * max(ra, rb, 1e-6):
        return False
    mid = 0.5 * (np.asarray(a["start"]) + np.asarray(a["end"]))
    if _point_line_dist(mid, np.asarray(b["start"]), ba) > 0.04:
        return False
    a0, a1 = sorted(float(np.asarray(p) @ aa) for p in (a["start"], a["end"]))
    b0, b1 = sorted(float(np.asarray(p) @ aa) for p in (b["start"], b["end"]))
    gap = max(a0, b0) - min(a1, b1)
    return gap < 0.3


def _union_shaft(a: dict, b: dict) -> dict:
    axis = np.asarray(a["axis"], dtype=np.float64)
    pts = [np.asarray(p, dtype=np.float64) for p in (a["start"], a["end"], b["start"], b["end"])]
    proj = [float(p @ axis) for p in pts]
    start = pts[int(np.argmin(proj))]
    end = pts[int(np.argmax(proj))]
    wa, wb = int(a["inliers"]), int(b["inliers"])
    radius = (float(a["radius_m"]) * wa + float(b["radius_m"]) * wb) / max(wa + wb, 1)
    out = _member(a["kind"], radius, start, end, axis, wa + wb)
    out["id"] = a["id"]
    return out


def _point_line_dist(point, origin, direction) -> float:
    rel = point - origin
    along = float(rel @ direction)
    return float(np.linalg.norm(rel - direction * along))


def _connect(members: list[dict], gap: float) -> list[dict]:
    edges = []
    for i, a in enumerate(members):
        ae = (np.asarray(a["start"], dtype=np.float64), np.asarray(a["end"], dtype=np.float64))
        ra = float(a["radius_m"])
        for b in members[i + 1:]:
            be = (np.asarray(b["start"], dtype=np.float64), np.asarray(b["end"], dtype=np.float64))
            rb = float(b["radius_m"])
            dist = min(float(np.linalg.norm(p - q)) for p in ae for q in be)
            if dist <= max(gap, ra + rb):
                edges.append({"a": a["id"], "b": b["id"], "gap_m": round(dist, 4)})
    return edges

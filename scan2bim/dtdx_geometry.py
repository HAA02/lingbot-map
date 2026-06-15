"""FR-1.1b .dtdx geometry decode + instance placement → world-space triangle soup.

Binary layout (verified): mesh.vp/tp are ABSOLUTE file offsets. At vp:
vc*3 little-endian float32 vertices (local frame). At tp: materialCount
int32 submesh-index-counts, then each submesh's int32 index array. Each
linkMesh places a copy of its base mesh (lid) at its own pivot/rotation/scale.

Output: per design-material-color triangle soup (non-indexed, world space),
ready for a single THREE.BufferGeometry per color in the web viewer.
"""
from __future__ import annotations

import struct
from pathlib import Path

import msgpack
import numpy as np

from scan2bim.dtdx import discipline_of

try:
    from scipy.spatial.transform import Rotation
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


def _read_msgpack(path):
    b = Path(path).read_bytes()
    jlen = struct.unpack_from("<i", b, 10)[0]
    obj = msgpack.unpackb(b[14:14 + jlen], raw=False, strict_map_key=False)
    return b, obj


def _transform(verts: np.ndarray, pivot, rot, scale) -> np.ndarray:
    v = verts
    if scale is not None and len(scale) == 3 and tuple(scale) != (1.0, 1.0, 1.0):
        v = v * np.asarray(scale, dtype=np.float64)
    if rot is not None and len(rot) == 3 and any(float(r) for r in rot) and _HAVE_SCIPY:
        R = Rotation.from_euler("XYZ", [float(r) for r in rot], degrees=True).as_matrix()
        v = v @ R.T
    if pivot is not None and len(pivot) == 3:
        v = v + np.asarray(pivot, dtype=np.float64)
    return v


def decode_geometry(path: str | Path) -> dict:
    b, o = _read_msgpack(path)
    mats = o.get("material") or []
    matcol = [tuple(float(x) for x in (m.get("d") or [0.5, 0.5, 0.5, 1.0])[:3]) for m in mats]

    # base meshes keyed by raw id: (local_verts, [(material_index, index_array)])
    bases: dict = {}
    base_place: list = []
    for mesh in o.get("mesh") or []:
        vc = mesh.get("vc") or 0
        vp, tp = mesh.get("vp"), mesh.get("tp")
        marr = mesh.get("m") or []
        if not vc or vp is None:
            continue
        verts = np.frombuffer(b, dtype="<f4", count=vc * 3, offset=vp).reshape(-1, 3).astype(np.float64)
        subs = []
        if marr and tp is not None:
            mc = len(marr)
            counts = np.frombuffer(b, dtype="<i4", count=mc, offset=tp)
            off = tp + mc * 4
            for si in range(mc):
                cnt = int(counts[si])
                idx = np.frombuffer(b, dtype="<i4", count=cnt, offset=off).astype(np.int64)
                off += cnt * 4
                subs.append((marr[si], idx))
        bases[mesh.get("id")] = (verts, subs)
        base_place.append((mesh.get("id"), mesh.get("p"), mesh.get("ro"), mesh.get("sc")))

    groups: dict = {}  # color -> list of (M,3) float32 triangle-soup arrays

    def emit(base_id, pivot, rot, scale):
        bg = bases.get(base_id)
        if not bg:
            return
        verts, subs = bg
        for matidx, idx in subs:
            if idx.size == 0:
                continue
            tri = _transform(verts[idx], pivot, rot, scale).astype(np.float32)
            col = matcol[matidx] if 0 <= matidx < len(matcol) else (0.5, 0.5, 0.5)
            groups.setdefault(col, []).append(tri)

    for bid, p, ro, sc in base_place:
        emit(bid, p, ro, sc)
    for lm in o.get("linkMesh") or []:
        emit(lm.get("lid"), lm.get("p"), lm.get("ro"), lm.get("s"))

    meshes = [{"color": c, "positions": (np.concatenate(v) if v else np.zeros((0, 3), np.float32))}
              for c, v in groups.items()]
    allp = [m["positions"] for m in meshes if len(m["positions"])]
    if allp:
        stacked = np.concatenate(allp)
        bbox = (stacked.min(0).tolist(), stacked.max(0).tolist())
    else:
        bbox = None
    code, disc = discipline_of(Path(path).name)
    return {
        "discipline": disc, "discipline_code": code,
        "meshes": meshes, "bbox": bbox,
        "triangle_count": sum(len(m["positions"]) for m in meshes) // 3,
    }

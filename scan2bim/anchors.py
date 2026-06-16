"""FR-2.3 object anchors — distinctive (non-repetitive) model objects extracted
per discipline as 3D instance centers + footprint, for matching against the
OWL-ViT detections (scan2bim.detect) to disambiguate location/yaw where repeated
geometry (pipes) cannot.

Each linkMesh / base placement = one object instance. Its world center and
bounding size let us classify it (ceiling AC, light, column, duct, pipe) and,
once a camera pose is known, match the detected anchor type+direction to the
model's same-type objects. AC cassettes (HXX) and ceiling lights (EXX) are the
strongest anchors: OWL-ViT detects them at 0.5-0.8 on real frames.

X is mirrored (flip_x=True) by default to match the co-play / authoring-viewer
frame — see [[dtdx-handedness-flip]] (Babylon LH -> Three RH).
"""
from __future__ import annotations

import struct
from pathlib import Path

import msgpack
import numpy as np

from scan2bim.dtdx import discipline_of
from scan2bim.dtdx_geometry import _transform

# anchor type vocabulary aligned with detect.DEFAULT_PROMPTS
AC = "ac"
LIGHT = "light"
COLUMN = "column"
DUCT = "duct"
PIPE = "pipe"
OTHER = "other"


def _read(path):
    b = Path(path).read_bytes()
    jlen = struct.unpack_from("<i", b, 10)[0]
    return b, msgpack.unpackb(b[14:14 + jlen], raw=False, strict_map_key=False)


def _base_geom(b, o):
    """base_id -> (centroid(3,), size(3,)) in the mesh-local frame."""
    out = {}
    for mesh in o.get("mesh") or []:
        vc, vp = mesh.get("vc") or 0, mesh.get("vp")
        if not vc or vp is None:
            continue
        v = np.frombuffer(b, dtype="<f4", count=vc * 3, offset=vp).reshape(-1, 3).astype(np.float64)
        out[mesh.get("id")] = (v.mean(0), v.max(0) - v.min(0))
    return out


def model_anchors(path: str | Path, *, flip_x: bool = True) -> list[dict]:
    """Per-object-instance anchors: [{discipline, id, center[x,y,z], size[x,y,z]}].

    Covers both base-mesh placements and linkMesh instances. center is the world
    centroid (placement applied to the local centroid); size is the local bbox
    extent scaled by the instance scale (rotation ignored for the size proxy).
    """
    b, o = _read(path)
    code, disc = discipline_of(Path(path).name)
    bases = _base_geom(b, o)
    anchors: list[dict] = []
    sx = -1.0 if flip_x else 1.0

    def emit(inst_id, base_id, pivot, rot, scale):
        bg = bases.get(base_id)
        if bg is None:
            return
        centroid, size = bg
        c = _transform(centroid.reshape(1, 3), pivot, rot, scale)[0]
        c = c.astype(float)
        c[0] *= sx
        s = np.asarray(scale, float) if scale and len(scale) == 3 else np.ones(3)
        anchors.append({
            "discipline": code, "id": inst_id,
            "center": [round(float(x), 3) for x in c],
            "size": [round(float(abs(v) * abs(sv)), 3) for v, sv in zip(size, s)],
        })

    for mesh in o.get("mesh") or []:
        emit(mesh.get("id"), mesh.get("id"), mesh.get("p"), mesh.get("ro"), mesh.get("sc"))
    for lm in o.get("linkMesh") or []:
        emit(lm.get("id"), lm.get("lid"), lm.get("p"), lm.get("ro"), lm.get("sc"))
    return anchors


def classify_anchor(disc_code: str, size, center, ceiling_y: float) -> str:
    """Geometry heuristic for the distinctive anchor type.

    disc_code drives the prior (HXX->AC/duct, EXX->light, SXX->column, FXX->pipe);
    size refines it (AC = wide+flat near ceiling, column = tall, light = small).
    """
    dx, dy, dz = (float(s) for s in (list(size) + [0, 0, 0])[:3])
    foot = max(dx, dz)
    d = (disc_code or "")[:1].upper()
    if d == "S":  # structure
        return COLUMN if dy >= max(dx, dz) * 1.5 and dy > 1.0 else OTHER
    if d == "H":  # hvac ceiling fixtures (cassettes/diffusers/vents) — flat MEP on ceiling
        if foot >= 0.2 and dy <= max(foot, 0.5):   # capture all flat fixtures (OWL-ViT sees these as AC)
            return AC
        return DUCT if foot >= 1.0 else OTHER
    if d == "E":  # electrical: ceiling light
        return LIGHT if foot <= 1.6 else OTHER
    if d == "F":  # fire
        return PIPE
    return OTHER


def anchor_inventory(paths, *, flip_x: bool = True) -> dict:
    """Typed anchor catalog across disciplines: type -> [anchor, ...].

    ceiling_y is taken as the global max anchor Y (the ceiling band) for the
    AC/light height test.
    """
    alla = []
    for p in paths:
        alla.extend(model_anchors(p, flip_x=flip_x))
    if not alla:
        return {}
    ceiling_y = max(a["center"][1] for a in alla)
    cat: dict = {}
    for a in alla:
        t = classify_anchor(a["discipline"], a["size"], a["center"], ceiling_y)
        a = {**a, "type": t}
        cat.setdefault(t, []).append(a)
    return cat

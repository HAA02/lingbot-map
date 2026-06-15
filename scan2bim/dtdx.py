"""FR-1.1 .dtdx ingestion — parse the DTDWeb model format into an object model.

`.dtdx` layout (version 0x00200000):
    signature(uint16 LE, 0xFF09) + version(int32) + formatType(int32) +
    jsonLength(int32) + [msgpack header] + [binary geometry blob]

The msgpack header carries: material (diffuse RGB — system color-coded),
mesh (base geometry meta), linkMesh (element instances), connector (topology),
attr (element-id → IfcGUID), header/revitInfo.

This module reads the metadata layer (color + topology + identity + inventory).
Binary vertex extraction (vp/tp offsets) is deferred to FR-1.1b.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path

import msgpack

# discipline code (first letter) → name. Files: A/S/E/F/H/P XX
_DISCIPLINE = {
    "A": "architecture",
    "S": "structure",
    "E": "electrical",
    "F": "fire",
    "H": "hvac",
    "P": "plumbing",
}
# IfcGUID: 22-char base64 over [0-9A-Za-z_$]
_GUID_RE = re.compile(r"^[0-9A-Za-z_$]{20,24}$")

_SIGNATURE = 0xFF09
_VERSION_0X00200000 = 0x00200000
_HEADER_BYTES = 14  # 2 + 4 + 4 + 4


def discipline_of(filename: str) -> tuple[str, str]:
    """Return (code, discipline_name) parsed from a Gasan-style filename.

    e.g. 'G7F_FAB_FXX_7F-0_Central_1.dtdx' -> ('FXX', 'fire')
    """
    parts = Path(filename).name.split("_")
    code = ""
    for i, p in enumerate(parts):
        if p == "FAB" and i + 1 < len(parts):
            code = parts[i + 1]
            break
    if not code and len(parts) > 2:
        code = parts[2]
    return code, _DISCIPLINE.get(code[:1].upper(), "unknown")


@dataclass
class Material:
    id: object
    diffuse_rgb: tuple  # (r, g, b) in 0..1
    alpha: float


@dataclass
class Element:
    """A placed element instance (linkMesh) referencing a base mesh."""
    instance_id: object
    base_mesh_id: object
    pivot: tuple
    scale: tuple


@dataclass
class DtdxModel:
    path: str
    filename: str
    signature: int
    version: int
    format_type: int
    discipline_code: str
    discipline: str
    materials: list
    meshes: list
    elements: list
    connectors: list
    attr: dict
    guids: list


def _get(o, k, default=None):
    return o.get(k, default) if isinstance(o, dict) else default


def _collect_guids(attr: dict) -> list:
    """Walk attr values and collect IfcGUID-like strings (bare or nested)."""
    out: list = []

    def walk(v):
        if isinstance(v, str):
            if _GUID_RE.match(v):
                out.append(v)
        elif isinstance(v, dict):
            for vv in v.values():
                walk(vv)
        elif isinstance(v, (list, tuple)):
            for vv in v:
                walk(vv)

    for v in attr.values():
        walk(v)
    return out


def load_dtdx(path: str | Path) -> DtdxModel:
    p = Path(path)
    b = p.read_bytes()
    sig = struct.unpack_from("<H", b, 0)[0]
    ver = struct.unpack_from("<i", b, 2)[0]
    ftype = struct.unpack_from("<i", b, 6)[0]
    jlen = struct.unpack_from("<i", b, 10)[0]
    if sig != _SIGNATURE:
        raise ValueError(f"{p.name}: bad signature 0x{sig:04X} (expected 0x{_SIGNATURE:04X})")
    if ver != _VERSION_0X00200000:
        raise ValueError(f"{p.name}: unsupported version 0x{ver:08X}")

    obj = msgpack.unpackb(b[_HEADER_BYTES:_HEADER_BYTES + jlen], raw=False, strict_map_key=False)

    materials = []
    for m in _get(obj, "material") or []:
        d = _get(m, "d") or [0.0, 0.0, 0.0, 1.0]
        materials.append(Material(
            id=_get(m, "id"),
            diffuse_rgb=tuple(float(x) for x in d[:3]),
            alpha=float(d[3]) if len(d) > 3 else 1.0,
        ))

    meshes = list(_get(obj, "mesh") or [])

    elements = []
    for lm in _get(obj, "linkMesh") or []:
        elements.append(Element(
            instance_id=_get(lm, "id"),
            base_mesh_id=_get(lm, "lid"),
            pivot=tuple(_get(lm, "p") or ()),
            scale=tuple(_get(lm, "s") or ()),
        ))

    connectors = list(_get(obj, "connector") or [])
    attr = _get(obj, "attr") or {}
    if not isinstance(attr, dict):
        attr = {}

    code, disc = discipline_of(p.name)
    return DtdxModel(
        path=str(p), filename=p.name, signature=sig, version=ver, format_type=ftype,
        discipline_code=code, discipline=disc,
        materials=materials, meshes=meshes, elements=elements,
        connectors=connectors, attr=attr, guids=_collect_guids(attr),
    )

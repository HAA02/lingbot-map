"""Read vertex x,y,z from a PLY. Extra properties (Gaussian scale, opacity, color) are skipped.

This is the means of a 3DGS PLY or the points of an ordinary cloud. It does not
load splat covariances.
"""
from __future__ import annotations

import numpy as np

_SIZE = {
    "char": 1, "uchar": 1, "int8": 1, "uint8": 1,
    "short": 2, "ushort": 2, "int16": 2, "uint16": 2,
    "int": 4, "uint": 4, "int32": 4, "uint32": 4,
    "float": 4, "float32": 4,
    "double": 8, "float64": 8,
}


def read_ply_xyz(path: str) -> np.ndarray:
    """Return (N, 3) float64 positions from element vertex properties x, y, z."""
    with open(path, "rb") as f:
        header, fmt, count, props = _header(f)
        if fmt == "ascii":
            text = f.read().decode("utf-8", "replace")
        else:
            blob = f.read(count * _stride(props))
    if count == 0:
        return np.zeros((0, 3), dtype=np.float64)
    ix = _index(props, "x")
    iy = _index(props, "y")
    iz = _index(props, "z")
    if fmt == "ascii":
        rows = [ln.split() for ln in text.splitlines() if ln.strip()]
        if len(rows) < count:
            raise ValueError(f"PLY vertex rows {len(rows)} < header count {count}")
        out = np.zeros((count, 3), dtype=np.float64)
        for i in range(count):
            out[i, 0] = float(rows[i][ix])
            out[i, 1] = float(rows[i][iy])
            out[i, 2] = float(rows[i][iz])
        return out
    if fmt != "binary_little_endian":
        raise ValueError(f"unsupported PLY format {fmt}")
    return _binary_xyz(blob, count, props, ix, iy, iz)


def _header(f):
    lines = []
    while True:
        raw = f.readline()
        if not raw:
            raise ValueError("PLY header without end_header")
        line = raw.decode("utf-8", "replace").strip()
        lines.append(line)
        if line == "end_header":
            break
    if not lines or lines[0] != "ply":
        raise ValueError("not a PLY file")
    fmt = None
    count = None
    props: list[tuple[str, str]] = []
    in_vertex = False
    for line in lines[1:]:
        parts = line.split()
        if not parts or parts[0] == "comment":
            continue
        if parts[0] == "format" and len(parts) >= 2:
            fmt = parts[1]
        elif parts[:2] == ["element", "vertex"] and len(parts) >= 3:
            count = int(parts[2])
            in_vertex = True
            props = []
        elif parts[0] == "element":
            in_vertex = False
        elif in_vertex and parts[0] == "property" and len(parts) >= 3:
            if parts[1] == "list":
                raise ValueError("list properties on vertex are not supported")
            props.append((parts[1], parts[2]))
    if fmt is None or count is None or not props:
        raise ValueError("PLY vertex element with x,y,z is required")
    return lines, fmt, count, props


def _index(props, name: str) -> int:
    for i, (_typ, prop) in enumerate(props):
        if prop == name:
            return i
    raise ValueError(f"PLY vertex has no property {name}")


def _stride(props) -> int:
    n = 0
    for typ, _name in props:
        if typ not in _SIZE:
            raise ValueError(f"unsupported PLY type {typ}")
        n += _SIZE[typ]
    return n


def _binary_xyz(blob: bytes, count: int, props, ix: int, iy: int, iz: int) -> np.ndarray:
    stride = _stride(props)
    if len(blob) < count * stride:
        raise ValueError("PLY binary vertex block is short")
    offsets = []
    off = 0
    for typ, _name in props:
        offsets.append(off)
        off += _SIZE[typ]
    want = (ix, iy, iz)
    out = np.zeros((count, 3), dtype=np.float64)
    for i in range(count):
        base = i * stride
        for k, pi in enumerate(want):
            typ = props[pi][0]
            start = base + offsets[pi]
            raw = blob[start:start + _SIZE[typ]]
            if typ in ("double", "float64"):
                out[i, k] = np.frombuffer(raw, dtype="<f8", count=1)[0]
            elif typ in ("float", "float32"):
                out[i, k] = np.frombuffer(raw, dtype="<f4", count=1)[0]
            else:
                raise ValueError(f"x,y,z must be float, got {typ}")
    return out

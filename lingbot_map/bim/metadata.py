"""BIM metadata loader.

Input JSON shape (minimum viable):
{
  "model_id": "...",
  "coordinate_system": "bim_world",
  "objects": [
    {
      "guid": "abc-123",
      "category": "Pipe",
      "system": "CHW_Supply",
      "zone": "B1-EAST",
      "center": [x, y, z],
      "start": [x, y, z],    // optional, for linear elements
      "end":   [x, y, z],    // optional
      "bbox":  [[xmin,ymin,zmin],[xmax,ymax,zmax]],  // optional
      "connectors": ["guid-of-neighbor", ...]
    }
  ]
}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BimObject:
    guid: str
    category: str = "Unknown"
    system: str | None = None
    zone: str | None = None
    center: np.ndarray | None = None
    start: np.ndarray | None = None
    end: np.ndarray | None = None
    bbox: np.ndarray | None = None  # (2,3) min/max
    connectors: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def representative_point(self) -> np.ndarray:
        if self.center is not None:
            return self.center
        if self.start is not None and self.end is not None:
            return 0.5 * (self.start + self.end)
        if self.bbox is not None:
            return 0.5 * (self.bbox[0] + self.bbox[1])
        raise ValueError(f"BimObject {self.guid} has no geometry")

    def sample_points(self, n: int = 8) -> np.ndarray:
        """Return a small set of representative 3D points for projection coverage."""
        pts = [self.representative_point()]
        if self.start is not None and self.end is not None:
            ts = np.linspace(0, 1, max(2, n))
            pts = [(1 - t) * self.start + t * self.end for t in ts]
        elif self.bbox is not None:
            mn, mx = self.bbox[0], self.bbox[1]
            corners = np.array([
                [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
                [mn[0], mx[1], mn[2]], [mx[0], mx[1], mn[2]],
                [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
                [mn[0], mx[1], mx[2]], [mx[0], mx[1], mx[2]],
            ])
            pts = list(corners)
        return np.asarray(pts, dtype=np.float64)


@dataclass
class BimModel:
    model_id: str
    coordinate_system: str
    objects: dict[str, BimObject]

    def __iter__(self):
        return iter(self.objects.values())

    def __len__(self):
        return len(self.objects)

    def neighbors(self, guid: str) -> list[str]:
        obj = self.objects.get(guid)
        return list(obj.connectors) if obj else []


def _as_np(v) -> np.ndarray | None:
    if v is None:
        return None
    arr = np.asarray(v, dtype=np.float64)
    return arr


def load_bim_metadata(path: str) -> BimModel:
    with open(path) as f:
        raw = json.load(f)

    objects: dict[str, BimObject] = {}
    for o in raw.get("objects", []):
        guid = o["guid"]
        bbox = o.get("bbox")
        bbox_arr = np.asarray(bbox, dtype=np.float64) if bbox else None
        obj = BimObject(
            guid=guid,
            category=o.get("category", "Unknown"),
            system=o.get("system"),
            zone=o.get("zone"),
            center=_as_np(o.get("center")),
            start=_as_np(o.get("start")),
            end=_as_np(o.get("end")),
            bbox=bbox_arr,
            connectors=list(o.get("connectors", [])),
            extra={k: v for k, v in o.items() if k not in {
                "guid", "category", "system", "zone",
                "center", "start", "end", "bbox", "connectors"
            }},
        )
        objects[guid] = obj

    return BimModel(
        model_id=raw.get("model_id", "unknown"),
        coordinate_system=raw.get("coordinate_system", "bim_world"),
        objects=objects,
    )

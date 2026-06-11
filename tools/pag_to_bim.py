"""Adapter: Revit/PAG export JSON -> lingbot BimMetadata JSON.

PAG categories handled (others are skipped):
  - structuralColumns, columns, genericModels, specialtyEquipment,
    mechanicalEquipment, electricalEquipment, plumbingFixtures, sprinklers,
    doors, windows                              -> use `center` + size as bbox
  - structuralFraming, walls                    -> use `start`/`end` (linear)
  - pipes, ducts, conduits, cableTrays          -> use `start`/`end` (linear) when present,
                                                   else `center`+`length`
  - elbows, tees, valves, reducers              -> use `center`

`system` is derived from `systems[*]` lookup when the object has a `systemName`
field; otherwise we fall back to the PAG top-level category. `zone` is set to
the `level` string (Korean level names like "레벨 7" are kept verbatim).

`connectors` is set from a top-level `connections` list if present; the gasan
export ships `connections: []`, so topology inference will be a no-op.

Usage:
    python tools/pag_to_bim.py models/pag_export.json out/bim_metadata.json
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any


# (pag_key, our_category) — order matters only for logging.
LINEAR_KEYS = [
    ("structuralFraming", "Beam"),
    ("walls", "Wall"),
    ("pipes", "Pipe"),
    ("ducts", "Duct"),
    ("conduits", "Conduit"),
    ("cableTrays", "CableTray"),
    ("flexDucts", "FlexDuct"),
    ("flexPipes", "FlexPipe"),
    ("wires", "Wire"),
]

POINT_KEYS = [
    ("structuralColumns", "Column"),
    ("columns", "Column"),
    ("genericModels", "GenericModel"),
    ("specialtyEquipment", "SpecialtyEquipment"),
    ("mechanicalEquipment", "MechanicalEquipment"),
    ("electricalEquipment", "ElectricalEquipment"),
    ("plumbingFixtures", "PlumbingFixture"),
    ("sprinklers", "Sprinkler"),
    ("doors", "Door"),
    ("windows", "Window"),
    ("elbows", "Elbow"),
    ("tees", "Tee"),
    ("valves", "Valve"),
    ("reducers", "Reducer"),
    ("lightingFixtures", "LightingFixture"),
    ("lightingDevices", "LightingDevice"),
    ("communicationDevices", "CommunicationDevice"),
    ("dataDevices", "DataDevice"),
    ("fireAlarmDevices", "FireAlarmDevice"),
    ("securityDevices", "SecurityDevice"),
    ("ductFittings", "DuctFitting"),
    ("ductAccessories", "DuctAccessory"),
    ("mepAccessories", "MepAccessory"),
    ("cableTrayFittings", "CableTrayFitting"),
    ("conduitFittings", "ConduitFitting"),
]


def _bbox_from_center(center: list[float], w: float, h: float, l: float) -> list[list[float]]:
    cx, cy, cz = center
    dx, dy, dz = abs(w) / 2 or 1.0, abs(l) / 2 or 1.0, abs(h) / 2 or 1.0
    return [[cx - dx, cy - dy, cz - dz], [cx + dx, cy + dy, cz + dz]]


def convert(pag: dict[str, Any]) -> dict[str, Any]:
    objects: list[dict[str, Any]] = []
    project_name = pag.get("project", {}).get("name", "unknown")
    counts: dict[str, int] = {}

    # connections list (if any) -> guid -> [neighbor_guids]
    connectors_map: dict[str, list[str]] = {}
    for conn in pag.get("connections", []) or []:
        a = conn.get("guidA") or conn.get("from")
        b = conn.get("guidB") or conn.get("to")
        if a and b:
            connectors_map.setdefault(a, []).append(b)
            connectors_map.setdefault(b, []).append(a)

    def _push(guid: str, category: str, system: str | None, zone: str | None,
              center=None, start=None, end=None, bbox=None, extra=None):
        rec: dict[str, Any] = {
            "guid": guid,
            "category": category,
            "system": system,
            "zone": zone,
        }
        if center is not None:
            rec["center"] = list(center)
        if start is not None and end is not None:
            rec["start"] = list(start)
            rec["end"] = list(end)
        if bbox is not None:
            rec["bbox"] = bbox
        rec["connectors"] = connectors_map.get(guid, [])
        if extra:
            rec["extra"] = extra
        objects.append(rec)
        counts[category] = counts.get(category, 0) + 1

    # linear elements
    for key, cat in LINEAR_KEYS:
        items = pag.get(key, []) or []
        for o in items:
            guid = o.get("guid")
            if not guid:
                continue
            start = o.get("start")
            end = o.get("end")
            center = o.get("center")
            zone = o.get("level")
            system = o.get("systemName") or o.get("system") or key
            if start and end:
                _push(guid, cat, system, zone, start=start, end=end,
                      center=center or [(s + e) / 2 for s, e in zip(start, end)])
            elif center is not None:
                length = o.get("length") or 0.0
                bbox = _bbox_from_center(center, o.get("width", 0) or length,
                                         o.get("height", 0) or 1.0, length or 1.0)
                _push(guid, cat, system, zone, center=center, bbox=bbox)

    # point/box elements
    for key, cat in POINT_KEYS:
        items = pag.get(key, []) or []
        for o in items:
            guid = o.get("guid")
            center = o.get("center")
            if not guid or center is None:
                continue
            w = o.get("width", 0) or o.get("overallWidth", 0) or 100.0
            h = o.get("height", 0) or o.get("overallHeight", 0) or 100.0
            l = o.get("length", 0) or 100.0
            bbox = _bbox_from_center(center, w, h, l)
            zone = o.get("level")
            system = o.get("systemName") or o.get("system") or key
            _push(guid, cat, system, zone, center=center, bbox=bbox)

    return {
        "model_id": project_name,
        "coordinate_system": "bim_world",
        "units": pag.get("project", {}).get("units", "mm"),
        "objects": objects,
        "_counts_by_category": counts,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="PAG export JSON")
    ap.add_argument("output", help="lingbot BIM metadata JSON output path")
    args = ap.parse_args()

    with open(args.input) as f:
        pag = json.load(f)
    out = convert(pag)
    counts = out.pop("_counts_by_category")
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"[pag->bim] project={out['model_id']} units={out['units']}")
    print(f"[pag->bim] total objects: {len(out['objects'])}")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k:25s} {v}")
    print(f"[pag->bim] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""FR-2.2 color↔system matching — text-free semantic bridge between the scan
and the design.

Segment scan points by coarse hue (reusing the design color classifier) and map
each chromatic cluster to the design disciplines/systems that share that hue.
Achromatic (gray/white/black) points carry no system meaning and are excluded;
those regions fall back to geometry/diameter/topology (FR-2.3).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from scan2bim.colormap import classify_color


def segment_by_color(points, colors, *, min_count: int = 20) -> list:
    """Group scan points by coarse hue. Returns chromatic clusters only.

    Each cluster: {hue, count, centroid[3], indices}. Clusters smaller than
    min_count are dropped as noise.
    """
    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors)
    buckets: dict = defaultdict(list)
    for i, c in enumerate(colors):
        rgb = (float(c[0]) / 255.0, float(c[1]) / 255.0, float(c[2]) / 255.0)
        kind, hue = classify_color(rgb)
        if kind != "chromatic":
            continue
        buckets[hue].append(i)
    clusters = []
    for hue, idxs in buckets.items():
        if len(idxs) < min_count:
            continue
        pts = points[idxs]
        clusters.append({
            "hue": hue,
            "count": len(idxs),
            "centroid": pts.mean(0).tolist(),
            "indices": idxs,
        })
    return sorted(clusters, key=lambda c: -c["count"])


def match_to_design(clusters, color_index) -> list:
    """Map each scan color cluster to design entries sharing its hue.

    color_index = scan2bim.colormap.build_color_index(models). Returns, per
    cluster: {hue, count, matches:[{discipline, discipline_code, material_id}]}.
    """
    by_hue: dict = defaultdict(list)
    for e in color_index:
        if e.get("usable") and e.get("hue"):
            by_hue[e["hue"]].append(e)
    out = []
    for c in clusters:
        matches = [{"discipline": e["discipline"], "discipline_code": e["discipline_code"],
                    "material_id": e["material_id"]}
                   for e in by_hue.get(c["hue"], [])]
        out.append({"hue": c["hue"], "count": c["count"], "matches": matches})
    return out

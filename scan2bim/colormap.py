"""FR-1.2 design color map — classify design material colors and build an
index of which (discipline, color) groups are usable for color↔system matching.

A color is *achromatic* (gray/black/white) when its saturation is below a
threshold; such elements (e.g. HVAC's single gray) cannot be matched by color
and must fall back to geometry/diameter/topology. *Chromatic* colors get a
coarse hue name (red/blue/...) used as the semantic anchor against the scan.
"""
from __future__ import annotations

import colorsys

# saturation below this → achromatic (gray scale). Tuned so 0.5-gray is gray
# but saturated system colors (fire red, plumbing blue) are chromatic.
_SAT_MIN = 0.18
_VAL_MIN = 0.06  # near-black → achromatic

# hue degree → coarse name
_HUE_BANDS = [
    (15, "red"), (45, "orange"), (70, "yellow"), (170, "green"),
    (200, "cyan"), (260, "blue"), (320, "magenta"), (345, "pink"), (360, "red"),
]


def _hue_name(deg: float) -> str:
    for upper, name in _HUE_BANDS:
        if deg < upper:
            return name
    return "red"


def classify_color(rgb, sat_min: float = _SAT_MIN):
    """Return (kind, hue) where kind in {'chromatic','achromatic'}.

    hue is a coarse color name for chromatic colors, else None.
    """
    r, g, b = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if v < _VAL_MIN or s < sat_min:
        return ("achromatic", None)
    return ("chromatic", _hue_name(h * 360.0))


def build_color_index(models) -> list:
    """Per-material color index across one or more DtdxModels.

    Each entry: {discipline, discipline_code, material_id, rgb, alpha,
                 kind, hue, usable} where usable = chromatic.
    """
    out = []
    for m in models:
        for mat in m.materials:
            kind, hue = classify_color(mat.diffuse_rgb)
            out.append({
                "discipline": m.discipline,
                "discipline_code": m.discipline_code,
                "material_id": mat.id,
                "rgb": tuple(round(c, 3) for c in mat.diffuse_rgb),
                "alpha": mat.alpha,
                "kind": kind,
                "hue": hue,
                "usable": kind == "chromatic",
            })
    return out

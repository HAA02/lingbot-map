#!/usr/bin/env python3
"""Measure pipes and columns from a point cloud or a Gaussian PLY (xyz means only).

Writes JSON: members (kind, diameter_m, length_m, start, end) and connections.
Does not align the cloud to a BIM and does not confirm a placement.
"""
from __future__ import annotations

import argparse
import json

from scan2bim.members import extract_members_from_ply


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ply", help="PLY with vertex x,y,z. Extra Gaussian fields are ignored.")
    ap.add_argument("--up-axis", type=int, default=2, choices=(0, 1, 2),
                    help="vertical axis index. 2=Z-up (default), 1=Y-up")
    ap.add_argument("--out", default=None, help="JSON path. Default: stdout")
    args = ap.parse_args()
    doc = extract_members_from_ply(args.ply, up_axis=args.up_axis)
    text = json.dumps(doc, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()

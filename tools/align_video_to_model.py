"""Drive the AutoPM scan-to-model pipeline for an already-uploaded video:
auto-place candidates -> save best -> coverage analyze -> summarize WHICH part
of the model the footage matched.

Assumes realtime/server.py is running and the upload's scan is reconstructed.

Usage:
    .venv/bin/python tools/align_video_to_model.py \
        --base-url http://127.0.0.1:8767 --model pipe_duct \
        --upload upload_XXXX --out reports/pipe_duct_align
"""
from __future__ import annotations
import argparse, json, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path


def req(base, path, *, query=None, method="GET", payload=None, expect=None, timeout=300):
    url = base.rstrip("/") + path
    if query:
        url += "?" + urllib.parse.urlencode(query)
    data = None; headers = {}
    if payload is not None:
        data = json.dumps(payload).encode(); headers["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            body = resp.read(); status = resp.status
    except urllib.error.HTTPError as e:
        body = e.read(); status = e.code
    allowed = expect or {200}
    try:
        j = json.loads(body.decode("utf-8", "replace"))
    except Exception:
        j = {"_raw": body[:400].decode("utf-8", "replace")}
    if status not in allowed:
        print(f"  ! {method} {path} -> {status} (expected {sorted(allowed)}): {str(j)[:300]}")
    return status, j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8767")
    ap.add_argument("--model", default="pipe_duct")
    ap.add_argument("--upload", required=True)
    ap.add_argument("--out", default="reports/pipe_duct_align")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    base, model, up = args.base_url, args.model, args.upload
    summary = {"upload": up, "model": model}

    # 1) wait for scan
    print("== 1. scan status ==")
    for _ in range(120):
        _, st = req(base, f"/api/uploads/{up}/coverage/status", query={"model_id": model})
        scan = (st.get("scan") or {})
        if scan.get("ok"):
            print(f"  scan ready: {scan.get('magic')} pts={scan.get('point_count')} poses={scan.get('pose_count')} src_ids={scan.get('has_source_ids')}")
            summary["scan"] = {k: scan.get(k) for k in ("magic", "point_count", "pose_count", "has_source_ids")}
            summary["coordinate_before"] = (st.get("coordinate_status") or {}).get("state")
            break
        time.sleep(3)
    else:
        print("  scan never became ready"); return

    # 2) auto-place candidates (persisted)
    print("== 2. auto-place candidates ==")
    _, cand = req(base, f"/api/uploads/{up}/alignment/candidates", method="POST",
                  payload={"model_id": model, "auto_place": True, "max_candidates": 5, "dry_run": False},
                  expect={200})
    cands = cand.get("candidates") or []
    rep = cand.get("repetition_risk") or {}
    print(f"  status={cand.get('status')} repetition_risk={rep.get('level')} n_candidates={len(cands)}")
    for c in cands:
        a = c.get("alignment") or {}; co = c.get("corroboration") or {}
        print(f"   - {c.get('id')} src={c.get('source')} q={a.get('quality')} score={c.get('score')} "
              f"can_save={c.get('can_save')} med_nn={co.get('median_nn_m')} inlier={co.get('inlier_ratio')} "
              f"t={[round(x,2) for x in (a.get('translation') or [])]}")
    summary["repetition_risk"] = rep
    summary["candidates"] = cands
    autos = [c for c in cands if c.get("source") == "auto_geometric"]
    if not autos:
        print("  no auto candidates"); summary["result"] = "no_auto_candidate"
        (out / f"{up}_align_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)); return
    best = max(autos, key=lambda c: (c.get("can_save", False), c.get("score") or 0))
    summary["best_candidate_id"] = best.get("id")
    print(f"  best: {best.get('id')} (can_save={best.get('can_save')})")

    # 3) try to save best candidate
    print("== 3. save best candidate ==")
    s_status, saved = req(base, f"/api/uploads/{up}/alignment", method="POST",
                          payload={"model_id": model, "candidate_id": best.get("id")},
                          expect={200, 409})
    summary["save_status"] = s_status
    aligned = (s_status == 200)
    print(f"  save -> {s_status} {('OK' if aligned else saved.get('status'))}")

    # 4) coverage analyze (needs alignment saved)
    cov = None
    if aligned:
        print("== 4. coverage analyze ==")
        a_status, an = req(base, f"/api/uploads/{up}/coverage/analyze", method="POST",
                           payload={"model_id": model, "coverage_options": {}}, expect={200, 400, 409})
        if a_status == 200:
            _, cov = req(base, f"/api/uploads/{up}/coverage", query={"model_id": model}, expect={200, 409})
        else:
            print(f"  analyze blocked: {an.get('status')}")
            summary["analyze_status"] = an.get("status")
    else:
        print("== 4. coverage skipped (candidate not saveable — repeated-object guard) ==")

    # 5) summarize matched region
    print("== 5. matched region ==")
    if cov and cov.get("objects") is not None:
        objs = cov["objects"]; counts = cov.get("status_counts") or {}
        summary["status_counts"] = counts
        summary["alignment_used"] = cov.get("alignment")
        print(f"  status_counts: {counts}")
        matched = [o for o in objs if o.get("status") in ("observed", "likely_observed")]
        by_cat, by_sys, by_zone = {}, {}, {}
        for o in matched:
            by_cat[o.get("category")] = by_cat.get(o.get("category"), 0) + 1
            by_sys[o.get("system")] = by_sys.get(o.get("system"), 0) + 1
            by_zone[o.get("zone")] = by_zone.get(o.get("zone"), 0) + 1
        print(f"  matched (observed+likely)={len(matched)} by_category={by_cat}")
        print(f"   by_system={by_sys}")
        print(f"   by_zone={by_zone}")
        topo = sorted([o for o in objs if o.get("status") == "observed"],
                      key=lambda o: -(o.get("coverage_ratio") or 0))[:15]
        summary["top_observed"] = [
            {k: o.get(k) for k in ("guid", "category", "system", "zone", "status", "coverage_ratio", "center", "bbox")}
            for o in topo
        ]
        summary["matched_counts"] = {"matched": len(matched), "by_category": by_cat, "by_system": by_sys, "by_zone": by_zone}
        if matched:
            xs = [o["center"] for o in matched if o.get("center")]
            if xs:
                import statistics as S
                mn = [min(p[i] for p in xs) for i in range(3)]
                mx = [max(p[i] for p in xs) for i in range(3)]
                summary["matched_region_bbox_m"] = [ [round(v,2) for v in mn], [round(v,2) for v in mx] ]
                print(f"  matched region bbox(centers) m: {summary['matched_region_bbox_m']}")
    else:
        # fall back: report candidate placement + aligned scan bounds
        a = best.get("alignment") or {}
        summary["fallback"] = "no_coverage"
        summary["candidate_transform"] = {"scale": a.get("scale"), "translation": a.get("translation"), "rotation": a.get("rotation")}
        summary["aligned_scan_bounds"] = best.get("aligned_scan_bounds")
        print(f"  (no coverage) candidate translation={a.get('translation')}")
        print(f"  aligned_scan_bounds={best.get('aligned_scan_bounds')}")

    (out / f"{up}_align_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nwrote {out}/{up}_align_summary.json")


if __name__ == "__main__":
    main()

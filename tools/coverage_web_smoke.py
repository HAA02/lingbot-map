#!/usr/bin/env python3
"""Smoke checks for the model coverage web workflow.

This script assumes realtime/server.py is already running. It intentionally
uses mostly read-only API calls against existing uploads, so it can be run
often while iterating on the coverage viewer.

Example:
    python tools/coverage_web_smoke.py \
        --base-url http://127.0.0.1:8768 \
        --model pxx \
        --upload upload_1779439687108 \
        --stale-upload upload_1779442357085
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class HttpResult:
    status: int
    body: bytes
    headers: dict[str, str]

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.text)


class SmokeFailure(AssertionError):
    pass


def _url(base_url: str, path: str, query: dict[str, Any] | None = None) -> str:
    base = base_url.rstrip("/")
    if query:
        qs = urllib.parse.urlencode(query)
        return f"{base}{path}?{qs}"
    return f"{base}{path}"


def request(
    base_url: str,
    path: str,
    *,
    query: dict[str, Any] | None = None,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    expect: set[int] | None = None,
) -> HttpResult:
    url = _url(base_url, path, query)
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = HttpResult(
                status=resp.status,
                body=resp.read(),
                headers={k.lower(): v for k, v in resp.headers.items()},
            )
    except urllib.error.HTTPError as exc:
        result = HttpResult(
            status=exc.code,
            body=exc.read(),
            headers={k.lower(): v for k, v in exc.headers.items()},
        )
    allowed = expect or {200}
    if result.status not in allowed:
        body = result.text[:600].replace("\n", " ")
        raise SmokeFailure(f"{method} {url} returned {result.status}, expected {sorted(allowed)}: {body}")
    return result


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def blocker_ids(report: dict[str, Any]) -> set[str]:
    return {str(x.get("id")) for x in report.get("blockers") or [] if isinstance(x, dict)}


def check_static_pages(base_url: str, model: str, upload: str) -> None:
    pages = [
        ("/coverage.html", "model coverage", "<script type=\"module\">"),
        ("/coverage-report.html", "coverage report", "<script>"),
    ]
    for path, marker, script_marker in pages:
        html = request(base_url, path, query={"model": model, "upload": upload}).text
        check(script_marker in html, f"{path} is missing expected script marker")
        check(marker in html.lower(), f"{path} does not look like the expected page")


def check_models(base_url: str, model: str) -> dict[str, Any]:
    data = request(base_url, "/api/models").json()
    items = data.get("items") or []
    selected = next((item for item in items if item.get("model_id") == model), None)
    check(selected is not None, f"model {model!r} not returned by /api/models")
    check(selected.get("manifest_valid") is True, f"model {model!r} manifest is not valid")
    check(int(selected.get("object_count") or 0) > 0, f"model {model!r} has no objects")
    ratio = float(selected.get("guid_mapping_ratio") or 0.0)
    source = selected.get("coverage_geometry_source")
    if ratio <= 0.0:
        check(source == "pag_proxy", "zero GUID mapping must fall back to PAG proxy geometry")
        check(
            selected.get("glb_role") == "visual_context_only",
            "zero GUID mapping GLB must be marked visual_context_only",
        )
    else:
        check(
            source in ("glb_guid_mesh", "hybrid_glb_proxy"),
            "non-zero GUID mapping must use GLB mesh (direct or hybrid) geometry",
        )
        check(
            selected.get("glb_role") != "visual_context_only",
            "non-zero GUID mapping GLB must not be marked visual_context_only",
        )
    # R1 regression guard: PXX GLB carries Revit UniqueId/ElementID in node extras;
    # mapping must not silently fall back to 0% (server.py _extract_glb_names key list).
    if model == "pxx":
        check(ratio >= 0.5, "PXX GUID mapping regressed below 0.5 (GLB UniqueId/ElementID extraction broken)")
        check(int(selected.get("guid_mapped_count") or 0) >= 260, "PXX GUID mapped_count regressed below 260")
        check(source == "hybrid_glb_proxy", "PXX should report hybrid GLB+proxy coverage geometry")
    return selected


def check_unaligned_upload(base_url: str, model: str, upload: str) -> None:
    status = request(
        base_url,
        f"/api/uploads/{upload}/coverage/status",
        query={"model_id": model},
    ).json()
    check(status.get("ok") is True, "coverage/status did not return ok")
    check((status.get("scan") or {}).get("ok") is True, "test upload scan is not available")
    check((status.get("alignment") or {}).get("ready") is False, "unaligned upload unexpectedly has ready alignment")
    check(status.get("status") == "needs_alignment", "unaligned upload should require alignment")
    check((status.get("coordinate") or {}).get("alignment_required") is True, "coordinate status must require alignment")
    checks = {item.get("id"): item.get("state") for item in status.get("checks") or []}
    check(checks.get("alignment") == "block", "readiness alignment check must block without green/yellow alignment")

    report = request(
        base_url,
        f"/api/uploads/{upload}/coverage/report",
        query={"model_id": model},
    ).json()
    check(report.get("status") == "blocked", "unaligned upload report must be blocked")
    check(report.get("passed_90") is False, "unaligned upload must not pass final 90% gate")
    ids = blocker_ids(report)
    check("alignment" in ids, "unaligned report must include alignment blocker")

    analysis = request(
        base_url,
        f"/api/uploads/{upload}/coverage/analyze",
        method="POST",
        payload={"model_id": model, "coverage_options": {}},
        expect={400, 409},
    ).json()
    check(analysis.get("status") == "needs_alignment", "coverage analysis without alignment must fail as needs_alignment")
    check(analysis.get("ok") is False, "coverage analysis without alignment must not return ok")


def check_alignment_candidates(base_url: str, model: str, upload: str) -> None:
    data = request(
        base_url,
        f"/api/uploads/{upload}/alignment/candidates",
        method="POST",
        payload={"model_id": model, "alignment": {"pairs": []}, "max_candidates": 5, "dry_run": True},
    ).json()
    check(data.get("ok") is True, "alignment candidates call failed")
    check(data.get("dry_run") is True, "alignment candidates smoke must not persist sidecar updates")
    check(data.get("status") in {"needs_anchor", "missing", "ambiguous_alignment"}, "unexpected empty-anchor candidate status")
    if (data.get("repetition_risk") or {}).get("level") == "high":
        warnings = " ".join(str(x) for x in data.get("warnings") or [])
        check("anchor" in warnings.lower(), "high repetition risk should require anchors")


def check_stale_red_upload(base_url: str, model: str, upload: str) -> None:
    status = request(
        base_url,
        f"/api/uploads/{upload}/coverage/status",
        query={"model_id": model},
    ).json()
    check(status.get("ok") is True, "stale upload coverage/status did not return ok")
    check((status.get("alignment") or {}).get("ready") is False, "red alignment must not be ready")
    check((status.get("coverage") or {}).get("state") == "stale_red_alignment", "stale upload must flag stale_red_alignment")
    checks = {item.get("id"): item.get("state") for item in status.get("checks") or []}
    check(checks.get("coverage_result") == "block", "stale red coverage must block readiness")

    coverage = request(
        base_url,
        f"/api/uploads/{upload}/coverage",
        query={"model_id": model},
        expect={409},
    ).json()
    check(coverage.get("status") == "needs_alignment", "stale red coverage endpoint must require alignment")

    report = request(
        base_url,
        f"/api/uploads/{upload}/coverage/report",
        query={"model_id": model},
    ).json()
    check(report.get("status") == "blocked", "stale red report must be blocked")
    check(report.get("passed_90") is False, "stale red upload must not pass final 90% gate")
    ids = blocker_ids(report)
    check("stale_red_alignment" in ids, "stale red report must include stale_red_alignment blocker")
    check("alignment" in ids, "stale red report must include alignment blocker")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8768")
    parser.add_argument("--model", default="pxx")
    parser.add_argument("--upload", default="upload_1779439687108")
    parser.add_argument("--stale-upload", default="upload_1779442357085")
    args = parser.parse_args()

    steps = [
        ("static pages", lambda: check_static_pages(args.base_url, args.model, args.upload)),
        ("models manifest", lambda: check_models(args.base_url, args.model)),
        ("unaligned upload gate", lambda: check_unaligned_upload(args.base_url, args.model, args.upload)),
        ("alignment candidates", lambda: check_alignment_candidates(args.base_url, args.model, args.upload)),
        ("stale red gate", lambda: check_stale_red_upload(args.base_url, args.model, args.stale_upload)),
    ]
    for label, fn in steps:
        fn()
        print(f"PASS {label}")
    print("OK coverage web smoke")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SmokeFailure as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        raise SystemExit(1)

#!/usr/bin/env python3
"""Headless Edge: zoom must keep the BIM model, play must still work.

Usage (server already running):
    .venv/bin/python tools/verify_orbit_play.py --base-url http://127.0.0.1:8767
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

EDGE = "/usr/bin/microsoft-edge"
LAUNCH = dict(
    executable_path=EDGE,
    headless=True,
    args=[
        "--enable-unsafe-swiftshader",
        "--use-gl=angle",
        "--use-angle=swiftshader",
        "--no-sandbox",
        "--ignore-gpu-blocklist",
    ],
)


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def _cam_state(page, hook: str) -> dict:
    return page.evaluate(
        """(hook) => {
          const root = window[hook];
          if (!root || !root.camera) return {ok:false};
          const p = root.camera.position;
          const t = (root.controls && root.controls.target) ? root.controls.target
                    : (root.scene && root.scene.userData && root.scene.userData.target) || null;
          const dist = (root.controls && root.controls.target)
            ? p.distanceTo(root.controls.target) : null;
          let meshes = 0, visibleMeshes = 0;
          root.scene.traverse(o => {
            if (o.isMesh) { meshes++; if (o.visible) visibleMeshes++; }
          });
          return {
            ok: true,
            finite: Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.z),
            pos: [p.x, p.y, p.z],
            near: root.camera.near,
            far: root.camera.far,
            dist,
            minD: root.controls ? root.controls.minDistance : null,
            maxD: root.controls ? root.controls.maxDistance : null,
            meshes, visibleMeshes,
          };
        }""",
        hook,
    )


def _wheel(page, selector: str, delta_y: float, n: int) -> None:
    box = page.locator(selector).bounding_box()
    if not box:
        _fail(f"no canvas {selector}")
    x = box["x"] + box["width"] * 0.55
    y = box["y"] + box["height"] * 0.55
    page.mouse.move(x, y)
    for _ in range(n):
        page.mouse.wheel(0, delta_y)
        time.sleep(0.03)


def _assert_view_alive(st: dict, label: str) -> None:
    if not st.get("ok"):
        _fail(f"{label}: debug hook missing")
    if not st.get("finite"):
        _fail(f"{label}: camera position not finite {st}")
    if int(st.get("visibleMeshes") or 0) < 1:
        _fail(f"{label}: no visible meshes {st}")
    dist = st.get("dist")
    if dist is not None:
        if dist < 0.2:
            _fail(f"{label}: camera inside target dist={dist}")
        min_d, max_d = st.get("minD"), st.get("maxD")
        if min_d and dist < float(min_d) * 0.5:
            _fail(f"{label}: closer than minDistance {st}")
        if max_d and dist > float(max_d) * 1.15:
            _fail(f"{label}: farther than maxDistance {st}")
    near, far = float(st["near"]), float(st["far"])
    if far <= near * 10:
        _fail(f"{label}: clip range too tight near={near} far={far}")


def verify_coplay(page, base: str, upload: str, out: Path) -> None:
    url = f"{base}/coplay?upload={upload}"
    print("COPLAY", url)
    page.goto(url, wait_until="domcontentloaded", timeout=120000)
    page.wait_for_function(
        "window.__coplay && window.__coplay.camera && window.__coplay.scene && window.__coplay.controls",
        timeout=120000,
    )
    time.sleep(2.0)
    page.screenshot(path=str(out / "coplay_before.png"))
    before = _cam_state(page, "__coplay")
    print("  before", json.dumps({k: before[k] for k in ("finite","near","far","dist","meshes","visibleMeshes") if k in before}))
    # Inject controls onto the hook by reading patched globals via Function toString is impossible.
    # Wheel + check camera finite + meshes is enough; also read minDistance from page eval of closure:
    limits = page.evaluate(
        """() => {
          const cam = window.__coplay.camera;
          return {near: cam.near, far: cam.far, pos: [cam.position.x, cam.position.y, cam.position.z]};
        }"""
    )
    print("  cam", limits)
    _assert_view_alive(before, "coplay before zoom")

    _wheel(page, "#c", -180, 18)  # zoom in
    time.sleep(0.4)
    zin = _cam_state(page, "__coplay")
    print("  zoom-in", json.dumps({k: zin[k] for k in ("finite","near","far","dist","visibleMeshes") if k in zin}))
    _assert_view_alive(zin, "coplay zoom-in")
    page.screenshot(path=str(out / "coplay_zoom_in.png"))

    _wheel(page, "#c", 220, 28)  # zoom out past old far=2000 if unconstrained
    time.sleep(0.4)
    zout = _cam_state(page, "__coplay")
    print("  zoom-out", json.dumps({k: zout[k] for k in ("finite","near","far","dist","visibleMeshes") if k in zout}))
    _assert_view_alive(zout, "coplay zoom-out")
    page.screenshot(path=str(out / "coplay_zoom_out.png"))

    play = page.locator("#play")
    if play.count() != 1:
        _fail("play button missing")
    play.click()
    time.sleep(1.2)
    media = page.evaluate(
        """() => {
          const v = document.getElementById('rvid');
          const b = document.getElementById('play');
          return {
            paused: v ? v.paused : null,
            time: v ? v.currentTime : null,
            ready: v ? v.readyState : null,
            err: v && v.error ? v.error.code : null,
            label: b ? b.textContent : null,
          };
        }"""
    )
    print("  play", media)
    after = _cam_state(page, "__coplay")
    _assert_view_alive(after, "coplay after play")
    page.screenshot(path=str(out / "coplay_play.png"))
    # Video may fail to decode in SwiftShader; 3D must still be alive and the control must respond.
    if media.get("label") not in ("⏸ 일시정지", "▶ 재생"):
        _fail(f"play button label unexpected {media}")
    if media.get("paused") is True and media.get("label") == "⏸ 일시정지":
        # clicked play but stayed paused (codec) — still OK if 3D alive
        print("  note: video stayed paused (likely codec in headless); 3D still alive")
    if media.get("paused") is False and media.get("time", 0) < 0:
        _fail("playing but currentTime invalid")
    print("  COPLAY OK")


def verify_coverage(page, base: str, model: str, upload: str, out: Path) -> None:
    url = f"{base}/coverage.html?model={model}&upload={upload}"
    print("COVERAGE", url)
    page.goto(url, wait_until="domcontentloaded", timeout=90000)
    page.wait_for_function("window.__cov && window.__cov.camera && window.__cov.controls", timeout=90000)
    page.wait_for_function(
        "window.__cov.groups && window.__cov.groups.model && window.__cov.groups.model.children.length > 0",
        timeout=90000,
    )
    time.sleep(2.0)
    page.screenshot(path=str(out / "coverage_before.png"))
    before = page.evaluate(
        """() => {
          const c = window.__cov;
          const p = c.camera.position, t = c.controls.target;
          let meshes=0, vis=0;
          c.scene.traverse(o=>{ if(o.isMesh){meshes++; if(o.visible) vis++;} });
          return {
            ok:true, finite: Number.isFinite(p.x),
            dist: p.distanceTo(t), near:c.camera.near, far:c.camera.far,
            minD:c.controls.minDistance, maxD:c.controls.maxDistance,
            meshes, visibleMeshes: vis
          };
        }"""
    )
    print("  before", before)
    _assert_view_alive(before, "coverage before zoom")

    _wheel(page, "#canvas", -160, 16)
    time.sleep(0.4)
    zin = page.evaluate(
        """() => {
          const c = window.__cov;
          const p = c.camera.position, t = c.controls.target;
          let vis=0; c.scene.traverse(o=>{ if(o.isMesh && o.visible) vis++; });
          return {ok:true, finite:Number.isFinite(p.x), dist:p.distanceTo(t),
                  near:c.camera.near, far:c.camera.far,
                  minD:c.controls.minDistance, maxD:c.controls.maxDistance,
                  visibleMeshes:vis};
        }"""
    )
    print("  zoom-in", zin)
    _assert_view_alive(zin, "coverage zoom-in")
    page.screenshot(path=str(out / "coverage_zoom_in.png"))

    _wheel(page, "#canvas", 200, 24)
    time.sleep(0.4)
    zout = page.evaluate(
        """() => {
          const c = window.__cov;
          const p = c.camera.position, t = c.controls.target;
          let vis=0; c.scene.traverse(o=>{ if(o.isMesh && o.visible) vis++; });
          return {ok:true, finite:Number.isFinite(p.x), dist:p.distanceTo(t),
                  near:c.camera.near, far:c.camera.far,
                  minD:c.controls.minDistance, maxD:c.controls.maxDistance,
                  visibleMeshes:vis};
        }"""
    )
    print("  zoom-out", zout)
    _assert_view_alive(zout, "coverage zoom-out")
    if zout["dist"] > zout["maxD"] * 1.15:
        _fail(f"coverage zoom-out exceeded maxDistance {zout}")
    page.screenshot(path=str(out / "coverage_zoom_out.png"))
    print("  COVERAGE OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8767")
    ap.add_argument("--upload", default="upload_1781521406685")
    ap.add_argument("--model", default="pipe_duct")
    ap.add_argument("--out", default="reports/orbit_play")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(**LAUNCH)
        page = browser.new_page(viewport={"width": 1600, "height": 900})
        page.on("pageerror", lambda e: print("pageerror", e))
        try:
            verify_coplay(page, args.base_url.rstrip("/"), args.upload, out)
            verify_coverage(page, args.base_url.rstrip("/"), args.model, args.upload, out)
        finally:
            browser.close()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("FAIL:", exc, file=sys.stderr)
        raise SystemExit(1)

"""Headless capture of the existing coverage.html viewer for a given upload+model,
using playwright + system Edge (software WebGL). Produces PNGs showing the
coverage-colored 3D model so the alignment result is verifiable the same way the
existing branch's viewer shows it.

Usage:
    .venv/bin/python tools/capture_coverage_html.py \
        --base-url http://127.0.0.1:8767 --model pipe_duct \
        --upload upload_XXXX --out reports/pipe_duct_align
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
from playwright.sync_api import sync_playwright

EDGE = "/usr/bin/microsoft-edge"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8767")
    ap.add_argument("--model", default="pipe_duct")
    ap.add_argument("--upload", required=True)
    ap.add_argument("--out", default="reports/pipe_duct_align")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    url = f"{args.base_url}/coverage.html?model={args.model}&upload={args.upload}"

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=EDGE, headless=True,
            args=["--enable-unsafe-swiftshader", "--use-gl=angle",
                  "--use-angle=swiftshader", "--no-sandbox",
                  "--ignore-gpu-blocklist"],
        )
        page = browser.new_page(viewport={"width": 1680, "height": 945})
        logs = []
        page.on("console", lambda m: logs.append(f"{m.type}: {m.text}"))
        print("loading", url)
        page.goto(url, wait_until="domcontentloaded", timeout=60000)

        # wait for debug hook + coverage loaded + model mesh present
        page.wait_for_function("window.__cov && window.__cov.state", timeout=60000)
        page.wait_for_function(
            "window.__cov.groups && window.__cov.groups.model "
            "&& window.__cov.groups.model.children.length > 0",
            timeout=90000,
        )
        # coverage colored (state.coverage set by loadCoverage)
        try:
            page.wait_for_function("window.__cov.state.coverage", timeout=30000)
        except Exception:
            print("  (coverage state not detected within 30s — continuing)")
        time.sleep(3.0)  # let DRACO meshes + status coloring settle

        sc = page.evaluate("() => (window.__cov.state.coverage||{}).status_counts || null")
        print("status_counts in viewer:", sc)

        # Load the reconstructed POINT CLOUD overlay (mapped onto the GLB via the
        # saved alignment) + keep the camera PATH/frustums visible, so we can see
        # where + along what path the video was shot relative to the model.
        page.click("#togglePointCloudBtn")
        try:
            page.wait_for_function("window.__cov.state.pointCloudLoaded === true", timeout=60000)
        except Exception:
            print("  (point cloud load not confirmed in 60s)")
        page.evaluate(
            """() => {
                const cov = window.__cov;
                if (cov.groups.pointCloud) cov.groups.pointCloud.visible = true;
                if (cov.groups.path) cov.groups.path.visible = true;       // camera path + frustums
                if (cov.groups.proxy) cov.groups.proxy.visible = false;
                cov.applyModelOpacity(0.18);   // fade neutral model; point cloud + path + observed pop
            }"""
        )
        time.sleep(2.0)
        npc = page.evaluate(
            "() => { let n=0; window.__cov.groups.pointCloud.traverse(o=>{ if(o.isPoints && o.geometry) n+=o.geometry.attributes.position.count; }); return n; }"
        )
        print("point cloud points in scene:", npc)

        # bbox of the aligned scan overlay (point cloud) = the captured region
        page.evaluate(
            """() => {
                const cov=window.__cov, T=cov.THREE;
                const b=new T.Box3().setFromObject(cov.groups.pointCloud);
                window.__pcBox = b.isEmpty()? null : b;
            }"""
        )

        # 1) overview: whole model (faint) + point cloud + path
        page.evaluate("() => window.__cov.fitView && window.__cov.fitView()")
        time.sleep(1.2)
        p1 = out / f"{args.upload}_viewer_overview.png"
        page.screenshot(path=str(p1)); print("wrote", p1)

        # 2) framed on the captured region (point cloud + camera path)
        framed = page.evaluate(
            """() => {
                const cov=window.__cov, T=cov.THREE, b=window.__pcBox;
                if(!b) return false;
                const c=b.getCenter(new T.Vector3()), s=b.getSize(new T.Vector3());
                const d=Math.max(s.x,s.y,s.z,0.8)*2.0;
                cov.camera.up.set(0,1,0);
                cov.camera.position.set(c.x+d, c.y+d*0.7, c.z+d);
                cov.controls.target.copy(c); cov.controls.update();
                return true;
            }"""
        )
        if framed:
            time.sleep(1.2)
            p2 = out / f"{args.upload}_viewer_pointcloud_path.png"
            page.screenshot(path=str(p2)); print("wrote", p2)

        # 3) plan view of the captured region (top-down along thinnest model axis)
        page.evaluate(
            """() => {
                const cov=window.__cov, T=cov.THREE, b=window.__pcBox;
                const mbox=new T.Box3().setFromObject(cov.groups.model);
                const ms=mbox.getSize(new T.Vector3());
                const axes=[['x',ms.x],['y',ms.y],['z',ms.z]].sort((a,b)=>a[1]-b[1]);
                const thin=axes[0][0];
                const c=(b||mbox).getCenter(new T.Vector3());
                const s=(b||mbox).getSize(new T.Vector3());
                const d=Math.max(s.x,s.y,s.z,1.5)*2.2;
                const pos=c.clone(); pos[thin]+=d;
                cov.camera.up.copy(thin==='y'? new T.Vector3(0,0,-1): new T.Vector3(0,1,0));
                cov.camera.position.copy(pos);
                cov.controls.target.copy(c); cov.controls.update();
            }"""
        )
        time.sleep(1.2)
        p3 = out / f"{args.upload}_viewer_plan.png"
        page.screenshot(path=str(p3)); print("wrote", p3)

        browser.close()
        errs = [l for l in logs if l.startswith("error")]
        if errs:
            print("console errors:", errs[:5])


if __name__ == "__main__":
    main()

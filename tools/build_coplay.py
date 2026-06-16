"""FR-4 co-play (compare mode) — view the PHOTOREAL point-cloud render video
(GPU-splatted, e.g. build_demo_map/demo render) side-by-side with the WebGL
design model, synchronized. WebGL renders the .dtdx model (meshes render
perfectly in the browser) with a camera frustum that tracks the reconstruction
path and highlights the elements the current frame maps to.

Rationale: high-quality point-cloud rendering is hard in WebGL; the demo GPU
render already nails it. So show that video for the cloud, and use WebGL only
for the model + which-objects-mapped — comparing as-built vs design.

Usage:
    PYTHONPATH=. .venv/bin/python tools/build_coplay.py --upload upload_1781521406685 \
        --dtdx models/Gasan_7F/G7F_FAB_FXX_7F-0_Central_1.dtdx \
        --demo-html reports/pointcloud_map/demo_dense_161613.html --demo-match 161613 \
        --render-video out_kakao_fmt/KakaoTalk_20260615_161613773_demo_format.mp4 \
        --out reports/coplay/coplay.html
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.request
from pathlib import Path

import numpy as np

from scan2bim.dtdx_geometry import decode_geometry

TEMPLATE = r"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>co-play · 점군 렌더 ↔ 설계모델 비교</title>
<style>
  *{margin:0;box-sizing:border-box} html,body{height:100%;background:#0a0d13;color:#e8eaf0;font-family:system-ui,sans-serif;overflow:hidden}
  #wrap{position:fixed;inset:0 0 64px 0;display:flex}
  #left,#right{position:relative;width:50%;height:100%;overflow:hidden;border-right:1px solid #1c2330}
  #left{background:#000} #left video{width:100%;height:100%;object-fit:contain;background:#000}
  #right{background:#0c0f16} #c{width:100%;height:100%;display:block}
  .lbl{position:absolute;left:12px;top:10px;background:rgba(10,13,19,.78);border:1px solid #283042;border-radius:8px;padding:6px 11px;font-size:13px;font-weight:600;z-index:5}
  .lbl small{display:block;font-weight:400;color:#9fb0c8;font-size:11px;margin-top:2px}
  #hud{position:absolute;right:12px;top:10px;background:rgba(10,13,19,.78);border:1px solid #283042;border-radius:8px;padding:8px 11px;font-size:12px;color:#9fb0c8;z-index:5}
  #hud b{color:#e8eaf0}
  #bar{position:fixed;left:0;right:0;bottom:0;height:64px;display:flex;gap:12px;align-items:center;background:#0e1219;border-top:1px solid #283042;padding:0 18px}
  #bar button{background:#1c2433;color:#e8eaf0;border:1px solid #2c3344;border-radius:6px;padding:6px 14px;cursor:pointer;font-size:13px}
  #bar input[type=range]{flex:1} #bar .t{font-variant-numeric:tabular-nums;color:#9fb0c8;min-width:64px}
  .legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:middle}
</style>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
</head><body>
<div id="wrap">
  <div id="left"><div class="lbl">정합 렌더링 (점군)<small>GPU 스플랫 photoreal · 촬영자 시점</small></div>
    <video id="rvid" src="__RVIDEO__" muted playsinline preload="none"></video></div>
  <div id="right"><div class="lbl">설계 모델 (dtdx)<small>__MODELNAME__ · 현재 위치/방향 + 매핑 객체</small></div>
    <div id="hud">삼각형 <b>__TRIS__</b> · 포즈 <b>__NPOSES__</b><br><span class="legend">__LEGEND__</span><br><span id="finfo">frame —</span></div>
    <canvas id="c"></canvas></div>
</div>
<div id="bar"><button id="play">▶ 재생</button><button id="fcam">자유 시점</button><input id="seek" type="range" min="0" max="1000" value="0"><span class="t" id="t">0.0s</span></div>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const MESHES=__MESHES__, POSES=__POSES__, META=__META__;
const right=document.getElementById('right');
const renderer=new THREE.WebGLRenderer({canvas:document.getElementById('c'),antialias:true});
function rsize(){ const w=right.clientWidth,h=right.clientHeight; renderer.setSize(w,h,false); camera.aspect=w/h; camera.updateProjectionMatrix(); }
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
const scene=new THREE.Scene(); scene.background=new THREE.Color(0x0c0f16);
const camera=new THREE.PerspectiveCamera(55,1,0.05,2000);
scene.add(new THREE.AmbientLight(0xffffff,0.85));
const dl=new THREE.DirectionalLight(0xffffff,0.9); dl.position.set(30,60,30); scene.add(dl);
scene.add(new THREE.HemisphereLight(0xbfd4ff,0x202830,0.5));
scene.add(new THREE.GridHelper(60,60,0x223,0x182030));
const controls=new OrbitControls(camera,renderer.domElement);
function b64f32(s){const bin=atob(s);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return new Float32Array(u.buffer);}
const box=new THREE.Box3(); const MESH_OBJS=[];
for(const m of MESHES){
  const pos=b64f32(m.b64); const n=pos.length/3;
  const g=new THREE.BufferGeometry(); g.setAttribute('position',new THREE.BufferAttribute(pos,3)); g.computeVertexNormals();
  const base=new Float32Array(pos.length); for(let i=0;i<n;i++){base[i*3]=m.color[0];base[i*3+1]=m.color[1];base[i*3+2]=m.color[2];}
  const col=base.slice(); g.setAttribute('color',new THREE.BufferAttribute(col,3));
  const mat=new THREE.MeshStandardMaterial({vertexColors:true,roughness:0.7,metalness:0.05,side:THREE.DoubleSide});
  const mesh=new THREE.Mesh(g,mat); scene.add(mesh); box.expandByObject(mesh);
  MESH_OBJS.push({mesh,pos,base,col,n});
}
const c0=box.getCenter(new THREE.Vector3()), sz=box.getSize(new THREE.Vector3());
const pts=POSES.map(p=>new THREE.Vector3(p.c[0],p.c[1],p.c[2]));
if(pts.length){ scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),new THREE.LineBasicMaterial({color:0xffb347,linewidth:2}))); }
const frustum=new THREE.Group();
const cone=new THREE.Mesh(new THREE.ConeGeometry(0.5,1.2,4),new THREE.MeshBasicMaterial({color:0x31d27c,wireframe:true})); cone.rotation.x=Math.PI/2; frustum.add(cone);
frustum.add(new THREE.Mesh(new THREE.SphereGeometry(0.3,16,12),new THREE.MeshBasicMaterial({color:0x31d27c})));
scene.add(frustum);
const viewCam=new THREE.PerspectiveCamera(70,1.5,0.1,Math.max(2,Math.min(sz.x,sz.z)*0.3));
const _fr=new THREE.Frustum(),_m4=new THREE.Matrix4(),_v=new THREE.Vector3(); let lastHi=-1;
function highlight(i){ const p=POSES[i]; if(!p)return;
  viewCam.position.set(p.c[0],p.c[1],p.c[2]); viewCam.up.set(p.u[0],p.u[1],p.u[2]); viewCam.lookAt(p.c[0]+p.f[0],p.c[1]+p.f[1],p.c[2]+p.f[2]);
  viewCam.updateMatrixWorld(true); viewCam.updateProjectionMatrix(); _fr.setFromProjectionMatrix(_m4.multiplyMatrices(viewCam.projectionMatrix,viewCam.matrixWorldInverse));
  let hit=0;
  for(const o of MESH_OBJS){ for(let k=0;k<o.n;k++){ _v.set(o.pos[k*3],o.pos[k*3+1],o.pos[k*3+2]);
    if(_fr.containsPoint(_v)){o.col[k*3]=0.2;o.col[k*3+1]=1.0;o.col[k*3+2]=0.55;hit++;} else {o.col[k*3]=o.base[k*3];o.col[k*3+1]=o.base[k*3+1];o.col[k*3+2]=o.base[k*3+2];} }
    o.mesh.geometry.attributes.color.needsUpdate=true; }
  document.getElementById('finfo').textContent='frame '+i+' / '+(POSES.length-1)+' · 매핑 객체정점 '+hit;
}
let followCam=true; const fbtn=document.getElementById('fcam');
function applyFollow(i){ const p=POSES[i]; if(!p)return; const fwd=new THREE.Vector3(p.f[0],p.f[1],p.f[2]).normalize(); const up=new THREE.Vector3(p.u[0],p.u[1],p.u[2]).normalize();
  camera.position.set(p.c[0],p.c[1],p.c[2]).addScaledVector(fwd,-2.2).addScaledVector(up,0.7); camera.up.copy(up); camera.lookAt(p.c[0]+fwd.x*2,p.c[1]+fwd.y*2,p.c[2]+fwd.z*2); }
fbtn.textContent='자유 시점';
fbtn.onclick=()=>{ followCam=!followCam; fbtn.textContent=followCam?'자유 시점':'촬영자 시점'; controls.enabled=!followCam; if(followCam) applyFollow(lastHi<0?0:lastHi); };
function setFrustum(i){ i=Math.max(0,Math.min(POSES.length-1,i|0)); const p=POSES[i]; if(!p)return;
  frustum.position.set(p.c[0],p.c[1],p.c[2]); frustum.up.set(p.u[0],p.u[1],p.u[2]); frustum.lookAt(p.c[0]+p.f[0],p.c[1]+p.f[1],p.c[2]+p.f[2]);
  if(i!==lastHi){ lastHi=i; highlight(i); } if(followCam) applyFollow(i); }
rsize(); setFrustum(0);
const d=Math.max(sz.x,sz.y,sz.z)*0.6; camera.position.set(c0.x+d,c0.y+d*0.6,c0.z+d); controls.target.copy(c0); controls.update();
controls.enabled=!followCam;  // follow 모드면 OrbitControls 끔 (충돌 방지)
// shared timeline driven by the render video
const rvid=document.getElementById('rvid'), seek=document.getElementById('seek'), tlab=document.getElementById('t'), playb=document.getElementById('play');
let dur=Math.max(META.duration||1,0.1);
rvid.addEventListener('loadedmetadata',()=>{dur=rvid.duration||dur;});
playb.onclick=()=>{ if(rvid.paused){rvid.play();playb.textContent='⏸ 일시정지';} else {rvid.pause();playb.textContent='▶ 재생';} };
seek.oninput=()=>{ rvid.currentTime=(seek.value/1000)*dur; };
function sync(t){ const f=Math.max(0,Math.min(1,t/dur)); seek.value=Math.round(f*1000); tlab.textContent=t.toFixed(1)+'s'; setFrustum(Math.round(f*(POSES.length-1))); }
addEventListener('resize', rsize);
function loop(){ requestAnimationFrame(loop); sync(rvid.currentTime||0); if(!followCam) controls.update(); renderer.render(scene,camera); }
loop();
window.__coplay={scene,camera,POSES,MESHES,box,setFrustum,setFollow:(v)=>{followCam=!!v;}};
</script></body></html>"""


def fetch_scan(base_url, upload, max_points=60000):
    u = (f"{base_url.rstrip('/')}/api/uploads/{upload}/scan"
         f"?variant=detail&include_points=true&include_thumbs=false&max_points={max_points}")
    with urllib.request.urlopen(u, timeout=180) as r:
        d = json.loads(r.read().decode("utf-8", "replace"))
    return (d.get("poses") or []), np.asarray(d.get("points") or [], dtype=np.float64)


def load_demo_cloud(html_path, match):
    import re
    html = Path(html_path).read_text(encoding="utf-8")
    m = (re.search(r'(\[\{"label".*?\}\])\s*[;\)]', html, re.S)
         or re.search(r'(\[\{.*?"xyz".*?\}\])', html, re.S))
    if not m:
        raise ValueError(f"no datasets in {html_path}")
    ds = json.loads(m.group(1))
    d = next((x for x in ds if match in f"{x.get('label','')}|{x.get('up','')}"), ds[-1])
    xyz = np.frombuffer(base64.b64decode(d["xyz"]), dtype="<f4").reshape(-1, 3).astype(np.float64)
    return (d.get("poses") or []), xyz


def viewer_pose(p12):
    c = np.asarray(p12, dtype=np.float64)
    R = np.array([[c[0], -c[1], -c[2]], [-c[4], c[5], c[6]], [-c[8], c[9], c[10]]])
    return R, np.array([c[3], -c[7], -c[11]])


def _rot_a_to_b(a, b):
    a = a / (np.linalg.norm(a) + 1e-12); b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b); c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def place_gravity(poses, scan_pts, bbox, scale=1.0):
    centers, fwd, up = [], [], []
    for p in poses:
        R, t = viewer_pose(p); centers.append(t); up.append(R[:, 1]); fwd.append(R[:, 2])
    centers = np.array(centers); up = np.array(up); fwd = np.array(fwd)
    g = up.mean(0); g /= (np.linalg.norm(g) + 1e-9)
    mn, mx = np.array(bbox[0]), np.array(bbox[1]); span = mx - mn
    upax = int(np.argmin(span)); mc = (mn + mx) / 2.0; pipe_level = mc[upax]
    mup = np.zeros(3); mup[upax] = 1.0
    Rg = _rot_a_to_b(g, mup)
    P = scan_pts.copy()
    if P.size:
        P[:, 1] *= -1.0; P[:, 2] *= -1.0; P = P @ Rg.T
    C = centers @ Rg.T; F = fwd @ Rg.T; U = up @ Rg.T
    horiz = [i for i in range(3) if i != upax]
    base_pts = P if P.size else C
    if P.size:
        P *= scale
    C *= scale
    up_src = (P[:, upax] if P.size else C[:, upax])
    ceil = float(np.percentile(up_src, 90)); cen = base_pts.mean(0) * scale
    off = np.zeros(3); off[upax] = pipe_level - ceil
    for i in horiz:
        off[i] = mc[i] - cen[i]
    C += off
    return [{"c": [round(float(x), 3) for x in C[i]], "f": [round(float(x), 4) for x in F[i]],
             "u": [round(float(x), 4) for x in U[i]]} for i in range(len(C))]


def place_registered(poses, scan_pts, model_ceiling, bbox):
    """Real registration: gravity-align scan, then register its CEILING band to
    the model ceiling (grid XY + yaw + scale + Umeyama-ICP). Returns placed
    poses + fit metrics. (Footage looks up → ceiling-to-ceiling locks well.)"""
    from scipy.spatial import cKDTree
    from scan2bim.registration import _icp
    centers, fwd, up = [], [], []
    for p in poses:
        R, t = viewer_pose(p); centers.append(t); up.append(R[:, 1]); fwd.append(R[:, 2])
    centers = np.array(centers); up = np.array(up); fwd = np.array(fwd)
    g = up.mean(0); g /= (np.linalg.norm(g) + 1e-9)
    Rg = _rot_a_to_b(g, np.array([0.0, 1.0, 0.0]))
    P = scan_pts.copy(); P[:, 1] *= -1.0; P[:, 2] *= -1.0; Pg = P @ Rg.T
    Cg = centers @ Rg.T; Fg = fwd @ Rg.T; Ug = up @ Rg.T
    Sc = Pg[Pg[:, 1] >= np.percentile(Pg[:, 1], 55)]
    sc = Sc[np.linspace(0, len(Sc) - 1, min(2500, len(Sc))).astype(int)]; scan_c = sc.mean(0)
    tree = cKDTree(model_ceiling)
    lo, hi = np.array(bbox[0]), np.array(bbox[1])
    yc = float(np.percentile(model_ceiling[:, 1], 50))

    # camera WALK direction (gravity-aligned, horizontal) vs model PIPE direction
    # — pins yaw so the 3D path follows pipes (the video walks straight along them).
    Ch = Cg[:, [0, 2]] - Cg[:, [0, 2]].mean(0)
    traj2d = np.linalg.eigh(Ch.T @ Ch)[1][:, -1]
    a_s = np.degrees(np.arctan2(traj2d[1], traj2d[0]))
    Mh = model_ceiling[:, [0, 2]] - model_ceiling[:, [0, 2]].mean(0)
    pipe2d = np.linalg.eigh(Mh.T @ Mh)[1][:, -1]
    a_p = np.degrees(np.arctan2(pipe2d[1], pipe2d[0]))

    def Ry(a):
        r = np.deg2rad(a); return np.array([[np.cos(r), 0, np.sin(r)], [0, 1, 0], [-np.sin(r), 0, np.cos(r)]])

    best = None
    for s in (1.0, 1.15, 1.3, 1.45, 1.6):
        for yaw in range(0, 360, 15):
            R = Ry(yaw)
            align = abs(float(np.cos(np.deg2rad(a_s + yaw - a_p))))  # 1=traj∥pipes
            for cx in np.linspace(lo[0] + 2, hi[0] - 2, 7):
                for cz in np.linspace(lo[2] + 2, hi[2] - 2, 9):
                    t = np.array([cx, yc, cz]) - s * (R @ scan_c)
                    d, _ = tree.query(s * (sc @ R.T) + t, workers=-1)
                    score = float((d < 0.25).mean()) + 0.25 * align
                    if best is None or score > best[0]:
                        best = (score, s, yaw, R, t)
    _sc, s0, yaw, R0, t0 = best
    s2, R2, t2, rmse, inl2 = _icp(sc, model_ceiling, tree, float(s0), R0, t0, iters=30)
    Ct = s2 * (Cg @ R2.T) + t2; Ft = Fg @ R2.T; Ut = Ug @ R2.T
    pose_json = [{"c": [round(float(x), 3) for x in Ct[i]], "f": [round(float(x), 4) for x in Ft[i]],
                  "u": [round(float(x), 4) for x in Ut[i]]} for i in range(len(Ct))]
    return pose_json, {"inlier": round(float(inl2), 3), "rmse": round(float(rmse), 3),
                       "scale": round(float(s2), 3), "yaw": int(yaw)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8767")
    ap.add_argument("--upload", required=True)
    ap.add_argument("--dtdx", nargs="+", required=True)
    ap.add_argument("--render-video", required=True, help="GPU point-cloud render mp4 (left panel)")
    ap.add_argument("--demo-html", default=None, help="pointcloud map HTML for poses (build_demo_map)")
    ap.add_argument("--demo-match", default="161613")
    ap.add_argument("--duration", type=float, default=35.3)
    ap.add_argument("--out", default="reports/coplay/coplay.html")
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    meshes_json, tris, names, model_pts = [], 0, [], []
    RENDER_CAP = 90000  # per color group (triangle-soup) for the browser
    for f in args.dtdx:
        g = decode_geometry(f); names.append(g["discipline_code"]); tris += g["triangle_count"]
        for m in g["meshes"]:
            pos = np.asarray(m["positions"], dtype=np.float32)
            if not len(pos):
                continue
            model_pts.append(pos)
            if len(pos) > RENDER_CAP:  # decimate by whole triangles
                ntri = len(pos) // 3
                keep = np.linspace(0, ntri - 1, RENDER_CAP // 3).astype(int)
                pos = pos.reshape(-1, 3, 3)[keep].reshape(-1, 3)
            meshes_json.append({"color": [round(c, 4) for c in m["color"]],
                                "b64": base64.b64encode(np.ascontiguousarray(pos).tobytes()).decode("ascii")})
    allp = np.concatenate(model_pts).astype(np.float64)
    lo = np.percentile(allp, 1, axis=0); hi = np.percentile(allp, 99, axis=0)
    allc = allp[((allp >= lo) & (allp <= hi)).all(1)]   # robust (clip outliers)
    bbox = (allc.min(0).tolist(), allc.max(0).tolist())
    ceil = allc[allc[:, 1] >= (bbox[1][1] - 1.5)]        # top 1.5 m = ceiling band
    ceil = ceil[np.linspace(0, len(ceil) - 1, min(60000, len(ceil))).astype(int)]

    if args.demo_html:
        poses, scan_pts = load_demo_cloud(args.demo_html, args.demo_match)
    else:
        poses, scan_pts = fetch_scan(args.base_url, args.upload)
    pose_json, reginfo = place_registered(poses, scan_pts, ceil, bbox)
    print("  registration:", reginfo)

    legend = []
    for m in sorted(meshes_json, key=lambda x: -len(x["b64"]))[:5]:
        hexc = "#%02x%02x%02x" % tuple(int(max(0, min(1, c)) * 255) for c in m["color"][:3])
        legend.append(f'<i style="background:{hexc}"></i>')

    rel_video = os.path.relpath(Path(args.render_video).resolve(), out.parent.resolve())
    html = (TEMPLATE
            .replace("__MESHES__", json.dumps(meshes_json))
            .replace("__POSES__", json.dumps(pose_json))
            .replace("__META__", json.dumps({"duration": args.duration}))
            .replace("__MODELNAME__", " + ".join(names))
            .replace("__TRIS__", f"{tris:,}")
            .replace("__NPOSES__", str(len(pose_json)))
            .replace("__LEGEND__", " ".join(legend))
            .replace("__RVIDEO__", rel_video))
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB) tris={tris:,} poses={len(pose_json)} render={rel_video}")


if __name__ == "__main__":
    main()

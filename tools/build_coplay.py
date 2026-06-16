"""FR-4 co-play harness — render a .dtdx model in WebGL alongside the source
video + the reconstructed scan point cloud, with a camera frustum moving along
the path synced to video time. "영상 + dtdx를 한 화면에서 함께 검증".

Placement uses the reconstruction GRAVITY (mean camera up-axis, the method the
coverage server uses) to align the scan's up to the model up-axis, scales to a
plausible footprint, and seats the camera path BELOW the pipe layer — because
the footage looks UP at ceiling pipes from a horizontal floor space. XY/yaw and
metric scale remain approximate until FR-2.1 registration on matched footage.

Usage:
    PYTHONPATH=. .venv/bin/python tools/build_coplay.py \
        --base-url http://127.0.0.1:8767 --upload upload_1781521406685 \
        --dtdx models/Gasan_7F/G7F_FAB_FXX_7F-0_Central_1.dtdx \
        --video realtime/_uploads/upload_1781521406685.mp4 \
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
<title>co-play · dtdx ↔ video</title>
<style>
  *{margin:0;box-sizing:border-box} html,body{height:100%;background:#0c0f16;color:#e8eaf0;font-family:system-ui,sans-serif;overflow:hidden}
  #c{position:fixed;inset:0;display:block}
  #vidpanel{position:fixed;right:16px;top:16px;width:380px;background:#11151f;border:1px solid #283042;border-radius:10px;overflow:hidden;box-shadow:0 6px 24px rgba(0,0,0,.5)}
  #vidpanel video{width:100%;display:block;background:#000}
  #vidpanel .cap{padding:6px 10px;font-size:12px;color:#9fb0c8;border-top:1px solid #283042}
  #hud{position:fixed;left:16px;top:16px;background:#11151f;border:1px solid #283042;border-radius:10px;padding:12px 14px;font-size:13px;max-width:360px}
  #hud h1{font-size:14px;margin-bottom:6px}
  #hud .row{margin:3px 0;color:#9fb0c8} #hud b{color:#e8eaf0}
  .legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:middle}
  #bar{position:fixed;left:16px;bottom:16px;right:412px;display:flex;gap:10px;align-items:center;background:#11151f;border:1px solid #283042;border-radius:10px;padding:8px 12px}
  #bar button{background:#1c2433;color:#e8eaf0;border:1px solid #2c3344;border-radius:6px;padding:4px 12px;cursor:pointer}
  #bar input[type=range]{flex:1}
  .note{color:#e8a35a}
</style>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
</head><body>
<canvas id="c"></canvas>
<div id="hud"><h1>영상 ↔ dtdx co-play</h1>
  <div class="row">모델: <b>__MODELNAME__</b></div>
  <div class="row">삼각형: <b>__TRIS__</b> · 포즈: <b>__NPOSES__</b> · 스캔점: <b>__NSCAN__</b></div>
  <div class="row legend">__LEGEND__</div>
  <div class="row note">중력정렬+천장고 배치(카메라 배관 아래·올려봄). XY·yaw·scale은 정합(FR-2.1)·Metric3D 전 근사.</div>
  <div class="row" id="frameinfo">frame —</div>
</div>
<div id="vidpanel"><video id="vid" src="__VIDEO__" muted playsinline preload="none"></video><div class="cap">원본 촬영영상 (촬영자 시점)</div></div>
<div id="bar"><button id="play">▶ 재생</button><button id="fcam">촬영자 시점</button><input id="seek" type="range" min="0" max="1000" value="0"><span id="t">0.0s</span></div>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const MESHES=__MESHES__, POSES=__POSES__, SCAN=__SCAN__, META=__META__;
const renderer=new THREE.WebGLRenderer({canvas:document.getElementById('c'),antialias:true});
renderer.setSize(innerWidth,innerHeight); renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));
const scene=new THREE.Scene(); scene.background=new THREE.Color(0x0c0f16);
const camera=new THREE.PerspectiveCamera(55,innerWidth/innerHeight,0.05,5000);
scene.add(new THREE.AmbientLight(0xffffff,0.8));
const dl=new THREE.DirectionalLight(0xffffff,0.85); dl.position.set(30,60,30); scene.add(dl);
scene.add(new THREE.HemisphereLight(0xbfd4ff,0x202830,0.5));
const controls=new OrbitControls(camera,renderer.domElement);
function b64f32(s){const bin=atob(s);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return new Float32Array(u.buffer);}
function b64u8(s){const bin=atob(s);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u;}
const box=new THREE.Box3();
const MESH_OBJS=[];
for(const m of MESHES){
  const pos=b64f32(m.b64); const n=pos.length/3;
  const g=new THREE.BufferGeometry(); g.setAttribute('position',new THREE.BufferAttribute(pos,3)); g.computeVertexNormals();
  const base=new Float32Array(pos.length);
  for(let i=0;i<n;i++){base[i*3]=m.color[0];base[i*3+1]=m.color[1];base[i*3+2]=m.color[2];}
  const col=base.slice(); g.setAttribute('color',new THREE.BufferAttribute(col,3));
  const mat=new THREE.MeshStandardMaterial({vertexColors:true,roughness:0.75,metalness:0.05,side:THREE.DoubleSide,transparent:true,opacity:0.85});
  const mesh=new THREE.Mesh(g,mat); scene.add(mesh); box.expandByObject(mesh);
  MESH_OBJS.push({mesh,pos,base,col,n});
}
// reconstructed scan point cloud (the real ceiling the camera filmed) — demo-quality soft round points
function discTex(){const cv=document.createElement('canvas');cv.width=cv.height=64;const x=cv.getContext('2d');
  x.fillStyle='#fff';x.beginPath();x.arc(32,32,31,0,Math.PI*2);x.fill();return new THREE.CanvasTexture(cv);}
let scanPts=null;
if(SCAN && SCAN.pos){
  const sp=b64f32(SCAN.pos), sc=b64u8(SCAN.col);
  const col=new Float32Array(sp.length); for(let i=0;i<sc.length;i++) col[i]=sc[i]/255;
  const g=new THREE.BufferGeometry(); g.setAttribute('position',new THREE.BufferAttribute(sp,3)); g.setAttribute('color',new THREE.BufferAttribute(col,3));
  // crisp opaque round points (no alpha wash) → photoreal like the demo video
  const pm=new THREE.PointsMaterial({size:__PSIZE__,map:discTex(),vertexColors:true,sizeAttenuation:true,transparent:false,depthWrite:true,alphaTest:0.5});
  scanPts=new THREE.Points(g,pm); scene.add(scanPts);
}
const c0=box.getCenter(new THREE.Vector3()), sz=box.getSize(new THREE.Vector3());
const pts=POSES.map(p=>new THREE.Vector3(p.c[0],p.c[1],p.c[2]));
if(pts.length){ scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),new THREE.LineBasicMaterial({color:0xffb347,transparent:true,opacity:0.9}))); }
const frustum=new THREE.Group();
const cone=new THREE.Mesh(new THREE.ConeGeometry(0.5,1.2,4),new THREE.MeshBasicMaterial({color:0x31d27c,wireframe:true})); cone.rotation.x=Math.PI/2; frustum.add(cone);
frustum.add(new THREE.Mesh(new THREE.SphereGeometry(0.3,16,12),new THREE.MeshBasicMaterial({color:0x31d27c})));
scene.add(frustum);
const viewCam=new THREE.PerspectiveCamera(70,1.4,0.1,Math.max(2,Math.min(sz.x,sz.z)*0.3));
const _fr=new THREE.Frustum(), _m4=new THREE.Matrix4(), _v=new THREE.Vector3();
let lastHi=-1;
function highlight(i){
  const p=POSES[i]; if(!p)return;
  viewCam.position.set(p.c[0],p.c[1],p.c[2]); viewCam.up.set(p.u[0],p.u[1],p.u[2]);
  viewCam.lookAt(p.c[0]+p.f[0],p.c[1]+p.f[1],p.c[2]+p.f[2]);
  viewCam.updateMatrixWorld(true); viewCam.updateProjectionMatrix();
  _fr.setFromProjectionMatrix(_m4.multiplyMatrices(viewCam.projectionMatrix,viewCam.matrixWorldInverse));
  let hit=0;
  for(const o of MESH_OBJS){
    for(let k=0;k<o.n;k++){ _v.set(o.pos[k*3],o.pos[k*3+1],o.pos[k*3+2]);
      if(_fr.containsPoint(_v)){o.col[k*3]=0.2;o.col[k*3+1]=1.0;o.col[k*3+2]=0.55;hit++;}
      else{o.col[k*3]=o.base[k*3];o.col[k*3+1]=o.base[k*3+1];o.col[k*3+2]=o.base[k*3+2];}}
    o.mesh.geometry.attributes.color.needsUpdate=true;
  }
  document.getElementById('frameinfo').textContent='frame '+i+' / '+(POSES.length-1)+' · 매핑 정점 '+hit;
}
let followCam=false; const fbtn=document.getElementById('fcam');
function applyFollow(i){ const p=POSES[i]; if(!p)return;
  const fwd=new THREE.Vector3(p.f[0],p.f[1],p.f[2]).normalize();
  const up=new THREE.Vector3(p.u[0],p.u[1],p.u[2]).normalize();
  // chase cam: behind + above the photographer, looking where they look
  camera.position.set(p.c[0],p.c[1],p.c[2]).addScaledVector(fwd,-2.5).addScaledVector(up,0.8);
  camera.up.copy(up); camera.lookAt(p.c[0]+fwd.x*2,p.c[1]+fwd.y*2,p.c[2]+fwd.z*2);
}
fbtn.onclick=()=>{ followCam=!followCam; fbtn.textContent=followCam?'자유 시점':'촬영자 시점'; controls.enabled=!followCam; if(followCam) applyFollow(lastHi<0?0:lastHi); };
function setFrustum(i){
  i=Math.max(0,Math.min(POSES.length-1,i|0)); const p=POSES[i]; if(!p)return;
  frustum.position.set(p.c[0],p.c[1],p.c[2]);
  const tgt=new THREE.Vector3(p.c[0]+p.f[0],p.c[1]+p.f[1],p.c[2]+p.f[2]);
  frustum.up.set(p.u[0],p.u[1],p.u[2]); frustum.lookAt(tgt);
  if(i!==lastHi){ lastHi=i; highlight(i); }
  if(followCam) applyFollow(i);
}
setFrustum(0);
const d=Math.max(sz.x,sz.y,sz.z)*1.1;
camera.position.set(c0.x+d,c0.y+d*0.8,c0.z+d); controls.target.copy(c0); controls.update();
const vid=document.getElementById('vid'), seek=document.getElementById('seek'), tlab=document.getElementById('t'), playb=document.getElementById('play');
let dur=Math.max(META.duration||1,0.1);
vid.addEventListener('loadedmetadata',()=>{dur=vid.duration||dur;});
playb.onclick=()=>{ if(vid.paused){vid.play();playb.textContent='⏸ 일시정지';} else {vid.pause();playb.textContent='▶ 재생';} };
seek.oninput=()=>{ vid.currentTime=(seek.value/1000)*dur; };
function syncFromTime(t){ const frac=Math.max(0,Math.min(1,t/dur)); seek.value=Math.round(frac*1000); tlab.textContent=t.toFixed(1)+'s'; setFrustum(Math.round(frac*(POSES.length-1))); }
addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
function loop(){ requestAnimationFrame(loop); syncFromTime(vid.currentTime||0); controls.update(); renderer.render(scene,camera); }
loop();
window.__coplay={scene,camera,POSES,MESHES,SCAN,box,setFrustum};
</script></body></html>"""


def fetch_scan(base_url, upload, max_points=60000):
    u = (f"{base_url.rstrip('/')}/api/uploads/{upload}/scan"
         f"?variant=detail&include_points=true&include_thumbs=false&max_points={max_points}")
    with urllib.request.urlopen(u, timeout=180) as r:
        d = json.loads(r.read().decode("utf-8", "replace"))
    poses = d.get("poses") or []
    pts = np.asarray(d.get("points") or [], dtype=np.float64)
    cols = np.asarray(d.get("colors") or [], dtype=np.uint8)
    return poses, pts, cols


def load_demo_cloud(html_path, match):
    """Load the dense, photoreal demo reconstruction (build_demo_map output) for
    a dataset matching `match` (label or upload id) from a pointcloud map HTML.
    Returns (poses, xyz_raw float64, rgb uint8) in the recon frame."""
    import re
    html = Path(html_path).read_text(encoding="utf-8")
    m = (re.search(r'(\[\{"label".*?\}\])\s*[;\)]', html, re.S)
         or re.search(r'(\[\{.*?"xyz".*?\}\])', html, re.S))
    if not m:
        raise ValueError(f"no datasets found in {html_path}")
    ds = json.loads(m.group(1))
    d = next((x for x in ds if match in f"{x.get('label','')}|{x.get('up','')}"), ds[-1])
    xyz = np.frombuffer(base64.b64decode(d["xyz"]), dtype="<f4").reshape(-1, 3).astype(np.float64)
    if d.get("rgb"):
        rgb = np.frombuffer(base64.b64decode(d["rgb"]), dtype=np.uint8).reshape(-1, 3)
    else:
        rgb = np.full((len(xyz), 3), 180, np.uint8)
    return (d.get("poses") or []), xyz, rgb


def viewer_pose(p12):
    c = np.asarray(p12, dtype=np.float64)
    R = np.array([[c[0], -c[1], -c[2]], [-c[4], c[5], c[6]], [-c[8], c[9], c[10]]])
    t = np.array([c[3], -c[7], -c[11]])
    return R, t


def _rot_a_to_b(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def place_gravity(poses, scan_pts, bbox, clearance=2.5, scale=1.0):
    """Gravity-align scan+path to the model up-axis and seat the camera below
    the pipe layer. Returns (pose_json, scan_pos_f32, info)."""
    centers, fwd, up = [], [], []
    for p in poses:
        R, t = viewer_pose(p)
        centers.append(t); up.append(R[:, 1]); fwd.append(R[:, 2])
    centers = np.array(centers); up = np.array(up); fwd = np.array(fwd)
    g = up.mean(0); g /= (np.linalg.norm(g) + 1e-9)  # recon gravity-up

    mn, mx = np.array(bbox[0]), np.array(bbox[1])
    span = mx - mn
    upax = int(np.argmin(span))                 # model up = thinnest (slab normal)
    mc = (mn + mx) / 2.0
    pipe_level = mc[upax]
    mup = np.zeros(3); mup[upax] = 1.0
    Rg = _rot_a_to_b(g, mup)

    P = scan_pts.copy()
    if P.size:
        P[:, 1] *= -1.0; P[:, 2] *= -1.0       # viewer points
        P = P @ Rg.T
    C = centers @ Rg.T
    F = fwd @ Rg.T
    U = up @ Rg.T

    horiz = [i for i in range(3) if i != upax]
    base_pts = P if P.size else C
    s = float(scale)  # recon is ~metric (auto-place found scale≈1.0); Metric3D refines.
    if P.size:
        P *= s
    C *= s

    # offset: ceiling(90th pct up of scan, else cameras) -> pipe_level; XY centered
    up_src = (P[:, upax] if P.size else C[:, upax])
    ceil = float(np.percentile(up_src, 90))
    cen = base_pts.mean(0) * s
    off = np.zeros(3)
    off[upax] = pipe_level - ceil
    for i in horiz:
        off[i] = mc[i] - cen[i]
    if P.size:
        P += off
    C += off

    pose_json = [{"c": [round(float(x), 3) for x in C[i]],
                  "f": [round(float(x), 4) for x in F[i]],
                  "u": [round(float(x), 4) for x in U[i]]} for i in range(len(C))]
    info = {"up_axis": "xyz"[upax], "pipe_level": round(pipe_level, 2),
            "camera_level": round(float(C[:, upax].mean()), 2), "scale": round(s, 4)}
    return pose_json, (P.astype(np.float32) if P.size else np.zeros((0, 3), np.float32)), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8767")
    ap.add_argument("--upload", required=True)
    ap.add_argument("--dtdx", nargs="+", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--duration", type=float, default=35.3)
    ap.add_argument("--scan-points", type=int, default=250000)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="recon→model scale (≈1 = metric; Metric3D refines)")
    ap.add_argument("--demo-html", default=None,
                    help="pointcloud map HTML with the dense photoreal demo cloud "
                         "(build_demo_map output) — far higher quality than realtime /scan")
    ap.add_argument("--demo-match", default="161613", help="dataset label/upload to pick from --demo-html")
    ap.add_argument("--point-size", type=float, default=0.025, help="webgl point size (m)")
    ap.add_argument("--out", default="reports/coplay/coplay.html")
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    meshes_json, tris, names, model_pts = [], 0, [], []
    for f in args.dtdx:
        g = decode_geometry(f)
        names.append(Path(f).name); tris += g["triangle_count"]
        for m in g["meshes"]:
            pos = np.asarray(m["positions"], dtype=np.float32)
            if not len(pos):
                continue
            model_pts.append(pos)
            meshes_json.append({"color": [round(c, 4) for c in m["color"]],
                                "b64": base64.b64encode(pos.tobytes()).decode("ascii")})
    allp = np.concatenate(model_pts) if model_pts else np.zeros((1, 3), np.float32)
    bbox = (allp.min(0).tolist(), allp.max(0).tolist())

    if args.demo_html:
        poses, scan_pts, scan_cols = load_demo_cloud(args.demo_html, args.demo_match)
    else:
        poses, scan_pts, scan_cols = fetch_scan(args.base_url, args.upload, args.scan_points)
    pose_json, scan_pos, info = place_gravity(poses, scan_pts, bbox, scale=args.scale)
    scan_json = None
    if len(scan_pos):
        if len(scan_cols) != len(scan_pos):
            scan_cols = np.full((len(scan_pos), 3), 180, np.uint8)
        scan_json = {"pos": base64.b64encode(scan_pos.tobytes()).decode("ascii"),
                     "col": base64.b64encode(scan_cols.astype(np.uint8).tobytes()).decode("ascii")}

    legend = []
    for m in sorted(meshes_json, key=lambda x: -len(x["b64"]))[:5]:
        hexc = "#%02x%02x%02x" % tuple(int(max(0, min(1, c)) * 255) for c in m["color"][:3])
        legend.append(f'<i style="background:{hexc}"></i>')

    rel_video = os.path.relpath(Path(args.video).resolve(), out.parent.resolve())
    html = (TEMPLATE
            .replace("__MESHES__", json.dumps(meshes_json))
            .replace("__POSES__", json.dumps(pose_json))
            .replace("__SCAN__", json.dumps(scan_json))
            .replace("__META__", json.dumps({"duration": args.duration}))
            .replace("__MODELNAME__", " + ".join(names))
            .replace("__TRIS__", f"{tris:,}")
            .replace("__NPOSES__", str(len(pose_json)))
            .replace("__NSCAN__", f"{len(scan_pos):,}")
            .replace("__LEGEND__", " ".join(legend))
            .replace("__PSIZE__", repr(float(args.point_size)))
            .replace("__VIDEO__", rel_video))
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB) tris={tris:,} poses={len(pose_json)} "
          f"scan={len(scan_pos):,} placement={info}")


if __name__ == "__main__":
    main()

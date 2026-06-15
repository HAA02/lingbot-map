"""FR-4 co-play harness — render a .dtdx model in WebGL alongside the source
video, with a camera frustum moving along the reconstruction path synced to
video time. "영상 + dtdx를 한 화면에서 함께 검증".

Placement of the camera path is a PLACEHOLDER fit into the model bbox until
FR-2.1 (structural-shell registration) supplies the real transform.

Usage:
    .venv/bin/python tools/build_coplay.py \
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
  #hud{position:fixed;left:16px;top:16px;background:#11151f;border:1px solid #283042;border-radius:10px;padding:12px 14px;font-size:13px;max-width:340px}
  #hud h1{font-size:14px;margin-bottom:6px}
  #hud .row{margin:3px 0;color:#9fb0c8}
  #hud b{color:#e8eaf0}
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
  <div class="row">삼각형: <b>__TRIS__</b> · 포즈: <b>__NPOSES__</b></div>
  <div class="row legend">__LEGEND__</div>
  <div class="row note">※ 경로 배치는 정합 전 placeholder (FR-2.1에서 실제 정합)</div>
  <div class="row" id="frameinfo">frame —</div>
</div>
<div id="vidpanel"><video id="vid" src="__VIDEO__" muted playsinline preload="none"></video><div class="cap">원본 촬영영상 (촬영자 시점)</div></div>
<div id="bar"><button id="play">▶ 재생</button><input id="seek" type="range" min="0" max="1000" value="0"><span id="t">0.0s</span></div>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const MESHES=__MESHES__, POSES=__POSES__, META=__META__;
const renderer=new THREE.WebGLRenderer({canvas:document.getElementById('c'),antialias:true});
renderer.setSize(innerWidth,innerHeight); renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));
const scene=new THREE.Scene(); scene.background=new THREE.Color(0x0c0f16);
const camera=new THREE.PerspectiveCamera(55,innerWidth/innerHeight,0.05,5000);
scene.add(new THREE.AmbientLight(0xffffff,0.75));
const dl=new THREE.DirectionalLight(0xffffff,0.9); dl.position.set(30,60,30); scene.add(dl);
scene.add(new THREE.HemisphereLight(0xbfd4ff,0x202830,0.5));
const controls=new OrbitControls(camera,renderer.domElement);
function b64f32(s){const bin=atob(s);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return new Float32Array(u.buffer);}
const box=new THREE.Box3();
const MESH_OBJS=[];
for(const m of MESHES){
  const pos=b64f32(m.b64); const n=pos.length/3;
  const g=new THREE.BufferGeometry(); g.setAttribute('position',new THREE.BufferAttribute(pos,3)); g.computeVertexNormals();
  const base=new Float32Array(pos.length);
  for(let i=0;i<n;i++){base[i*3]=m.color[0];base[i*3+1]=m.color[1];base[i*3+2]=m.color[2];}
  const col=base.slice();
  g.setAttribute('color',new THREE.BufferAttribute(col,3));
  const mat=new THREE.MeshStandardMaterial({vertexColors:true,roughness:0.75,metalness:0.05,side:THREE.DoubleSide,transparent:true,opacity:0.92});
  const mesh=new THREE.Mesh(g,mat); scene.add(mesh); box.expandByObject(mesh);
  MESH_OBJS.push({mesh,pos,base,col,n});
}
const c0=box.getCenter(new THREE.Vector3()), sz=box.getSize(new THREE.Vector3());
// camera path + frustum
const pts=POSES.map(p=>new THREE.Vector3(p.c[0],p.c[1],p.c[2]));
if(pts.length){
  const lg=new THREE.BufferGeometry().setFromPoints(pts);
  scene.add(new THREE.Line(lg,new THREE.LineBasicMaterial({color:0xffb347,transparent:true,opacity:0.85})));
}
// frustum marker
const frustum=new THREE.Group();
const cone=new THREE.Mesh(new THREE.ConeGeometry(0.6,1.4,4),new THREE.MeshBasicMaterial({color:0x31d27c,wireframe:true}));
cone.rotation.x=Math.PI/2; frustum.add(cone);
const ball=new THREE.Mesh(new THREE.SphereGeometry(0.35,16,12),new THREE.MeshBasicMaterial({color:0x31d27c})); frustum.add(ball);
scene.add(frustum);
// view camera + frustum-based "what does this frame map to" highlight
const viewCam=new THREE.PerspectiveCamera(70,1.4,0.1,Math.max(2,Math.min(sz.x,sz.z)*0.25));
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
    for(let k=0;k<o.n;k++){
      _v.set(o.pos[k*3],o.pos[k*3+1],o.pos[k*3+2]);
      if(_fr.containsPoint(_v)){o.col[k*3]=0.2;o.col[k*3+1]=1.0;o.col[k*3+2]=0.55;hit++;}
      else{o.col[k*3]=o.base[k*3];o.col[k*3+1]=o.base[k*3+1];o.col[k*3+2]=o.base[k*3+2];}
    }
    o.mesh.geometry.attributes.color.needsUpdate=true;
  }
  document.getElementById('frameinfo').textContent='frame '+i+' / '+(POSES.length-1)+' · 매핑 정점 '+hit;
}
function setFrustum(i){
  i=Math.max(0,Math.min(POSES.length-1,i|0)); const p=POSES[i]; if(!p)return;
  frustum.position.set(p.c[0],p.c[1],p.c[2]);
  const fwd=new THREE.Vector3(p.f[0],p.f[1],p.f[2]).normalize();
  const tgt=new THREE.Vector3().addVectors(frustum.position,fwd);
  frustum.up.set(p.u[0],p.u[1],p.u[2]); frustum.lookAt(tgt);
  if(i!==lastHi){ lastHi=i; highlight(i); }
}
setFrustum(0);
// fit view
const d=Math.max(sz.x,sz.y,sz.z)*1.1;
camera.position.set(c0.x+d, c0.y+d*0.8, c0.z+d); controls.target.copy(c0); controls.update();
// video sync
const vid=document.getElementById('vid'), seek=document.getElementById('seek'), tlab=document.getElementById('t'), playb=document.getElementById('play');
let dur=Math.max(META.duration||1,0.1);
vid.addEventListener('loadedmetadata',()=>{dur=vid.duration||dur;});
playb.onclick=()=>{ if(vid.paused){vid.play();playb.textContent='⏸ 일시정지';} else {vid.pause();playb.textContent='▶ 재생';} };
seek.oninput=()=>{ const tt=(seek.value/1000)*dur; vid.currentTime=tt; };
function syncFromTime(t){ const frac=Math.max(0,Math.min(1,t/dur)); seek.value=Math.round(frac*1000); tlab.textContent=t.toFixed(1)+'s'; setFrustum(Math.round(frac*(POSES.length-1))); }
addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
function loop(){ requestAnimationFrame(loop); syncFromTime(vid.currentTime||0); controls.update(); renderer.render(scene,camera); }
loop();
window.__coplay={scene,camera,POSES,MESHES,box,setFrustum};
</script></body></html>"""


def fetch_poses(base_url, upload):
    url = f"{base_url.rstrip('/')}/api/uploads/{upload}/scan"
    with urllib.request.urlopen(url, timeout=60) as r:
        d = json.loads(r.read().decode("utf-8", "replace"))
    return d.get("poses") or []


def pose_vectors(poses):
    """c2w 12-float (row-major 3x4) -> center, forward(+Z), up(-Y) per pose."""
    out = []
    for p in poses:
        a = np.asarray(p, dtype=np.float64).reshape(3, 4)
        c = a[:, 3]
        fwd = a[:, 2]
        up = -a[:, 1]
        out.append((c, fwd, up))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8767")
    ap.add_argument("--upload", required=True)
    ap.add_argument("--dtdx", nargs="+", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--duration", type=float, default=35.3)
    ap.add_argument("--out", default="reports/coplay/coplay.html")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 1) model geometry (per color) from .dtdx
    meshes_json, tris, legend, names = [], 0, [], []
    model_pts = []
    for f in args.dtdx:
        g = decode_geometry(f)
        names.append(Path(f).name)
        tris += g["triangle_count"]
        for m in g["meshes"]:
            pos = np.asarray(m["positions"], dtype=np.float32)
            if not len(pos):
                continue
            model_pts.append(pos)
            meshes_json.append({"color": [round(c, 4) for c in m["color"]],
                                "b64": base64.b64encode(pos.tobytes()).decode("ascii")})
    allp = np.concatenate(model_pts) if model_pts else np.zeros((1, 3), np.float32)
    mn, mx = allp.min(0), allp.max(0)
    center, span = (mn + mx) / 2.0, (mx - mn)

    # 2) camera path — PLACEHOLDER fit into model bbox (pre-registration)
    poses = fetch_poses(args.base_url, args.upload)
    pv = pose_vectors(poses)
    pose_json = []
    if pv:
        centers = np.array([c for c, _, _ in pv])
        pmn, pmx = centers.min(0), centers.max(0)
        pext = float(np.linalg.norm(pmx - pmn)) or 1.0
        scale = 0.5 * float(min(span[0], span[2])) / pext  # fit XZ
        pcentroid = centers.mean(0)
        for c, fwd, up in pv:
            tc = (c - pcentroid) * scale + center
            pose_json.append({"c": [round(float(x), 3) for x in tc],
                              "f": [round(float(x), 4) for x in fwd],
                              "u": [round(float(x), 4) for x in up]})

    # color legend (top groups)
    for m in sorted(meshes_json, key=lambda x: -len(x["b64"]))[:5]:
        col = m["color"]
        hexc = "#%02x%02x%02x" % tuple(int(max(0, min(1, c)) * 255) for c in col[:3])
        legend.append(f'<i style="background:{hexc}"></i>')
    legend_html = " ".join(legend)

    rel_video = os.path.relpath(Path(args.video).resolve(), out.parent.resolve())
    html = (TEMPLATE
            .replace("__MESHES__", json.dumps(meshes_json))
            .replace("__POSES__", json.dumps(pose_json))
            .replace("__META__", json.dumps({"duration": args.duration}))
            .replace("__MODELNAME__", " + ".join(names))
            .replace("__TRIS__", f"{tris:,}")
            .replace("__NPOSES__", str(len(pose_json)))
            .replace("__LEGEND__", legend_html)
            .replace("__VIDEO__", rel_video))
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB) tris={tris:,} poses={len(pose_json)} video={rel_video}")


if __name__ == "__main__":
    main()

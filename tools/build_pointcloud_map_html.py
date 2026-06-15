"""Build a self-contained interactive 3D point-cloud MAP from reconstructed
uploads — the demo-style view: real video -> point cloud -> map with camera
trajectory. Uses coverage.html's proven coordinate transforms (point mirror
x,-y,-z and poseToMatrix) so orientation matches the product viewer. No GLB,
no alignment — pure reconstruction map.

Requires realtime/server.py running (fetches /api/uploads/{id}/scan).

Usage:
    .venv/bin/python tools/build_pointcloud_map_html.py \
        --base-url http://127.0.0.1:8767 \
        --uploads upload_1781508053560:cam1 upload_1781508087763:cam2 \
        --out reports/pointcloud_map/map.html --max-points 180000
"""
from __future__ import annotations
import argparse, base64, json, urllib.request
from pathlib import Path
import numpy as np


def fetch_scan(base, up, max_points):
    def get(variant):
        url = (f"{base}/api/uploads/{up}/scan?variant={variant}"
               f"&include_points=true&include_thumbs=false&max_points={max_points}")
        return json.load(urllib.request.urlopen(url, timeout=180))
    j = get("detail")
    if not j.get("ok") or not j.get("points"):
        j = get("mesh")
    return j


PALETTE = ["#5aa7ff", "#ff7eb6", "#7CFFB2", "#ffd166"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8767")
    ap.add_argument("--uploads", nargs="+", required=True, help="upload_id[:label] ...")
    ap.add_argument("--out", default="reports/pointcloud_map/map.html")
    ap.add_argument("--max-points", type=int, default=180000)
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    datasets = []
    for i, spec in enumerate(args.uploads):
        up, _, label = spec.partition(":")
        label = label or up
        print(f"fetching scan {up} ...")
        scan = fetch_scan(args.base_url, up, args.max_points)
        pts = np.asarray(scan.get("points") or [], np.float32)
        if pts.size == 0:
            print(f"  ! {up}: no points"); continue
        cols = np.asarray(scan.get("colors") or [], np.uint8)
        if cols.shape[0] != pts.shape[0]:
            cols = np.full((pts.shape[0], 3), 200, np.uint8)
        poses = scan.get("poses") or []
        print(f"  {up}: {pts.shape[0]} pts, {len(poses)} poses")
        datasets.append({
            "label": label, "up": up, "n": int(pts.shape[0]),
            "color": PALETTE[i % len(PALETTE)],
            "xyz": base64.b64encode(pts.tobytes()).decode("ascii"),
            "rgb": base64.b64encode(cols.tobytes()).decode("ascii"),
            "poses": poses,
        })
    if not datasets:
        print("no datasets"); return

    html = TEMPLATE.replace("__DATASETS__", json.dumps(datasets))
    out.write_text(html, encoding="utf-8")
    print(f"\nwrote {out}  ({out.stat().st_size/1e6:.1f} MB, {len(datasets)} datasets)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Point Cloud Map — lingbot-map</title>
<style>
  html,body{margin:0;height:100%;background:#0c0f16;color:#e6e8ee;font-family:system-ui,'Pretendard',sans-serif;overflow:hidden}
  #c{position:fixed;inset:0}
  #panel{position:fixed;top:12px;left:12px;background:rgba(18,22,32,.92);border:1px solid #2a3142;
    border-radius:12px;padding:14px 16px;font-size:13px;min-width:230px;backdrop-filter:blur(6px)}
  #panel h1{font-size:14px;margin:0 0 8px;font-weight:700;letter-spacing:.3px}
  #panel .row{display:flex;align-items:center;gap:8px;margin:6px 0}
  #panel label{cursor:pointer;user-select:none}
  #panel .dot{width:10px;height:10px;border-radius:3px;display:inline-block}
  #panel .muted{color:#8b93a7;font-size:11px}
  #panel hr{border:0;border-top:1px solid #2a3142;margin:10px 0}
  #panel button{background:#1d2433;color:#e6e8ee;border:1px solid #344056;border-radius:8px;
    padding:5px 9px;cursor:pointer;font-size:12px}
  #panel button:hover{background:#283142}
  input[type=range]{width:110px}
  #hint{position:fixed;bottom:10px;left:12px;color:#6b7488;font-size:11px}
</style>
<script type="importmap">{"imports":{
  "three":"https://unpkg.com/three@0.160.0/build/three.module.js",
  "three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"
}}</script></head>
<body>
<canvas id="c"></canvas>
<div id="panel">
  <h1>점군 지도 · point cloud map</h1>
  <div id="datasets"></div>
  <hr>
  <div class="row"><label><input type="checkbox" id="tgPts" checked> 점군 points</label></div>
  <div class="row"><label><input type="checkbox" id="tgPath" checked> 카메라 경로·방향</label></div>
  <div class="row">크기 <input type="range" id="size" min="1" max="60" value="18"></div>
  <div class="row"><button id="fit">전체 보기</button><button id="top">평면도(top)</button></div>
  <div class="muted" id="stat"></div>
</div>
<div id="hint">드래그=회전 · 휠=줌 · 우클릭드래그=이동</div>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const DATA = __DATASETS__;

const renderer = new THREE.WebGLRenderer({canvas:document.getElementById('c'),antialias:true});
renderer.setPixelRatio(devicePixelRatio); renderer.setSize(innerWidth,innerHeight);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0c0f16);
const camera = new THREE.PerspectiveCamera(55, innerWidth/innerHeight, 0.01, 5000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.AmbientLight(0xffffff,0.9));
const grid = new THREE.GridHelper(40,40,0x334,0x223); grid.material.opacity=0.25; grid.material.transparent=true; scene.add(grid);
scene.add(new THREE.AxesHelper(1.0));

function b64bytes(b){const s=atob(b);const u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return u;}

// coverage.html makePointCloud: mirror (x,-y,-z), real RGB
function makePointCloud(xyz,rgb,size){
  const n=xyz.length/3;
  const pos=new Float32Array(n*3), col=new Float32Array(n*3);
  for(let i=0;i<n;i++){
    pos[i*3]=xyz[i*3]; pos[i*3+1]=-xyz[i*3+1]; pos[i*3+2]=-xyz[i*3+2];
    col[i*3]=rgb[i*3]/255; col[i*3+1]=rgb[i*3+1]/255; col[i*3+2]=rgb[i*3+2]/255;
  }
  const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.BufferAttribute(pos,3));
  g.setAttribute('color',new THREE.BufferAttribute(col,3));
  const m=new THREE.PointsMaterial({size:size,vertexColors:true,sizeAttenuation:true,transparent:true,opacity:0.95,depthWrite:false});
  return new THREE.Points(g,m);
}
// coverage.html poseToMatrix (no alignment): mirror conjugation
function poseToMatrix(c){
  const R00=c[0],R01=-c[1],R02=-c[2],t0=c[3];
  const R10=-c[4],R11=c[5],R12=c[6],t1=-c[7];
  const R20=-c[8],R21=c[9],R22=c[10],t2=-c[11];
  const m=new THREE.Matrix4();
  m.set(R00,R01,R02,t0, R10,R11,R12,t1, R20,R21,R22,t2, 0,0,0,1);
  return {matrix:m, position:new THREE.Vector3(t0,t1,t2)};
}
function buildPath(poses,color){
  const grp=new THREE.Group();
  const fr=new THREE.LineBasicMaterial({color:color,transparent:true,opacity:0.5});
  const pts=[];
  const stride=Math.max(1,Math.floor(poses.length/45));
  for(let i=0;i<poses.length;i++){
    const {matrix,position}=poseToMatrix(poses[i]); pts.push(position.clone());
    if(i%stride!==0 && i!==poses.length-1) continue;
    const s=0.18,w=0.13,h=0.16;
    const v=new Float32Array([0,0,0,w,h,-s,0,0,0,-w,h,-s,0,0,0,w,-h,-s,0,0,0,-w,-h,-s,w,h,-s,w,-h,-s,w,-h,-s,-w,-h,-s,-w,-h,-s,-w,h,-s,-w,h,-s,w,h,-s]);
    const fg=new THREE.BufferGeometry(); fg.setAttribute('position',new THREE.BufferAttribute(v,3));
    const ln=new THREE.LineSegments(fg,fr); ln.applyMatrix4(matrix); grp.add(ln);
  }
  const lg=new THREE.BufferGeometry().setFromPoints(pts);
  grp.add(new THREE.Line(lg,new THREE.LineBasicMaterial({color:0xffb347,transparent:true,opacity:0.85})));
  if(pts.length){
    const start=new THREE.Mesh(new THREE.SphereGeometry(0.09,16,10),new THREE.MeshBasicMaterial({color:0x31d27c}));
    start.position.copy(pts[0]);
    const end=new THREE.Mesh(new THREE.BoxGeometry(0.15,0.15,0.15),new THREE.MeshBasicMaterial({color:0xff5f6d}));
    end.position.copy(pts[pts.length-1]); grp.add(start,end);
  }
  return grp;
}

const pointGroups=[], pathGroups=[];
const box=new THREE.Box3();
const dsDiv=document.getElementById('datasets');
let totalPts=0;
DATA.forEach((d,i)=>{
  const xyz=new Float32Array(b64bytes(d.xyz).buffer);
  const rgb=b64bytes(d.rgb);
  const pc=makePointCloud(xyz,rgb,0.018); scene.add(pc); pointGroups.push(pc);
  const pg=buildPath(d.poses||[],new THREE.Color(d.color).getHex()); scene.add(pg); pathGroups.push(pg);
  box.expandByObject(pc); totalPts+=d.n;
  const row=document.createElement('div'); row.className='row';
  row.innerHTML=`<input type="checkbox" data-i="${i}" class="dsToggle" checked>`+
    `<span class="dot" style="background:${d.color}"></span>`+
    `<label>${d.label} · ${d.n.toLocaleString()} pts</label>`;
  dsDiv.appendChild(row);
});
document.getElementById('stat').textContent=`${DATA.length} scans · ${totalPts.toLocaleString()} points`;

function fit(){
  const c=box.getCenter(new THREE.Vector3()), s=box.getSize(new THREE.Vector3());
  const r=Math.max(s.x,s.y,s.z)||4;
  controls.target.copy(c);
  camera.position.set(c.x+r*1.1, c.y+r*0.8, c.z+r*1.3);
  camera.near=r/100; camera.far=r*50; camera.updateProjectionMatrix(); controls.update();
}
function top(){
  const c=box.getCenter(new THREE.Vector3()), s=box.getSize(new THREE.Vector3());
  const r=Math.max(s.x,s.y,s.z)||4;
  controls.target.copy(c); camera.up.set(0,0,-1);
  camera.position.set(c.x, c.y+r*1.6, c.z+0.001); controls.update();
}
fit();

document.querySelectorAll('.dsToggle').forEach(cb=>cb.onchange=e=>{
  const i=+e.target.dataset.i; pointGroups[i].visible=e.target.checked&&document.getElementById('tgPts').checked;
  pathGroups[i].visible=e.target.checked&&document.getElementById('tgPath').checked;
});
document.getElementById('tgPts').onchange=e=>pointGroups.forEach((g,i)=>{const cb=document.querySelector(`.dsToggle[data-i="${i}"]`);g.visible=e.target.checked&&cb.checked;});
document.getElementById('tgPath').onchange=e=>pathGroups.forEach((g,i)=>{const cb=document.querySelector(`.dsToggle[data-i="${i}"]`);g.visible=e.target.checked&&cb.checked;});
document.getElementById('size').oninput=e=>{const s=e.target.value/1000;pointGroups.forEach(g=>{g.material.size=s;});};
document.getElementById('fit').onclick=()=>{camera.up.set(0,1,0);fit();};
document.getElementById('top').onclick=top;

addEventListener('resize',()=>{renderer.setSize(innerWidth,innerHeight);camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();});
(function loop(){requestAnimationFrame(loop);controls.update();renderer.render(scene,camera);})();
window.__map={scene,camera,controls,box,pointGroups,pathGroups,fit,top};
</script></body></html>
"""


if __name__ == "__main__":
    main()

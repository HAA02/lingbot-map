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
<title>co-play · dtdx ↔ 점군영상</title>
<style>
*{margin:0;box-sizing:border-box} html,body{height:100%;background:#0a0d13;color:#e8eaf0;font-family:system-ui,sans-serif;overflow:hidden}
#c{position:fixed;inset:0;display:block}
#hud{position:fixed;left:12px;top:12px;background:rgba(10,13,19,.8);border:1px solid #283042;border-radius:8px;padding:8px 11px;font-size:12px;z-index:5;max-width:360px}
#hud b{color:#fff} #hud .legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px} .hl{color:#31d27c}
#pip{position:fixed;right:18px;top:18px;width:360px;height:235px;background:#11151f;border:1px solid #2c3344;border-radius:10px;overflow:hidden;z-index:20;box-shadow:0 8px 30px rgba(0,0,0,.6);display:flex;flex-direction:column;min-width:160px;min-height:110px;max-width:92vw;max-height:88vh}
#pip header{display:flex;align-items:center;gap:6px;padding:5px 8px;background:#161b26;cursor:move;font-size:12px;color:#cdd6e6;user-select:none}
#pip header .sp{flex:1} #pip header button{background:#222b3a;color:#cdd6e6;border:1px solid #313c4f;border-radius:5px;padding:0 8px;cursor:pointer;font-size:13px;line-height:18px}
#pip video{width:100%;flex:1;min-height:0;display:block;object-fit:contain;background:#000} #pip.min{height:auto!important} #pip.min video{display:none}
#pip .rsz{position:absolute;z-index:31}
#pip .rsz.n{top:-3px;left:10px;right:10px;height:8px;cursor:ns-resize}
#pip .rsz.s{bottom:-3px;left:10px;right:10px;height:8px;cursor:ns-resize}
#pip .rsz.e{right:-3px;top:10px;bottom:10px;width:8px;cursor:ew-resize}
#pip .rsz.w{left:-3px;top:10px;bottom:10px;width:8px;cursor:ew-resize}
#pip .rsz.ne{top:-3px;right:-3px;width:14px;height:14px;cursor:nesw-resize}
#pip .rsz.nw{top:-3px;left:-3px;width:14px;height:14px;cursor:nwse-resize}
#pip .rsz.se{bottom:-3px;right:-3px;width:14px;height:14px;cursor:nwse-resize}
#pip .rsz.sw{bottom:-3px;left:-3px;width:14px;height:14px;cursor:nesw-resize}
#bar{position:fixed;left:12px;right:12px;bottom:12px;display:flex;gap:10px;align-items:center;background:rgba(14,18,25,.92);border:1px solid #283042;border-radius:10px;padding:8px 14px;z-index:10}
#bar button{background:#1c2433;color:#e8eaf0;border:1px solid #2c3344;border-radius:6px;padding:6px 12px;cursor:pointer;font-size:13px}
#bar button.on{background:#31d27c;color:#06210f;border-color:#31d27c}
#bar input[type=range]{flex:1} #bar .t{color:#9fb0c8;min-width:60px;font-variant-numeric:tabular-nums}
#place{position:fixed;left:12px;bottom:60px;background:rgba(10,13,19,.85);border:1px solid #31d27c;border-radius:8px;padding:7px 10px;font-size:12px;color:#9fb0c8;z-index:10;display:none}
#place b{color:#e8eaf0}
</style>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
</head><body>
<canvas id="c"></canvas>
<div id="hud"><b>설계모델 ↔ 영상 co-play</b> <span id="pmode" style="font-weight:700;padding:2px 8px;border-radius:6px;background:__PMODECOL__;color:#0a0d13">__PMODE__</span><br><span style="color:#9fb0c8">__MODELNAME__ · 삼각형 __TRIS__ · 포즈 __NPOSES__</span><br><span class="legend">__LEGEND__</span><br><span id="finfo" style="color:#9fb0c8">frame —</span></div>
<div id="pip"><header><span>정합 렌더 영상 (점군)</span><span class="sp"></span><button id="pmin" title="최소화">▭</button></header><video id="rvid" src="__RVIDEO__" muted playsinline preload="none"></video><div class="rsz n"></div><div class="rsz s"></div><div class="rsz e"></div><div class="rsz w"></div><div class="rsz ne"></div><div class="rsz nw"></div><div class="rsz se"></div><div class="rsz sw"></div></div>
<div id="place">모델 클릭=첫 위치 · <b>화살표</b>이동 · <b>[</b>/<b>]</b>회전 · <b>,</b>/<b>.</b>스케일 · <b>m</b>좌우반전 · <b>PgUp/Dn</b>높이<br><span id="pp" style="color:#cdd6e6"></span></div>
<div id="bar"><button id="play">▶ 재생</button><button id="placeBtn">위치 지정</button><button id="fcam">촬영자 추적</button><button id="peer" style="display:__PEERSHOW__;background:__PEERCOL__;color:#0a0d13;font-weight:700" onclick="location.href='__PEERURL__'">__PEERLABEL__</button><input id="seek" type="range" min="0" max="1000" value="0"><span class="t" id="t">0.0s</span></div>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const MESHES=__MESHES__, RAWP=__POSES__, META=__META__;
const renderer=new THREE.WebGLRenderer({canvas:document.getElementById('c'),antialias:true});
renderer.setSize(innerWidth,innerHeight); renderer.setPixelRatio(Math.min(devicePixelRatio,2));
const scene=new THREE.Scene(); scene.background=new THREE.Color(0x0c0f16);
const camera=new THREE.PerspectiveCamera(55,innerWidth/innerHeight,0.05,2000);
scene.add(new THREE.AmbientLight(0xffffff,0.85));
const dl=new THREE.DirectionalLight(0xffffff,0.9); dl.position.set(30,60,30); scene.add(dl);
scene.add(new THREE.HemisphereLight(0xbfd4ff,0x202830,0.5)); scene.add(new THREE.GridHelper(60,60,0x223,0x182030));
const controls=new OrbitControls(camera,renderer.domElement);
function b64f32(s){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Float32Array(u.buffer);}
const box=new THREE.Box3(); const MESH_OBJS=[]; const raycaster=new THREE.Raycaster(); const mouse=new THREE.Vector2();
for(const m of MESHES){
  const pos=b64f32(m.b64); const n=pos.length/3;
  const g=new THREE.BufferGeometry(); g.setAttribute('position',new THREE.BufferAttribute(pos,3)); g.computeVertexNormals();
  const base=new Float32Array(pos.length); for(let i=0;i<n;i++){base[i*3]=m.color[0];base[i*3+1]=m.color[1];base[i*3+2]=m.color[2];}
  const col=base.slice(); g.setAttribute('color',new THREE.BufferAttribute(col,3));
  const mesh=new THREE.Mesh(g,new THREE.MeshStandardMaterial({vertexColors:true,roughness:0.7,metalness:0.05,side:THREE.DoubleSide}));
  scene.add(mesh); box.expandByObject(mesh); MESH_OBJS.push({mesh,pos,base,col,n});
}
const c0=box.getCenter(new THREE.Vector3()), sz=box.getSize(new THREE.Vector3());
let cx0=0,cy0=0,cz0=0; for(const p of RAWP){cx0+=p.c[0];cy0+=p.c[1];cz0+=p.c[2];} cx0/=RAWP.length;cy0/=RAWP.length;cz0/=RAWP.length;
const CANON=RAWP.map(p=>({c:[p.c[0]-cx0,p.c[1]-cy0,p.c[2]-cz0],f:p.f.slice(),u:p.u.slice()}));
let offset=[cx0,cy0,cz0], yaw=0, pscale=1, flipX=1;
function rotY(v,deg){const r=deg*Math.PI/180,c=Math.cos(r),s=Math.sin(r);return [c*v[0]+s*v[2],v[1],-s*v[0]+c*v[2]];}
function wp(i){const q=CANON[i];const cc=[q.c[0]*flipX*pscale,q.c[1]*pscale,q.c[2]*pscale];const c=rotY(cc,yaw);return {c:[c[0]+offset[0],c[1]+offset[1],c[2]+offset[2]],f:rotY([q.f[0]*flipX,q.f[1],q.f[2]],yaw),u:rotY([q.u[0]*flipX,q.u[1],q.u[2]],yaw)};}
const pathGeo=new THREE.BufferGeometry(); const pathPos=new Float32Array(RAWP.length*3);
// 경로: depth-test 끄고 항상 위에 그려 객체 뒤에 있어도 보이게 (오버라이드)
const pathLine=new THREE.Line(pathGeo,new THREE.LineBasicMaterial({color:0xffc04a,depthTest:false,depthWrite:false,transparent:true}));pathLine.renderOrder=990;scene.add(pathLine);
const pathDots=new THREE.Points(pathGeo,new THREE.PointsMaterial({color:0xffe08a,size:5,sizeAttenuation:false,depthTest:false,depthWrite:false,transparent:true}));pathDots.renderOrder=991;scene.add(pathDots);
function rebuildPath(){ for(let i=0;i<RAWP.length;i++){const w=wp(i);pathPos[i*3]=w.c[0];pathPos[i*3+1]=w.c[1];pathPos[i*3+2]=w.c[2];} pathGeo.setAttribute('position',new THREE.BufferAttribute(pathPos,3)); pathGeo.attributes.position.needsUpdate=true; pathGeo.computeBoundingSphere(); const _pp=document.getElementById("pp"); if(_pp)_pp.textContent="위치("+offset[0].toFixed(1)+", "+offset[2].toFixed(1)+") yaw "+yaw.toFixed(0)+"° 스케일 "+pscale.toFixed(2)+(flipX<0?" ⇄반전":""); }
// 카메라 마커: 방향 콘(초록) + 위치 구(빨강), depth-test 끄고 항상 위에 (객체 뒤에서도 보임)
const frustum=new THREE.Group();
const cone=new THREE.Mesh(new THREE.ConeGeometry(0.6,1.6,4),new THREE.MeshBasicMaterial({color:0x31d27c,depthTest:false,depthWrite:false,transparent:true}));cone.rotation.x=Math.PI/2;frustum.add(cone);
frustum.add(new THREE.Mesh(new THREE.SphereGeometry(0.42,18,14),new THREE.MeshBasicMaterial({color:0xff4466,depthTest:false,depthWrite:false,transparent:true})));
frustum.traverse(o=>{o.renderOrder=999;});scene.add(frustum);
// 수직 빔(beacon): 현재 카메라 위치를 바닥~천장 관통하는 기둥으로 표시 → 3D에서 위치 즉시 파악
const beacon=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,-sz.y,0),new THREE.Vector3(0,sz.y,0)]),new THREE.LineBasicMaterial({color:0xff6688,depthTest:false,depthWrite:false,transparent:true,opacity:0.55}));beacon.renderOrder=995;scene.add(beacon);
const viewCam=new THREE.PerspectiveCamera(70,1.5,0.1,Math.max(2,Math.min(sz.x,sz.z)*0.3));
const _fr=new THREE.Frustum(),_m4=new THREE.Matrix4(),_v=new THREE.Vector3(); let lastHi=-1,dirty=true,curFrame=0;
function highlight(w){ viewCam.position.set(w.c[0],w.c[1],w.c[2]);viewCam.up.set(w.u[0],w.u[1],w.u[2]);viewCam.lookAt(w.c[0]+w.f[0],w.c[1]+w.f[1],w.c[2]+w.f[2]);
  viewCam.updateMatrixWorld(true);viewCam.updateProjectionMatrix();_fr.setFromProjectionMatrix(_m4.multiplyMatrices(viewCam.projectionMatrix,viewCam.matrixWorldInverse));
  let hit=0; for(const o of MESH_OBJS){for(let k=0;k<o.n;k++){_v.set(o.pos[k*3],o.pos[k*3+1],o.pos[k*3+2]); if(_fr.containsPoint(_v)){o.col[k*3]=0.2;o.col[k*3+1]=1;o.col[k*3+2]=0.55;hit++;}else{o.col[k*3]=o.base[k*3];o.col[k*3+1]=o.base[k*3+1];o.col[k*3+2]=o.base[k*3+2];}}o.mesh.geometry.attributes.color.needsUpdate=true;}
  document.getElementById('finfo').innerHTML='frame '+curFrame+' / '+(RAWP.length-1)+' · 매핑 <span class="hl">'+hit+'</span>';
}
let followCam=false; const fbtn=document.getElementById('fcam'), placeBtn=document.getElementById('placeBtn'), placeHud=document.getElementById('place');
let placeMode=false;
function updCtl(){ controls.enabled = placeMode || !followCam; if(typeof frustum!=='undefined'){frustum.visible=!followCam; beacon.visible=!followCam;} }
const _fe=new THREE.Vector3(),_fl=new THREE.Vector3(),_fu=new THREE.Vector3(0,1,0),_lt=new THREE.Vector3(); let _finit=false;
function applyFollow(w){const f=new THREE.Vector3(w.f[0],w.f[1],w.f[2]).normalize(),u=new THREE.Vector3(w.u[0],w.u[1],w.u[2]).normalize();
  _fe.set(w.c[0],w.c[1],w.c[2]).addScaledVector(f,-2.2).addScaledVector(u,0.7); _fl.set(w.c[0]+f.x*2,w.c[1]+f.y*2,w.c[2]+f.z*2); _fu.copy(u);
  if(!_finit){camera.position.copy(_fe);_lt.copy(_fl);_finit=true;}}
function setFrame(i){ i=Math.max(0,Math.min(RAWP.length-1,i|0)); curFrame=i; const w=wp(i);
  frustum.position.set(w.c[0],w.c[1],w.c[2]);frustum.up.set(w.u[0],w.u[1],w.u[2]);frustum.lookAt(w.c[0]+w.f[0],w.c[1]+w.f[1],w.c[2]+w.f[2]);
  beacon.position.set(w.c[0],c0.y,w.c[2]);
  if(i!==lastHi||dirty){lastHi=i;dirty=false;highlight(w);} if(followCam&&!placeMode)applyFollow(w); }
fbtn.onclick=()=>{followCam=!followCam;fbtn.classList.toggle('on',followCam);if(followCam)_finit=false;frustum.visible=!followCam;beacon.visible=!followCam;updCtl();};
placeBtn.onclick=()=>{placeMode=!placeMode;placeBtn.classList.toggle('on',placeMode);placeHud.style.display=placeMode?'block':'none';updCtl();};
updCtl(); rebuildPath(); setFrame(0);
// DTDWebThree 기본뷰와 동일 방향: +X+Y+Z 코너 45°H/30°V, up=+Y (반전 인상 제거)
{const _md=Math.max(sz.x,sz.y,sz.z),_fv=camera.fov*Math.PI/180,_d=Math.abs(_md/Math.sin(_fv/2))*0.85;
 camera.position.set(c0.x+_d*0.612,c0.y+_d*0.5,c0.z+_d*0.612); camera.up.set(0,1,0);}
controls.target.copy(c0); controls.update();
renderer.domElement.addEventListener('click',e=>{ if(!placeMode)return;
  const rc=renderer.domElement.getBoundingClientRect();mouse.x=((e.clientX-rc.left)/rc.width)*2-1;mouse.y=-((e.clientY-rc.top)/rc.height)*2+1;raycaster.setFromCamera(mouse,camera);
  const hits=raycaster.intersectObjects(MESH_OBJS.map(o=>o.mesh),false);
  if(hits.length){offset[0]=hits[0].point.x;offset[2]=hits[0].point.z;dirty=true;rebuildPath();} });
addEventListener('keydown',e=>{ if(!placeMode)return; const st=0.2;
  if(e.key==='ArrowLeft')offset[0]-=st; else if(e.key==='ArrowRight')offset[0]+=st;
  else if(e.key==='ArrowUp')offset[2]-=st; else if(e.key==='ArrowDown')offset[2]+=st;
  else if(e.key==='PageUp')offset[1]+=st; else if(e.key==='PageDown')offset[1]-=st;
  else if(e.key==='[')yaw-=3; else if(e.key===']')yaw+=3; else if(e.key===',')pscale=Math.max(0.2,pscale/1.08); else if(e.key==='.')pscale*=1.08; else if(e.key==='m'||e.key==='M')flipX*=-1; else return;
  e.preventDefault();dirty=true;rebuildPath(); });
const pip=document.getElementById('pip'); const head=pip.querySelector('header'); let drag=null;
head.addEventListener('mousedown',e=>{ if(e.target.tagName==='BUTTON')return; drag={x:e.clientX-pip.offsetLeft,y:e.clientY-pip.offsetTop}; e.preventDefault(); });
let rsz=null;
document.querySelectorAll('#pip .rsz').forEach(h=>h.addEventListener('mousedown',e=>{ rsz={dir:h.classList[1],x:e.clientX,y:e.clientY,w:pip.offsetWidth,h:pip.offsetHeight,l:pip.offsetLeft,t:pip.offsetTop}; e.preventDefault(); e.stopPropagation(); }));
addEventListener('mousemove',e=>{ if(drag){pip.style.left=(e.clientX-drag.x)+'px';pip.style.top=(e.clientY-drag.y)+'px';pip.style.right='auto';}
  else if(rsz){const dx=e.clientX-rsz.x,dy=e.clientY-rsz.y,d=rsz.dir; let w=rsz.w,h=rsz.h,l=rsz.l,t=rsz.t;
   if(d.indexOf('e')>=0)w=rsz.w+dx; if(d.indexOf('w')>=0)w=rsz.w-dx; if(d.indexOf('s')>=0)h=rsz.h+dy; if(d.indexOf('n')>=0)h=rsz.h-dy;
   w=Math.max(160,Math.min(innerWidth*0.92,w)); h=Math.max(110,Math.min(innerHeight*0.88,h));
   if(d.indexOf('w')>=0)l=rsz.l+(rsz.w-w); if(d.indexOf('n')>=0)t=rsz.t+(rsz.h-h);
   pip.style.width=w+'px';pip.style.height=h+'px';pip.style.left=l+'px';pip.style.top=t+'px';pip.style.right='auto';} });
addEventListener('mouseup',()=>{drag=null;rsz=null;});
document.getElementById('pmin').onclick=()=>pip.classList.toggle('min');
const rvid=document.getElementById('rvid'),seek=document.getElementById('seek'),tlab=document.getElementById('t'),playb=document.getElementById('play');
let dur=Math.max(META.duration||1,0.1); rvid.addEventListener('loadedmetadata',()=>{dur=rvid.duration||dur;});
playb.onclick=()=>{ if(rvid.paused){rvid.play();playb.textContent='⏸ 일시정지';}else{rvid.pause();playb.textContent='▶ 재생';} };
seek.oninput=()=>{rvid.currentTime=(seek.value/1000)*dur;};
function sync(t){const fr=Math.max(0,Math.min(1,t/dur));seek.value=Math.round(fr*1000);tlab.textContent=t.toFixed(1)+'s';setFrame(Math.round(fr*(RAWP.length-1)));}
addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
function loop(){requestAnimationFrame(loop);sync(rvid.currentTime||0);
  if(followCam&&!placeMode){camera.position.lerp(_fe,0.12);_lt.lerp(_fl,0.12);camera.up.lerp(_fu,0.12);camera.lookAt(_lt);}
  else if(!placeMode){controls.update();}
  renderer.render(scene,camera);}
loop();
window.__coplay={scene,camera,box,RAWP,setFrame,setFollow:v=>{followCam=!!v;updCtl();}};
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
    """재구성 외부파라미터 [R|t]는 WORLD-TO-CAMERA(여기선 viewer Y-up로 플립).
    카메라 중심 = -R^T t (t를 그대로 쓰면 회전이 병진으로 새어들어 회전 시 ~4배
    유령 가속 — 실측 corr(회전,병진) 0.985→0.12로 해소). forward=-R^T col2(걷는
    방향과 정합 +0.64), up=R^T col1(중력 +Y, 일관도 0.93). 반환 (center, forward, up)."""
    c = np.asarray(p12, dtype=np.float64)
    R = np.array([[c[0], -c[1], -c[2]], [-c[4], c[5], c[6]], [-c[8], c[9], c[10]]])
    t = np.array([c[3], -c[7], -c[11]])
    Rt = R.T
    return -Rt @ t, -Rt[:, 2], Rt[:, 1]


def _rot_a_to_b(a, b):
    a = a / (np.linalg.norm(a) + 1e-12); b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b); c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def _icp_rigid(src, dst, tree, s, R, t, iters=30):
    """Rigid ICP at FIXED scale s — refine R,t only (Kabsch). Scale comes from the
    robust vertical/ceiling-height anchor, NOT horizontal ceiling matching which is
    ambiguous on repeated ceilings (it collapsed to 0.409). apply = s*(src@R.T)+t."""
    S = s * src
    for _ in range(iters):
        d, idx = tree.query(S @ R.T + t, workers=-1)
        thr = max(float(np.median(d)) * 3.0, 1e-6); m = d < thr
        if int(m.sum()) < 3:
            break
        A, B = S[m], dst[idx[m]]
        ca, cb = A.mean(0), B.mean(0)
        U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
        Dd = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(Vt.T @ U.T)))])
        Rn = Vt.T @ Dd @ U.T
        tn = cb - ca @ Rn.T
        done = np.allclose(Rn, R, atol=1e-9) and np.allclose(tn, t, atol=1e-9)
        R, t = Rn, tn
        if done:
            break
    d, _ = tree.query(S @ R.T + t, workers=-1)
    rmse = float(np.sqrt((d ** 2).mean()))
    inl = float((d < max(float(np.median(d)) * 3.0, 0.05)).mean())
    return s, R, t, rmse, inl


def place_gravity(poses, scan_pts, bbox, scale=1.0):
    centers, fwd, up = [], [], []
    for p in poses:
        center, f, u = viewer_pose(p); centers.append(center); up.append(u); fwd.append(f)
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


def place_registered(poses, scan_pts, model_ceiling, bbox, anchor=None):
    """Real registration: gravity-align scan, then register its CEILING band to
    the model ceiling (grid XY + yaw + scale + Umeyama-ICP). Returns placed
    poses + fit metrics. (Footage looks up → ceiling-to-ceiling locks well.)"""
    from scipy.spatial import cKDTree
    centers, fwd, up = [], [], []
    for p in poses:
        center, f, u = viewer_pose(p); centers.append(center); up.append(u); fwd.append(f)
    centers = np.array(centers); up = np.array(up); fwd = np.array(fwd)
    g = up.mean(0); g /= (np.linalg.norm(g) + 1e-9)
    Rg = _rot_a_to_b(g, np.array([0.0, 1.0, 0.0]))
    P = scan_pts.copy(); P[:, 1] *= -1.0; P[:, 2] *= -1.0; Pg = P @ Rg.T
    Cg = centers @ Rg.T; Fg = fwd @ Rg.T; Ug = up @ Rg.T
    # 모델은 X반전(미러)됨 — 스캔/궤적도 같은 chirality로 수평 반사해야 좌/우 회전과
    # 복도가 일치. proper rotation만으론 거울차이를 못 메움(평행배관 천장은 대칭이라
    # inlier는 높아도 궤적 회전이 뒤집힘 → "영상 좌회전=모델 우회전"+벽 통과).
    Pg[:, 0] *= -1.0; Cg[:, 0] *= -1.0; Fg[:, 0] *= -1.0; Ug[:, 0] *= -1.0
    Sc = Pg[Pg[:, 1] >= np.percentile(Pg[:, 1], 55)]
    sc = Sc[np.linspace(0, len(Sc) - 1, min(2500, len(Sc))).astype(int)]; scan_c = sc.mean(0)
    tree = cKDTree(model_ceiling)
    lo, hi = np.array(bbox[0]), np.array(bbox[1])
    yc = float(np.percentile(model_ceiling[:, 1], 50))

    # METRIC scale from the VERTICAL ceiling-height ratio — robust because looking
    # up captures floor↔ceiling extent well, whereas horizontal ceiling matching is
    # ambiguous on repeated geometry (it collapsed to 0.409 → 3.6 m phantom walk).
    vext = float(np.percentile(Pg[:, 1], 97) - np.percentile(Pg[:, 1], 3))
    s_vert = float((hi[1] - lo[1]) / max(vext, 1e-6))

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

    # coarse prior (1-click 첫 위치): restrict the search to ±3.5m around it →
    # 반복배관/노이즈와 무관하게 그 방에서 국소 정합(보장).
    if anchor is not None:
        ax, az = float(anchor[0]), float(anchor[1])
        gx = np.linspace(ax - 3.5, ax + 3.5, 7)
        gz = np.linspace(az - 3.5, az + 3.5, 7)
    else:
        gx = np.linspace(lo[0] + 2, hi[0] - 2, 7)
        gz = np.linspace(lo[2] + 2, hi[2] - 2, 9)

    best = None
    for s in (s_vert * 0.92, s_vert, s_vert * 1.08):   # narrow band around metric anchor
        for yaw in range(0, 360, 15):
            R = Ry(yaw)
            align = abs(float(np.cos(np.deg2rad(a_s + yaw - a_p))))  # 1=traj∥pipes
            for cx in gx:
                for cz in gz:
                    t = np.array([cx, yc, cz]) - s * (R @ scan_c)
                    d, _ = tree.query(s * (sc @ R.T) + t, workers=-1)
                    score = float((d < 0.25).mean()) + 0.25 * align
                    if best is None or score > best[0]:
                        best = (score, s, yaw, R, t)
    _sc, s0, yaw, R0, t0 = best
    s2, R2, t2, rmse, inl2 = _icp_rigid(sc, model_ceiling, tree, float(s0), R0, t0, iters=30)
    Ct = s2 * (Cg @ R2.T) + t2; Ft = Fg @ R2.T; Ut = Ug @ R2.T
    # 카메라 EYE-LEVEL 보정: 천장정합은 점군 천장을 맞추지만 카메라는 천장이 아니라
    # 눈높이에 있음. 재구성의 카메라↔천장 거리가 압축돼 천장 플레넘에 박힘(바닥+2.8m).
    # 궤적 형태는 두고 높이만 바닥+1.5m(눈높이)로 시프트 → 천장구조 관통 제거.
    eye = float(bbox[0][1]) + 1.5
    Ct[:, 1] += eye - float(np.median(Ct[:, 1]))
    pose_json = [{"c": [round(float(x), 3) for x in Ct[i]], "f": [round(float(x), 4) for x in Ft[i]],
                  "u": [round(float(x), 4) for x in Ut[i]]} for i in range(len(Ct))]
    return pose_json, {"inlier": round(float(inl2), 3), "rmse": round(float(rmse), 3),
                       "scale": round(float(s2), 3), "yaw": int(yaw),
                       "cam_h": round(eye - float(bbox[0][1]), 2)}


def _resample(pts, n):
    pts = np.asarray(pts, float)
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    if d[-1] < 1e-9:
        return np.repeat(pts[:1], n, axis=0)
    u = np.linspace(0, d[-1], n)
    return np.column_stack([np.interp(u, d, pts[:, k]) for k in range(pts.shape[1])])


def _umeyama2d(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    Xs, Xd = src - ms, dst - md
    U, D, Vt = np.linalg.svd((Xd.T @ Xs) / len(src))
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    s = float((D * np.diag(S)).sum() / ((Xs ** 2).sum() / len(src)))
    return s, R, md - s * (R @ ms)   # dst ≈ s*(src@R.T)+t


def place_gtpath(poses, scan_pts, bbox, waypoints, snap=True):
    """Place the camera trajectory using a USER-GIVEN ground-truth path (model XZ
    waypoints). Ceiling auto-registration is ambiguous on parallel pipe columns;
    the drawn path resolves it.

    snap=True (default): WARP each pose onto the GT polyline by arc-length fraction
    → a straight walk renders straight (monocular drift bends it otherwise). Camera
    look-direction = GT tangent rotated by the recon's forward-vs-motion offset
    (keeps look-up/pan). snap=False: rigid Umeyama-2D fit (keeps recon curve).
    Vertical = eye level. [[alignment-monocular-scale]]"""
    centers, fwd, up = [], [], []
    for p in poses:
        c, f, u = viewer_pose(p); centers.append(c); up.append(u); fwd.append(f)
    centers = np.array(centers); up = np.array(up); fwd = np.array(fwd)
    g = up.mean(0); g /= (np.linalg.norm(g) + 1e-9)
    Rg = _rot_a_to_b(g, np.array([0.0, 1.0, 0.0]))
    Cg = centers @ Rg.T; Fg = fwd @ Rg.T; Ug = up @ Rg.T
    Cg[:, 0] *= -1; Fg[:, 0] *= -1; Ug[:, 0] *= -1   # chirality (model is X-mirrored)
    gt = np.asarray(waypoints, float)
    traj = Cg[:, [0, 2]]

    if snap:
        d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(traj, axis=0), axis=1))])
        gd = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(gt, axis=0), axis=1))])
        gtot = float(gd[-1]); pos_s = (d / max(d[-1], 1e-9)) * gtot         # arc-length fraction → GT length
        XZ = np.column_stack([np.interp(pos_s, gd, gt[:, 0]), np.interp(pos_s, gd, gt[:, 1])])
        ah = np.column_stack([np.interp(np.minimum(pos_s + .3, gtot), gd, gt[:, 0]), np.interp(np.minimum(pos_s + .3, gtot), gd, gt[:, 1])])
        bh = np.column_stack([np.interp(np.maximum(pos_s - .3, 0), gd, gt[:, 0]), np.interp(np.maximum(pos_s - .3, 0), gd, gt[:, 1])])
        gt_ang = np.arctan2((ah - bh)[:, 1], (ah - bh)[:, 0])
        vel = np.vstack([traj[1] - traj[0], np.diff(traj, axis=0)])
        ker = np.ones(9) / 9.0                                              # smooth recon motion dir (drift jitter)
        mv_ang = np.arctan2(np.convolve(vel[:, 1], ker, "same"), np.convolve(vel[:, 0], ker, "same"))
        rot = gt_ang - mv_ang; cr, sr = np.cos(rot), np.sin(rot)            # per-pose yaw onto GT tangent
        Fxz = np.column_stack([cr * Fg[:, 0] - sr * Fg[:, 2], sr * Fg[:, 0] + cr * Fg[:, 2]])
        Uxz = np.column_stack([cr * Ug[:, 0] - sr * Ug[:, 2], sr * Ug[:, 0] + cr * Ug[:, 2]])
        s = gtot / max(d[-1], 1e-9)
    else:
        s, R2, t2 = _umeyama2d(_resample(traj, 120), _resample(gt, 120))
        XZ = s * (traj @ R2.T) + t2
        Fxz = Fg[:, [0, 2]] @ R2.T; Uxz = Ug[:, [0, 2]] @ R2.T

    eye = float(bbox[0][1]) + 1.5
    # 눈높이 + 소량 bob. 수직변동을 수평스케일 s로 곱하면(자동 s=5↑) 비현실적으로 출렁 →
    # ±0.2m로 클램프해 높이를 자동/수동 일관·현실화.
    Y = eye + np.clip((Cg[:, 1] - np.median(Cg[:, 1])) * s, -0.2, 0.2)
    pose_json = [{"c": [round(float(XZ[i, 0]), 3), round(float(Y[i]), 3), round(float(XZ[i, 1]), 3)],
                  "f": [round(float(Fxz[i, 0]), 4), round(float(Fg[i, 1]), 4), round(float(Fxz[i, 1]), 4)],
                  "u": [round(float(Uxz[i, 0]), 4), round(float(Ug[i, 1]), 4), round(float(Uxz[i, 1]), 4)]}
                 for i in range(len(XZ))]
    plen = float(np.linalg.norm(np.diff(XZ, axis=0), axis=1).sum())
    return pose_json, {"mode": "gt-snap" if snap else "gt-umeyama", "scale": round(s, 3),
                       "path_m": round(plen, 2), "cam_h": round(eye - float(bbox[0][1]), 2)}


def _detect_cached(frame_paths, cache_path, threshold=0.2):
    """Run OWL-ViT once per frame; cache to JSON so re-builds are instant."""
    cache = {}
    cp = Path(cache_path)
    if cp.exists():
        cache = json.loads(cp.read_text())
    todo = [f for f in frame_paths if f not in cache]
    if todo:
        from scan2bim.detect import detect
        for i in range(0, len(todo), 16):
            for r in detect(todo[i:i + 16], threshold=threshold):
                cache[r["path"]] = r["detections"]
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(cache))
    return {f: cache.get(f, []) for f in frame_paths}


def place_autolocalize(pose_json, dtdx_files, frames_dir, *, hfov=69.0, stride=3,
                       cache="reports/coplay/_det_cache.json"):
    """Refine a PRIOR trajectory (pose_json, model coords) via per-frame object-anchor
    PnP. Detect AC/light/column on stride-sampled frames → match to model anchors of
    same type (prior = the input trajectory pose) → solvePnP → drift-free model-locked
    poses; unsampled/weak frames interpolate. Returns refined pose_json + summary."""
    import glob as _glob
    from scan2bim.anchors import anchor_inventory
    from scan2bim.localize import intrinsics, pose_from_lookat, localize_trajectory, camera_center
    cat = anchor_inventory(dtdx_files)                      # flip_x=True → display frame (matches pose_json)
    anchors = cat.get("ac", [])                            # HXX 천장기구(디퓨저/벤트) — OWL-ViT가 AC로 검출, 경로 위 분포
    frames = sorted(_glob.glob(f"{frames_dir}/*.png"))
    if not frames or not anchors:
        return pose_json, {"mode": "auto-localize", "error": "no frames or anchors"}
    from PIL import Image
    W, H = Image.open(frames[0]).size
    K = intrinsics(W, H, hfov)
    N, Nf = len(pose_json), len(frames)
    # prior = recon per-frame orientation (carries the real ~19° up-look at the ceiling)
    priors = [pose_from_lookat(p["c"], p["f"], p["u"]) for p in pose_json]
    # map pose i -> frame; detect only every `stride`-th (rest interpolate)
    fidx = [min(Nf - 1, int(round(i / max(N - 1, 1) * (Nf - 1)))) for i in range(N)]
    sampled = sorted({fidx[i] for i in range(0, N, stride)})
    det_map = _detect_cached([frames[j] for j in sampled], cache)
    dets_by_frame = [det_map.get(frames[fidx[i]], []) if (i % stride == 0) else [] for i in range(N)]
    refined, info = localize_trajectory(K, priors, dets_by_frame, anchors, max_px=160,
                                        ransac_px=16, max_dist=8.0)
    out = []
    for i in range(N):
        R, c = refined[i]["R"], refined[i]["center"]
        out.append({"c": [round(float(x), 3) for x in c],
                    "f": [round(float(x), 4) for x in R[2]],       # forward = +Z row (pose_from_lookat/PnP)
                    "u": [round(float(x), 4) for x in R[1]]})      # up = +Y row (round-trips pose_from_lookat)
    info["mode"] = "auto-localize"
    return out, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8767")
    ap.add_argument("--upload", required=True)
    ap.add_argument("--dtdx", nargs="+", required=True)
    ap.add_argument("--render-video", required=True, help="GPU point-cloud render mp4 (left panel)")
    ap.add_argument("--demo-html", default=None, help="pointcloud map HTML for poses (build_demo_map)")
    ap.add_argument("--demo-match", default="161613")
    ap.add_argument("--anchor", default=None, help="coarse 첫 위치 'x,z' (모델 좌표) → 그 근처 국소 정합")
    ap.add_argument("--gt-path", default=None, help="실제 촬영경로 waypoints 'x1,z1 x2,z2 ...' (모델 XZ) → 궤적을 이에 직접 피팅")
    ap.add_argument("--gt-mode", default="snap", choices=["snap", "umeyama"], help="snap=GT선에 스냅(직선 walk가 직선), umeyama=강체피팅(재구성 곡선 유지)")
    ap.add_argument("--auto-pipe", action="store_true", help="메인 소화배관(FXX) 런 자동검출 → 그 아래로 경로 자동(수동 waypoint 불요)")
    ap.add_argument("--auto-localize", action="store_true", help="prior 궤적을 프레임별 객체-앵커 PnP로 자동 정제(드리프트 제거)")
    ap.add_argument("--frames-dir", default=None, help="--auto-localize용 추출 프레임 폴더")
    ap.add_argument("--hfov", type=float, default=69.0, help="카메라 수평 FOV(도) — PnP 내부파라미터")
    ap.add_argument("--peer-url", default=None, help="전환 버튼이 열 상대 뷰어 파일명(예: coplay_autopipe.html)")
    ap.add_argument("--duration", type=float, default=35.3)
    ap.add_argument("--out", default="reports/coplay/coplay.html")
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    meshes_json, tris, names, model_pts = [], 0, [], []
    RENDER_CAP = 90000  # per color group (triangle-soup) for the browser
    for f in args.dtdx:
        g = decode_geometry(f); names.append(g["discipline_code"]); tris += g["triangle_count"]
        for m in g["meshes"]:
            pos = np.asarray(m["positions"], dtype=np.float32).copy()
            if not len(pos):
                continue
            pos[:, 0] *= -1.0   # Babylon(LH)→Three(RH): X 반전해야 원본 저작뷰어(정상)와 좌우 일치
            model_pts.append(pos)
            if len(pos) > RENDER_CAP:  # decimate by whole triangles
                ntri = len(pos) // 3
                keep = np.linspace(0, ntri - 1, RENDER_CAP // 3).astype(int)
                pos = pos.reshape(-1, 3, 3)[keep].reshape(-1, 3)
            meshes_json.append({"color": [round(c, 4) for c in m["color"]],
                                "b64": base64.b64encode(np.ascontiguousarray(pos).tobytes()).decode("ascii")})
    allp = np.concatenate(model_pts).astype(np.float64)
    med = np.median(allp, axis=0)
    allc = allp[(np.abs(allp - med) < 60).all(1)]       # drop only extreme outliers; keep full building (incl. SXX wing)
    bbox = (allc.min(0).tolist(), allc.max(0).tolist())
    ceil = allc[allc[:, 1] >= (bbox[1][1] - 1.5)]        # top 1.5 m = ceiling band
    ceil = ceil[np.linspace(0, len(ceil) - 1, min(60000, len(ceil))).astype(int)]

    if args.demo_html:
        poses, scan_pts = load_demo_cloud(args.demo_html, args.demo_match)
    else:
        poses, scan_pts = fetch_scan(args.base_url, args.upload)
    anchor = [float(x) for x in args.anchor.split(",")] if args.anchor else None
    if args.auto_pipe:
        from scan2bim.pipe_path import main_pipe_run_L, trajectory_turn_fraction
        # 재구성 궤적의 turn-fraction → 올바른 분기(코너) 선택 (의존성0, B1)
        vp = [viewer_pose(p) for p in poses]
        cen = np.array([v[0] for v in vp]); up_v = np.array([v[2] for v in vp])
        g = up_v.mean(0); g /= (np.linalg.norm(g) + 1e-9)
        Cg = cen @ _rot_a_to_b(g, np.array([0.0, 1.0, 0.0])).T
        tf, tang = trajectory_turn_fraction(Cg[:, [0, 2]])
        fxx = next((f for f in args.dtdx if "FXX" in f), args.dtdx[0])
        wps = main_pipe_run_L(fxx, turn_fraction=(tf if tang > 30 else None)).tolist()
        pose_json, reginfo = place_gtpath(poses, scan_pts, bbox, wps, snap=(args.gt_mode == "snap"))
        print(f"  auto-pipe(L): turn_frac={tf:.2f}({tang:.0f}°)", reginfo, "polyline:", [[round(v, 1) for v in w] for w in wps])
    elif args.gt_path:
        wps = [[float(v) for v in seg.split(",")] for seg in args.gt_path.split()]
        pose_json, reginfo = place_gtpath(poses, scan_pts, bbox, wps, snap=(args.gt_mode == "snap"))
        print("  gt-path fit:", reginfo, "waypoints:", wps)
    else:
        pose_json, reginfo = place_registered(poses, scan_pts, ceil, bbox, anchor=anchor)
        print("  registration:", reginfo, "anchor:", anchor)
    if args.auto_localize and args.frames_dir:
        pose_json, locinfo = place_autolocalize(pose_json, args.dtdx, args.frames_dir, hfov=args.hfov)
        print("  auto-localize:", locinfo)

    # 좌상단 배지: 자동(초록) vs 수동(주황) 구별
    if args.auto_localize and args.frames_dir:
        pmode, pcol = "🤖 자동 · 객체 PnP", "#31d27c"
    elif args.auto_pipe:
        pmode, pcol = "🤖 자동 · 배관런 검출", "#31d27c"
    elif args.gt_path:
        pmode, pcol = "✋ 수동 · GT경로 입력", "#ffb347"
    else:
        pmode, pcol = "🤖 자동 · 천장정합", "#31d27c"
    is_manual = bool(args.gt_path) and not (args.auto_pipe or (args.auto_localize and args.frames_dir))
    peer_label = "🤖 자동 보기" if is_manual else "✋ 수동 보기"
    peer_col = "#31d27c" if is_manual else "#ffb347"
    peer_show = "inline-block" if args.peer_url else "none"

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
            .replace("__PMODE__", pmode)
            .replace("__PMODECOL__", pcol)
            .replace("__PEERURL__", args.peer_url or "#")
            .replace("__PEERLABEL__", peer_label)
            .replace("__PEERCOL__", peer_col)
            .replace("__PEERSHOW__", peer_show)
            .replace("__RVIDEO__", rel_video))
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB) tris={tris:,} poses={len(pose_json)} render={rel_video}")


if __name__ == "__main__":
    main()

"""High-quality WebGL point-cloud renderer via additive Gaussian splatting +
weighted normalization + EDL — makes a raw point cloud read as a continuous
surface in the browser (not a pile of discs).

Pipeline per frame:
  1. depth pass : points as small opaque discs -> depth texture (for EDL/occlusion)
  2. splat pass : points as soft Gaussian discs, ADDITIVE blend -> accum RT
                  frag = vec4(color*w, w),  w = exp(-k*r^2)
  3. resolve    : color = accum.rgb / accum.a (normalized blend) then EDL shade

Usage:
    PYTHONPATH=. .venv/bin/python tools/build_splat.py \
        --demo-html reports/pointcloud_map/demo_dense_161613.html --demo-match 161613 \
        --out reports/coplay/splat.html
"""
from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

import numpy as np

TEMPLATE = r"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>point cloud · splat</title>
<style>*{margin:0}html,body{height:100%;background:#0a0d13;overflow:hidden}#c{display:block}
#hud{position:fixed;left:12px;top:12px;color:#cdd6e6;font:12px system-ui;background:rgba(10,13,19,.7);border:1px solid #283042;border-radius:8px;padding:8px 11px}
#hud b{color:#fff}</style>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
</head><body><canvas id="c"></canvas>
<div id="hud">point cloud splat · <b>__N__</b> pts<br>드래그=회전 휠=줌 · [+/-]크기 · [e]EDL</div>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const POS=__POS__, COL=__COL__, SPLAT=__SPLAT__, EDLSTR=__EDLSTR__;
function b64f32(s){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Float32Array(u.buffer);}
function b64u8(s){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return u;}
const renderer=new THREE.WebGLRenderer({canvas:document.getElementById('c'),antialias:false});
renderer.setSize(innerWidth,innerHeight); renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));
renderer.autoClear=false;
const scene=new THREE.Scene(), camera=new THREE.PerspectiveCamera(55,innerWidth/innerHeight,0.02,400);
const controls=new OrbitControls(camera,renderer.domElement);
const pos=b64f32(POS), rawc=b64u8(COL), n=pos.length/3;
const col=new Float32Array(pos.length);
for(let i=0;i<rawc.length;i++) col[i]=Math.min(1,Math.pow(rawc[i]/255,0.8)*1.4); // 저조도 톤업
const geo=new THREE.BufferGeometry();
geo.setAttribute('position',new THREE.BufferAttribute(pos,3));
geo.setAttribute('color',new THREE.BufferAttribute(col,3));
geo.computeBoundingBox(); const bb=geo.boundingBox, ctr=bb.getCenter(new THREE.Vector3()), size=bb.getSize(new THREE.Vector3());
let psize={value:SPLAT};
// --- splat (additive gaussian) material ---
const splatMat=new THREE.ShaderMaterial({ transparent:true, depthTest:false, depthWrite:false,
  blending:THREE.AdditiveBlending, uniforms:{ psize:psize, vh:{value:innerHeight} },
  vertexShader:'attribute vec3 color; varying vec3 vC; uniform float psize,vh; void main(){ vC=color; vec4 mv=modelViewMatrix*vec4(position,1.0); gl_Position=projectionMatrix*mv; gl_PointSize=psize*vh/(-mv.z); }',
  fragmentShader:'varying vec3 vC; void main(){ vec2 d=gl_PointCoord-0.5; float r2=dot(d,d); if(r2>0.25)discard; float w=exp(-7.0*r2); gl_FragColor=vec4(vC*w,w); }' });
// --- depth material (small opaque discs) for EDL/occlusion ---
const depthMat=new THREE.ShaderMaterial({ uniforms:{ psize:{value:SPLAT*0.6}, vh:{value:innerHeight} },
  vertexShader:'uniform float psize,vh; void main(){ vec4 mv=modelViewMatrix*vec4(position,1.0); gl_Position=projectionMatrix*mv; gl_PointSize=psize*vh/(-mv.z); }',
  fragmentShader:'void main(){ vec2 d=gl_PointCoord-0.5; if(dot(d,d)>0.22)discard; gl_FragColor=vec4(1.0); }' });
const pts=new THREE.Points(geo,splatMat); scene.add(pts);
const ptsD=new THREE.Points(geo,depthMat);
const dscene=new THREE.Scene(); dscene.add(ptsD);
// --- render targets ---
function mkRT(fl){ const w=Math.max(2,Math.floor(innerWidth*renderer.getPixelRatio())), h=Math.max(2,Math.floor(innerHeight*renderer.getPixelRatio()));
  const o={minFilter:THREE.NearestFilter,magFilter:THREE.NearestFilter}; if(fl) o.type=THREE.HalfFloatType;
  const rt=new THREE.WebGLRenderTarget(w,h,o); if(fl===2){ rt.depthTexture=new THREE.DepthTexture(w,h); rt.depthTexture.type=THREE.FloatType; } return rt; }
let accRT=mkRT(1), depRT=mkRT(2);
// --- resolve (normalize + EDL) ---
const resScene=new THREE.Scene(), resCam=new THREE.OrthographicCamera(-1,1,1,-1,0,1);
let edlOn=true;
const resMat=new THREE.ShaderMaterial({ uniforms:{ tAcc:{value:accRT.texture}, tDepth:{value:depRT.depthTexture}, res:{value:new THREE.Vector2(accRT.width,accRT.height)},
   edl:{value:EDLSTR}, nearF:{value:camera.near}, farF:{value:camera.far}, edlR:{value:1.3} },
  vertexShader:'varying vec2 vUv; void main(){ vUv=uv; gl_Position=vec4(position.xy,0.0,1.0);} ',
  fragmentShader:[
   'uniform sampler2D tAcc,tDepth; uniform vec2 res; uniform float edl,nearF,farF,edlR; varying vec2 vUv;',
   'float lz(vec2 u){ float z=texture2D(tDepth,u).x; return nearF*farF/(farF - z*(farF-nearF)); }',
   'void main(){ vec4 a=texture2D(tAcc,vUv); if(a.a<0.002){ gl_FragColor=vec4(0.039,0.051,0.075,1.0); return; }',
   ' vec3 c=a.rgb/a.a; float d0=lz(vUv); float s=0.0; vec2 px=edlR/res;',
   ' vec2 o[8]; o[0]=vec2(1.,0.);o[1]=vec2(-1.,0.);o[2]=vec2(0.,1.);o[3]=vec2(0.,-1.);o[4]=vec2(.7,.7);o[5]=vec2(-.7,.7);o[6]=vec2(.7,-.7);o[7]=vec2(-.7,-.7);',
   ' if(edl>0.5){ for(int i=0;i<8;i++){ float dn=lz(vUv+o[i]*px); s+=max(0.0,log2(d0)-log2(dn)); } }',
   ' float sh=exp(-s*edl/8.0); gl_FragColor=vec4(c*sh,1.0); }'
  ].join('\n') });
resScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2),resMat));
const d=Math.max(size.x,size.y,size.z)*0.9; camera.position.set(ctr.x+d,ctr.y+d*0.5,ctr.z+d); controls.target.copy(ctr); controls.update();
function resize(){ camera.aspect=innerWidth/innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth,innerHeight);
  splatMat.uniforms.vh.value=innerHeight; depthMat.uniforms.vh.value=innerHeight;
  accRT.dispose(); depRT.dispose(); accRT=mkRT(1); depRT=mkRT(2);
  resMat.uniforms.tAcc.value=accRT.texture; resMat.uniforms.tDepth.value=depRT.depthTexture; resMat.uniforms.res.value.set(accRT.width,accRT.height); }
addEventListener('resize',resize);
addEventListener('keydown',e=>{ if(e.key==='e')edlOn=!edlOn; if(e.key==='+'||e.key==='=')psize.value*=1.2; if(e.key==='-')psize.value/=1.2; depthMat.uniforms.psize.value=psize.value*0.6; });
function loop(){ requestAnimationFrame(loop); controls.update();
  resMat.uniforms.edl.value=edlOn?EDLSTR:0.0; resMat.uniforms.nearF.value=camera.near; resMat.uniforms.farF.value=camera.far;
  renderer.setRenderTarget(depRT); renderer.clear(true,true,true); renderer.render(dscene,camera);
  renderer.setRenderTarget(accRT); renderer.setClearColor(0x000000,0); renderer.clear(true,true,true); renderer.render(scene,camera);
  renderer.setRenderTarget(null); renderer.clear(true,true,true); renderer.render(resScene,resCam);
}
loop();
window.__splat={camera,controls,n,bb};
</script></body></html>"""


def load_demo_cloud(html_path, match):
    html = Path(html_path).read_text(encoding="utf-8")
    m = (re.search(r'(\[\{"label".*?\}\])\s*[;\)]', html, re.S)
         or re.search(r'(\[\{.*?"xyz".*?\}\])', html, re.S))
    ds = json.loads(m.group(1))
    d = next((x for x in ds if match in f"{x.get('label','')}|{x.get('up','')}"), ds[-1])
    xyz = np.frombuffer(base64.b64decode(d["xyz"]), dtype="<f4").reshape(-1, 3)
    rgb = (np.frombuffer(base64.b64decode(d["rgb"]), dtype=np.uint8).reshape(-1, 3)
           if d.get("rgb") else np.full((len(xyz), 3), 180, np.uint8))
    return xyz.astype(np.float32), rgb.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-html", required=True)
    ap.add_argument("--demo-match", default="161613")
    ap.add_argument("--splat-size", type=float, default=0.06)
    ap.add_argument("--edl-strength", type=float, default=20.0)
    ap.add_argument("--out", default="reports/coplay/splat.html")
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    xyz, rgb = load_demo_cloud(args.demo_html, args.demo_match)
    html = (TEMPLATE
            .replace("__POS__", json.dumps(base64.b64encode(xyz.tobytes()).decode("ascii")))
            .replace("__COL__", json.dumps(base64.b64encode(rgb.tobytes()).decode("ascii")))
            .replace("__N__", f"{len(xyz):,}")
            .replace("__SPLAT__", repr(float(args.splat_size)))
            .replace("__EDLSTR__", repr(float(args.edl_strength))))
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB) n={len(xyz):,}")


if __name__ == "__main__":
    main()

"""Patch coplay HTML so orbit zoom cannot clip the BIM model away.

Existing per-upload `*.coplay.html` is tens of MB and is served as-is. This
rewrites a few unique JS snippets at serve time (and is used by the builder
template) so a rebuild is not required.
"""
from __future__ import annotations

MARKER = "/* cov-orbit-clip */"

_FRAME = "controls.target.copy(c0); controls.update();"
_PLAY = (
    "playb.onclick=()=>{ if(rvid.paused){rvid.play();playb.textContent='⏸ 일시정지';}"
    "else{rvid.pause();playb.textContent='▶ 재생';} };"
)
_LOOP = "function loop(){requestAnimationFrame(loop);sync(rvid.currentTime||0);"

_AFTER_FRAME = """
""" + MARKER + """
function _camOk(){return Number.isFinite(camera.position.x)&&Number.isFinite(controls.target.x)&&camera.position.distanceTo(controls.target)>0.25;}
function _clipCam(){const d=Math.max(0.5,camera.position.distanceTo(controls.target));const near=Math.max(0.05,Math.min(1.5,d/80));const far=Math.max(400,d*25,(typeof sz!=='undefined'?Math.max(sz.x,sz.y,sz.z,1):50)*8);if(Math.abs(camera.near-near)>0.01||Math.abs(camera.far-far)>1){camera.near=near;camera.far=far;camera.updateProjectionMatrix();}}
function _reframe(){const _md=Math.max((typeof sz!=='undefined'?Math.max(sz.x,sz.y,sz.z):20),1);const _fv=camera.fov*Math.PI/180;const _d=Math.abs(_md/Math.sin(_fv/2))*0.85;camera.position.set(c0.x+_d*0.612,c0.y+_d*0.5,c0.z+_d*0.612);camera.up.set(0,1,0);controls.target.copy(c0);controls.update();_clipCam();}
{const span=Math.max(sz.x,sz.y,sz.z,1);controls.minDistance=Math.max(0.6,span*0.015);controls.maxDistance=Math.max(120,span*6);controls.zoomSpeed=0.85;controls.maxPolarAngle=Math.PI*0.92;_clipCam();}
addEventListener('wheel',e=>{if(e.ctrlKey)e.preventDefault();},{passive:false});
"""

_NEW_PLAY = (
    "playb.onclick=()=>{ const d=camera.position.distanceTo(controls.target);"
    "if(!_camOk()||d<controls.minDistance*1.05||d>controls.maxDistance*0.98)_reframe();"
    "if(rvid.paused){const p=rvid.play();if(p&&p.catch)p.catch(()=>{});playb.textContent='⏸ 일시정지';}"
    "else{rvid.pause();playb.textContent='▶ 재생';} };"
)

_NEW_LOOP = (
    "function loop(){requestAnimationFrame(loop);sync(rvid.currentTime||0);"
    "if(!_camOk())_reframe();_clipCam();"
)

_HOOK_OLD = (
    "window.__coplay={scene,camera,box,RAWP,setFrame,setFollow:v=>{followCam=!!v;updCtl();}};"
)
_HOOK_NEW = (
    "window.__coplay={scene,camera,controls,box,RAWP,setFrame,"
    "setFollow:v=>{followCam=!!v;updCtl();}};"
)


def apply_coplay_orbit_clip(html: str) -> str:
    """Idempotent rewrite of coplay viewer JS. Unchanged if already patched or snippet missing."""
    out = html
    if MARKER not in out and _FRAME in out:
        out = out.replace(_FRAME, _FRAME + _AFTER_FRAME, 1)
        if _PLAY in out:
            out = out.replace(_PLAY, _NEW_PLAY, 1)
        if _LOOP in out:
            out = out.replace(_LOOP, _NEW_LOOP, 1)
    if _HOOK_OLD in out:
        out = out.replace(_HOOK_OLD, _HOOK_NEW, 1)
    return out

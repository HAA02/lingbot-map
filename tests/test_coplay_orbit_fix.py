"""Orbit zoom must not send the coplay camera through the model (empty view / dead play)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "realtime"))

from coplay_orbit_fix import MARKER, apply_coplay_orbit_clip  # noqa: E402


_MINI = """
const camera=new THREE.PerspectiveCamera(55,innerWidth/innerHeight,0.05,2000);
const controls=new OrbitControls(camera,renderer.domElement);
const c0=box.getCenter(new THREE.Vector3()), sz=box.getSize(new THREE.Vector3());
controls.target.copy(c0); controls.update();
playb.onclick=()=>{ if(rvid.paused){rvid.play();playb.textContent='⏸ 일시정지';}else{rvid.pause();playb.textContent='▶ 재생';} };
function loop(){requestAnimationFrame(loop);sync(rvid.currentTime||0);
  if(followCam&&!placeMode){camera.position.lerp(_fe,0.12);}
  renderer.render(scene,camera);}
window.__coplay={scene,camera,box,RAWP,setFrame,setFollow:v=>{followCam=!!v;updCtl();}};
"""


class TestCoplayOrbitClip(unittest.TestCase):
    def test_inserts_clip_and_zoom_limits(self):
        out = apply_coplay_orbit_clip(_MINI)
        self.assertIn(MARKER, out)
        self.assertIn("controls.minDistance", out)
        self.assertIn("controls.maxDistance", out)
        self.assertIn("_clipCam", out)
        self.assertIn("_reframe", out)
        self.assertIn("if(!_camOk())_reframe();_clipCam();", out)
        self.assertIn("rvid.play()", out)
        self.assertIn("p.catch", out)
        self.assertIn("window.__coplay={scene,camera,controls,box", out)

    def test_idempotent(self):
        once = apply_coplay_orbit_clip(_MINI)
        twice = apply_coplay_orbit_clip(once)
        self.assertEqual(once, twice)
        self.assertEqual(once.count(MARKER), 1)

    def test_unknown_html_unchanged(self):
        raw = "<html>no coplay camera</html>"
        self.assertEqual(apply_coplay_orbit_clip(raw), raw)

    def test_gasan_built_html_has_snippets(self):
        path = ROOT / "realtime/_uploads/upload_1781521406685.coplay.html"
        if not path.exists():
            self.skipTest("gasan coplay html not in this checkout")
        # only scan the script head/tail via unique ASCII lines, not the 19MB mesh payload
        found = {"frame": False, "play": False, "loop": False}
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                if "controls.target.copy(c0); controls.update();" in line:
                    found["frame"] = True
                if "playb.onclick=()=>{ if(rvid.paused){rvid.play();" in line:
                    found["play"] = True
                if "function loop(){requestAnimationFrame(loop);sync(rvid.currentTime||0);" in line:
                    found["loop"] = True
                if all(found.values()):
                    break
        self.assertTrue(all(found.values()), found)

    def test_gasan_patch_applies(self):
        path = ROOT / "realtime/_uploads/upload_1781521406685.coplay.html"
        if not path.exists():
            self.skipTest("gasan coplay html not in this checkout")
        html = path.read_text(encoding="utf-8")
        out = apply_coplay_orbit_clip(html)
        self.assertIn(MARKER, out)
        self.assertIn("controls.minDistance", out)
        self.assertIn("window.__coplay={scene,camera,controls,box", out)


if __name__ == "__main__":
    unittest.main()

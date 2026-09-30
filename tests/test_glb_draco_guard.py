"""validate_wall_anchor._assert_draco_decodable — Draco 압축 GLB 를 DracoPy 없이 열면 조용히
all-zero 정점이 되는 문제(docs/glb-auto-mapping-review.md '12 m 불일치')를 즉시 실패로 바꾼다."""
import json
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.validate_wall_anchor import _assert_draco_decodable  # noqa: E402


def _write_glb(path: Path, doc: dict) -> Path:
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    body = struct.pack("<II", len(js), 0x4E4F534A) + js
    path.write_bytes(b"glTF" + struct.pack("<II", 2, 12 + len(body)) + body)
    return path


class TestDracoGuard(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())

    def test_uncompressed_glb_passes(self):
        p = _write_glb(self.tmp / "plain.glb", {"asset": {"version": "2.0"}})
        _assert_draco_decodable(p)  # no raise

    def test_non_glb_file_is_ignored(self):
        p = self.tmp / "x.obj"
        p.write_text("v 0 0 0\n")
        _assert_draco_decodable(p)

    def test_draco_glb_requires_dracopy(self):
        p = _write_glb(self.tmp / "draco.glb",
                       {"asset": {"version": "2.0"}, "extensionsUsed": ["KHR_draco_mesh_compression"]})
        try:
            import DracoPy  # noqa: F401
            has = True
        except ImportError:
            has = False
        if has:
            _assert_draco_decodable(p)  # DracoPy 설치 환경: 통과해야 함
        else:
            with self.assertRaises(RuntimeError) as cm:
                _assert_draco_decodable(p)
            self.assertIn("DracoPy", str(cm.exception))


if __name__ == "__main__":
    unittest.main()

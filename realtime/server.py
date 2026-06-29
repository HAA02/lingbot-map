"""Phase 1 realtime WebRTC server.

Receives a video track from an Android browser, decodes frames, and pushes
statistics back over a DataChannel. No model inference yet (Phase 2).

Run:
    .venv/bin/python realtime/server.py --host 0.0.0.0 --port 8443

The first run creates a self-signed cert at realtime/cert.pem / key.pem
unless --no-ssl is given (HTTP, only works on localhost — Chrome requires
HTTPS for getUserMedia from non-localhost origins).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import re
import ssl
import struct
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from aiortc import (
    MediaStreamTrack, RTCPeerConnection, RTCSessionDescription,
)
from fastapi import FastAPI, Request, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from inference_worker import InferenceWorker

log = logging.getLogger("realtime")
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
MODELS_DIR = REPO_ROOT / "models"
FRAME_DUMP_DIR = ROOT / "_received_frames"
FRAME_DUMP_DIR.mkdir(exist_ok=True)

# Single shared state. Phase 1 only supports one active connection at a time.
class SessionState:
    def __init__(self) -> None:
        self.pcs: set[RTCPeerConnection] = set()
        self.frames_received: int = 0
        self.last_frame_ts: float = 0.0
        self.last_frame_shape: tuple[int, int] | None = None
        self.start_ts: float = time.time()
        self.dump_every_n: int = 30   # save 1 frame to disk every N for inspection
        self.latest_jpeg: bytes | None = None        # most recent frame for /preview
        self.latest_jpeg_version: int = 0            # bumps on each new frame
        self.frame_event: asyncio.Event = asyncio.Event()
        self.viewer_sockets: set[WebSocket] = set()  # /ws viewers receiving point clouds
        self.worker: InferenceWorker | None = None
        self.last_infer_ts: float = 0.0
        self.last_infer_points: int = 0
        self.last_payload: bytes | None = None    # cache for late-joining viewers
        self.last_thumbs: list[bytes] | None = None  # per-frame JPEG thumbnails

STATE = SessionState()


class FrameConsumer:
    """Pulls frames from a MediaStreamTrack and updates STATE."""

    def __init__(self, track: MediaStreamTrack, datachannel) -> None:
        self.track = track
        self.datachannel = datachannel
        self.task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        last_emit = 0.0
        try:
            while True:
                frame = await self.track.recv()
                img = frame.to_ndarray(format="bgr24")
                STATE.frames_received += 1
                STATE.last_frame_shape = img.shape[:2]  # (H, W)
                STATE.last_frame_ts = time.time()

                # encode latest frame for /preview.mjpeg viewer (downscale to ~720p max)
                h, w = img.shape[:2]
                if w > 1280:
                    scale = 1280 / w
                    img_view = cv2.resize(img, (int(w*scale), int(h*scale)))
                else:
                    img_view = img
                ok, buf = cv2.imencode(".jpg", img_view, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    STATE.latest_jpeg = buf.tobytes()
                    STATE.latest_jpeg_version += 1
                    STATE.frame_event.set()
                    STATE.frame_event.clear()

                if STATE.frames_received % STATE.dump_every_n == 1:
                    out = FRAME_DUMP_DIR / f"frame_{STATE.frames_received:06d}.jpg"
                    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 80])

                # feed inference worker (skip frames to keep buffer manageable: take ~5 fps)
                if STATE.worker is not None and STATE.frames_received % 6 == 0:
                    try:
                        await STATE.worker.push_frame(img)
                    except Exception as e:
                        log.warning("push_frame: %s", e)

                now = time.time()
                if now - last_emit > 0.5 and self.datachannel.readyState == "open":
                    elapsed = now - STATE.start_ts
                    fps = STATE.frames_received / max(elapsed, 1e-3)
                    msg = {
                        "type": "stats",
                        "frames": STATE.frames_received,
                        "shape": list(STATE.last_frame_shape) if STATE.last_frame_shape else None,
                        "server_fps": round(fps, 2),
                        "elapsed_s": round(elapsed, 1),
                    }
                    self.datachannel.send(json.dumps(msg))
                    last_emit = now
        except Exception as e:
            log.warning(f"frame consumer ended: {e}")


# ---------------- FastAPI ----------------

app = FastAPI()
if MODELS_DIR.exists():
    app.mount("/models", StaticFiles(directory=str(MODELS_DIR)), name="models")
_OUT_KAKAO_FMT = ROOT.parent / "out_kakao_fmt"
if _OUT_KAKAO_FMT.exists():
    app.mount("/out_kakao_fmt", StaticFiles(directory=str(_OUT_KAKAO_FMT)), name="out_kakao_fmt")

# per-upload coplay: dtdx files used when auto-generating coplay HTML
_COPLAY_DTDX = sorted((ROOT.parent / "models/Gasan_7F").glob("G7F_FAB_*_7F-0_Central_1.dtdx")) if (ROOT.parent / "models/Gasan_7F").exists() else []


@app.get("/", response_class=HTMLResponse)
async def index():
    # Default landing page in upload mode is the upload UI.
    return FileResponse(ROOT / "upload.html", headers={"Cache-Control": "no-store"})


@app.get("/upload", response_class=HTMLResponse)
async def upload_page():
    return FileResponse(ROOT / "upload.html", headers={"Cache-Control": "no-store"})


@app.get("/stream", response_class=HTMLResponse)
async def stream_page():
    # Legacy WebRTC streaming page kept for reference; not used in upload mode.
    return FileResponse(ROOT / "index.html")


@app.get("/viewer", response_class=HTMLResponse)
async def viewer_page():
    return FileResponse(ROOT / "viewer.html", headers={"Cache-Control": "no-store"})


@app.get("/audit", response_class=HTMLResponse)
async def audit_page():
    return FileResponse(ROOT / "audit.html", headers={"Cache-Control": "no-store"})


@app.get("/coverage", response_class=HTMLResponse)
async def coverage_page():
    return FileResponse(ROOT / "coverage.html", headers={"Cache-Control": "no-store"})


@app.get("/coverage.html", response_class=HTMLResponse)
async def coverage_html_page():
    return FileResponse(ROOT / "coverage.html", headers={"Cache-Control": "no-store"})


@app.get("/coverage-report", response_class=HTMLResponse)
async def coverage_report_page():
    return FileResponse(ROOT / "coverage_report.html", headers={"Cache-Control": "no-store"})


@app.get("/coverage-report.html", response_class=HTMLResponse)
async def coverage_report_html_page():
    return FileResponse(ROOT / "coverage_report.html", headers={"Cache-Control": "no-store"})


def _serve_per_upload_coplay(upload: str) -> HTMLResponse:
    """Serve a built per-upload coplay.html, rewriting the PiP <video> src to the
    upload's video API so poses and footage come from the SAME upload."""
    import re as _re
    content = (UPLOAD_DIR / f"{upload}.coplay.html").read_text(encoding="utf-8")
    content = _re.sub(
        r'(<video\b[^>]*\s+src=")[^"]*(")',
        f'\\1/api/uploads/{upload}/video\\2',
        content,
    )
    return HTMLResponse(content, headers={"Cache-Control": "no-store"})


def _latest_coplay_upload() -> str | None:
    """Most recent upload_id that has a built coplay.html."""
    best, best_ts = None, -1
    for p in UPLOAD_DIR.glob("*.coplay.html"):
        uid = p.name.removesuffix(".coplay.html")
        ts = _upload_ts_ms(uid)
        if ts > best_ts:
            best, best_ts = uid, ts
    return best


@app.get("/coplay", response_class=HTMLResponse)
async def coplay_page(upload: str | None = None):
    if upload:
        if (UPLOAD_DIR / f"{upload}.coplay.html").exists():
            return _serve_per_upload_coplay(upload)
        # coplay not built yet for this upload
        return HTMLResponse(
            f'<html><body style="background:#111;color:#eee;font:15px sans-serif;padding:20px">'
            f'<p>upload <b>{upload}</b> coplay 아직 준비 중입니다.</p>'
            f'<p>잠시 후 새로고침하거나 <a href="/coplay" style="color:#88bbff">기본 coplay</a>를 확인하세요.</p>'
            f'</body></html>',
            status_code=202,
        )
    # no upload param → serve the latest built per-upload coplay (poses+video consistent)
    latest = _latest_coplay_upload()
    if latest:
        return _serve_per_upload_coplay(latest)
    fallback = ROOT / "coplay.html"
    if not fallback.exists():
        return HTMLResponse("<html><body>coplay.html not found</body></html>", status_code=404)
    return FileResponse(fallback, headers={"Cache-Control": "no-store"})


@app.get("/coplay.html", response_class=HTMLResponse)
async def coplay_html_page(upload: str | None = None):
    return await coplay_page(upload)


UPLOAD_DIR = ROOT / "_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

_jobs: dict = {}  # upload_id → {status, ...result}
BIM_DIR = ROOT / "_bim"
BIM_DIR.mkdir(exist_ok=True)
AUDIT_RESULTS = ROOT / "audit_results.json"


MODEL_SPECS = {
    "gasan-7f": {
        "name": "gasan-7F",
        "glb": "gasan-7F.glb",
        "metadata": "pag_export.json",
        "manifest": "gasan-7F.model_manifest.json",
    },
    "pxx": {
        "name": "PXX",
        "glb": "PXX.glb",
        "metadata": "pxx_pag_export.json",
        "manifest": "PXX.model_manifest.json",
    },
    "sxx": {
        "name": "SXX",
        "glb": "SXX.glb",
        "metadata": "pag_export.json",
        "manifest": "SXX.model_manifest.json",
    },
}


LINEAR_PAG_KEYS = [
    ("pipes", "Pipe"),
    ("ducts", "Duct"),
    ("conduits", "Conduit"),
    ("cableTrays", "CableTray"),
    ("flexDucts", "FlexDuct"),
    ("flexPipes", "FlexPipe"),
    ("wires", "Wire"),
    ("structuralFraming", "Beam"),
    ("walls", "Wall"),
]


POINT_PAG_KEYS = [
    ("elbows", "Elbow"),
    ("tees", "Tee"),
    ("valves", "Valve"),
    ("reducers", "Reducer"),
    ("ductFittings", "DuctFitting"),
    ("ductAccessories", "DuctAccessory"),
    ("mepAccessories", "MepAccessory"),
    ("cableTrayFittings", "CableTrayFitting"),
    ("conduitFittings", "ConduitFitting"),
    ("mechanicalEquipment", "MechanicalEquipment"),
    ("electricalEquipment", "ElectricalEquipment"),
    ("plumbingFixtures", "PlumbingFixture"),
    ("sprinklers", "Sprinkler"),
    ("genericModels", "GenericModel"),
    ("specialtyEquipment", "SpecialtyEquipment"),
    ("columns", "Column"),
    ("structuralColumns", "Column"),
    ("doors", "Door"),
    ("windows", "Window"),
]


def _bad_id(value: str, prefix: str) -> bool:
    return not value.startswith(prefix) or "/" in value or ".." in value


def _read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_model_id(doc: dict | list) -> str | None:
    if not isinstance(doc, dict):
        return None
    if doc.get("model_id"):
        return str(doc.get("model_id"))
    model = doc.get("model")
    if isinstance(model, dict):
        value = model.get("model_id") or model.get("id")
        return str(value) if value else None
    return None


def _load_scoped_upload_json(upload_id: str, suffix: str, model_id: str | None = None) -> dict | None:
    candidates: list[Path] = []
    if model_id:
        candidates.append(UPLOAD_DIR / f"{upload_id}.{model_id}.{suffix}.json")
    candidates.append(UPLOAD_DIR / f"{upload_id}.{suffix}.json")
    for path in candidates:
        if not path.exists():
            continue
        doc = _read_json(path)
        if model_id and _json_model_id(doc) != model_id:
            continue
        if isinstance(doc, dict):
            return doc
    return None


def _coverage_review_path(upload_id: str, model_id: str) -> Path:
    return UPLOAD_DIR / f"{upload_id}.{model_id}.coverage_review.json"


def _empty_coverage_review(upload_id: str, model_id: str) -> dict:
    return {
        "ok": True,
        "upload_id": upload_id,
        "model_id": model_id,
        "object_reviews": {},
        "frame_reviews": {},
        "notes": "",
        "updated_at": None,
    }


def _load_coverage_review(upload_id: str, model_id: str) -> dict:
    path = _coverage_review_path(upload_id, model_id)
    if not path.exists():
        return _empty_coverage_review(upload_id, model_id)
    doc = _read_json(path)
    if not isinstance(doc, dict):
        return _empty_coverage_review(upload_id, model_id)
    base = _empty_coverage_review(upload_id, model_id)
    base.update(doc)
    if not isinstance(base.get("object_reviews"), dict):
        base["object_reviews"] = {}
    if not isinstance(base.get("frame_reviews"), dict):
        base["frame_reviews"] = {}
    return base


def _sanitize_review_entry(value) -> dict:
    if not isinstance(value, dict):
        return {}
    accepted = value.get("accepted")
    if accepted is not None:
        accepted = bool(accepted)
    frame = value.get("frame")
    try:
        frame = int(frame) if frame is not None else None
    except (TypeError, ValueError):
        frame = None
    note = str(value.get("note") or "")[:500]
    reviewed_at = str(value.get("reviewed_at") or time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    out = {"accepted": accepted, "note": note, "reviewed_at": reviewed_at}
    if frame is not None:
        out["frame"] = frame
    return out


def _pag_units_scale(units: str | None) -> float:
    u = (units or "").lower()
    if u in ("mm", "millimeter", "millimeters"):
        return 0.001
    if u in ("cm", "centimeter", "centimeters"):
        return 0.01
    return 1.0


def _vec_scaled(v, scale: float) -> list[float] | None:
    if v is None:
        return None
    return [float(x) * scale for x in v]


def _bbox_from_center(center: list[float], w: float, h: float, l: float, scale: float) -> list[list[float]]:
    cx, cy, cz = _vec_scaled(center, scale)
    dx = abs(float(w or 0.0)) * scale / 2 or 0.05
    dy = abs(float(l or 0.0)) * scale / 2 or 0.05
    dz = abs(float(h or 0.0)) * scale / 2 or 0.05
    return [[cx - dx, cy - dy, cz - dz], [cx + dx, cy + dy, cz + dz]]


def _extract_glb_names(path: Path, *, max_names: int = 200_000) -> set[str]:
    """Return node/mesh names from a GLB JSON chunk without a full glTF loader."""
    if not path.exists() or path.suffix.lower() != ".glb":
        return set()
    names: set[str] = set()
    try:
        with path.open("rb") as f:
            header = f.read(12)
            if len(header) != 12 or header[:4] != b"glTF":
                return names
            while True:
                chunk_header = f.read(8)
                if len(chunk_header) < 8:
                    break
                length, ctype = struct.unpack("<I4s", chunk_header)
                data = f.read(length)
                if ctype == b"JSON":
                    doc = json.loads(data.decode("utf-8"))
                    for key in ("nodes", "meshes"):
                        for item in doc.get(key, []) or []:
                            name = item.get("name")
                            if name:
                                names.add(str(name))
                            extras = item.get("extras")
                            if isinstance(extras, dict):
                                for k in ("guid", "GUID", "ifcGuid", "revitId", "RevitId", "UniqueId", "uniqueId", "ElementID", "elementId"):
                                    if extras.get(k):
                                        names.add(str(extras[k]))
                            if len(names) >= max_names:
                                return names
                    return names
    except Exception as e:
        log.warning("GLB name scan failed for %s: %s", path, e)
    return names


_GLB_NAME_CACHE: dict[str, tuple[float, set[str]]] = {}


def _glb_name_set(glb_path: Path) -> set[str]:
    """Cached GLB identifier set (node/mesh name + extras GUIDs), keyed by path+mtime."""
    try:
        mtime = glb_path.stat().st_mtime
    except OSError:
        return set()
    key = str(glb_path)
    cached = _GLB_NAME_CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1]
    names = _extract_glb_names(glb_path)
    _GLB_NAME_CACHE[key] = (mtime, names)
    return names


_GLB_DOC_CACHE: dict[str, tuple[float, dict, bytes | None]] = {}


def _glb_doc_and_bin(glb_path: Path) -> tuple[dict | None, bytes | None]:
    """Cached GLB JSON chunk + binary chunk, keyed by path+mtime."""
    try:
        mtime = glb_path.stat().st_mtime
    except OSError:
        return None, None
    key = str(glb_path)
    cached = _GLB_DOC_CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1], cached[2]
    try:
        with glb_path.open("rb") as f:
            header = f.read(12)
            if len(header) != 12 or header[:4] != b"glTF":
                return None, None
            doc = None
            bin_chunk = None
            while True:
                chunk_header = f.read(8)
                if len(chunk_header) < 8:
                    break
                length, ctype = struct.unpack("<I4s", chunk_header)
                data = f.read(length)
                if ctype == b"JSON":
                    doc = json.loads(data.decode("utf-8"))
                elif ctype == b"BIN\x00":
                    bin_chunk = data
            if doc is None:
                return None, None
            _GLB_DOC_CACHE[key] = (mtime, doc, bin_chunk)
            return doc, bin_chunk
    except Exception as e:
        log.warning("GLB JSON scan failed for %s: %s", glb_path, e)
        return None, None


def _glb_json(glb_path: Path) -> dict | None:
    return _glb_doc_and_bin(glb_path)[0]


def _model_registry() -> dict:
    """Static specs whose files exist + auto-discovered models/*.glb (GLB-native)."""
    specs: dict[str, dict] = {}
    for mid, spec in MODEL_SPECS.items():
        if (MODELS_DIR / spec["glb"]).exists():
            specs[mid] = spec
    if MODELS_DIR.exists():
        for pth in sorted(MODELS_DIR.glob("*.glb")):
            if any(sp.get("glb") == pth.name for sp in specs.values()):
                continue
            mid = re.sub(r"[^a-z0-9_-]", "-", pth.stem.lower()) or "model"
            if mid in specs:
                continue
            specs[mid] = {
                "name": pth.stem,
                "glb": pth.name,
                "metadata": None,
                "manifest": f"{pth.stem}.model_manifest.json",
                "kind": "glb_native",
            }
    return specs


def _default_model_id() -> str:
    """등록된 첫 모델 id (하드코딩 제거용 동적 기본값)."""
    return next(iter(_model_registry()), "")


GLB_NATIVE_CATEGORY = {
    "Pipes": "Pipe",
    "Pipe Fittings": "PipeFitting",
    "Ducts": "Duct",
    "Duct Fittings": "DuctFitting",
    "Duct Accessories": "DuctAccessory",
    "Pipe Accessories": "PipeAccessory",
    "Cable Trays": "CableTray",
    "Conduits": "Conduit",
}


def _glb_node_matrix(n: dict) -> np.ndarray:
    if "matrix" in n:
        return np.array(n["matrix"], dtype=np.float64).reshape(4, 4).T
    T = np.eye(4)
    if "translation" in n:
        T[:3, 3] = n["translation"]
    R = np.eye(4)
    if "rotation" in n:
        x, y, z, w = n["rotation"]
        R[:3, :3] = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
    S = np.eye(4)
    if "scale" in n:
        S[0, 0], S[1, 1], S[2, 2] = n["scale"]
    return T @ R @ S


def _load_glb_native_model(model_id: str, spec: dict) -> tuple[dict, list[dict]]:
    """GLB 자체를 메타데이터 소스로 객체를 추출한다 (PAG JSON 불필요).
    노드 extras의 IfcGUID/Category와 accessor bbox + 노드 변환으로 객체를 만들고,
    glTF Y-up(m) -> model_world Z-up(m) 변환((x, -z, y))을 적용한다."""
    glb_path = MODELS_DIR / spec["glb"]
    doc, bin_chunk = _glb_doc_and_bin(glb_path)
    if not doc:
        raise FileNotFoundError(str(glb_path))
    nodes = doc.get("nodes", [])
    meshes = doc.get("meshes", [])
    accs = doc.get("accessors", [])
    bvs = doc.get("bufferViews", [])

    def read_positions(acc_idx: int) -> np.ndarray | None:
        """비압축 POSITION 정점 직접 디코드 (일부 익스포터의 accessor min/max 오류 대비)."""
        a = accs[acc_idx]
        if bin_chunk is None or "bufferView" not in a or a.get("componentType") != 5126 or a.get("type") != "VEC3":
            return None
        bv = bvs[a["bufferView"]]
        if bv.get("byteStride") not in (None, 12):
            return None
        boff = int(bv.get("byteOffset", 0)) + int(a.get("byteOffset", 0))
        cnt = int(a["count"])
        raw = bin_chunk[boff:boff + cnt * 12]
        if len(raw) < cnt * 12:
            return None
        return np.frombuffer(raw, dtype="<f4").reshape(cnt, 3).astype(np.float64)
    parent: dict[int, int] = {}
    for i, n in enumerate(nodes):
        for c in n.get("children", []) or []:
            parent[c] = i

    def world(i: int) -> np.ndarray:
        M = _glb_node_matrix(nodes[i])
        p = parent.get(i)
        while p is not None:
            M = _glb_node_matrix(nodes[p]) @ M
            p = parent.get(p)
        return M

    def to_model(pts: np.ndarray) -> np.ndarray:
        # glTF Y-up -> Z-up: (x, y, z) -> (x, -z, y)
        out = np.empty_like(pts)
        out[:, 0] = pts[:, 0]
        out[:, 1] = -pts[:, 2]
        out[:, 2] = pts[:, 1]
        return out

    # 노드별 객체 생성. 일부 익스포터는 같은 IfcGUID를 타입 인스턴스들에 공유하므로
    # 병합하지 않고 발생 순서 접미사(base@k)로 고유화한다. bbox는 가능하면 실제 정점에서
    # 계산한다(이 익스포터의 accessor min/max는 일부 축이 부정확함을 실측으로 확인).
    per_node: list[dict] = []
    guid_seen: dict[str, int] = {}
    for i, n in enumerate(nodes):
        ex = n.get("extras") or {}
        base_guid = ex.get("IfcGUID") or ex.get("UniqueId")
        if not base_guid or "mesh" not in n:
            continue
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        ok = False
        M = world(i)
        identity = bool(np.allclose(M, np.eye(4), atol=1e-12))
        for prim in meshes[n["mesh"]].get("primitives", []):
            a_idx = prim["attributes"]["POSITION"]
            verts = read_positions(a_idx)
            if verts is not None:
                vw = verts if identity else (np.c_[verts, np.ones(len(verts))] @ M.T)[:, :3]
                lo = np.minimum(lo, vw.min(0))
                hi = np.maximum(hi, vw.max(0))
                ok = True
                continue
            a = accs[a_idx]
            if "min" not in a or "max" not in a:
                continue
            cr = np.array([[x, y, z]
                           for x in (a["min"][0], a["max"][0])
                           for y in (a["min"][1], a["max"][1])
                           for z in (a["min"][2], a["max"][2])], dtype=np.float64)
            crw = (np.c_[cr, np.ones(8)] @ M.T)[:, :3]
            lo = np.minimum(lo, crw.min(0))
            hi = np.maximum(hi, crw.max(0))
            ok = True
        if not ok:
            continue
        k = guid_seen.get(str(base_guid), 0)
        guid_seen[str(base_guid)] = k + 1
        guid = str(base_guid) if k == 0 else f"{base_guid}@{k}"
        per_node.append({"guid": guid, "base_guid": str(base_guid), "lo": lo, "hi": hi, "extras": ex})

    objects: list[dict] = []
    counts: dict[str, int] = {}
    for ent in per_node:
        guid = ent["guid"]
        ex = ent["extras"]
        box = to_model(np.vstack([ent["lo"], ent["hi"]]))
        b_lo = box.min(0)
        b_hi = box.max(0)
        center = ((b_lo + b_hi) * 0.5).tolist()
        span = b_hi - b_lo
        order = np.argsort(span)
        category = GLB_NATIVE_CATEGORY.get(str(ex.get("Category") or ""), str(ex.get("Category") or "Unknown"))
        # linear 판정: 진짜 선형 카테고리만. Fitting류는 GUID 병합 시 L자형 bbox가 되어
        # 장축 start/end가 실제 형상과 무관한 대각선이 되므로 point(bbox) 객체로 둔다.
        start = end = None
        linear_ok = category in ("Pipe", "Duct", "Conduit", "CableTray", "FlexDuct", "FlexPipe")
        if linear_ok and span[order[2]] >= 0.5 and span[order[2]] >= 3.0 * max(span[order[1]], 1e-6):
            axis = np.zeros(3)
            axis[order[2]] = span[order[2]] * 0.5
            c = np.asarray(center)
            start = (c - axis).tolist()
            end = (c + axis).tolist()
        rec = {
            "guid": str(guid),
            "base_guid": ent["base_guid"],
            "category": category,
            "system": ex.get("System Name") or ex.get("System Type"),
            "zone": ex.get("Level") or ex.get("Reference Level"),
            "center": center,
            "start": start,
            "end": end,
            "bbox": [b_lo.tolist(), b_hi.tolist()],
            "is_linear": bool(start is not None),
            "diameter_m": float(span[order[1]]),
            "source_key": "glb_native",
            "has_glb_mesh": True,
        }
        objects.append(rec)
        counts[category] = counts.get(category, 0) + 1

    manifest = {
        "id": model_id,
        "model_id": model_id,
        "name": spec["name"],
        "glb": spec["glb"],
        "metadata": None,
        "manifest": spec["manifest"],
        "units": "m",
        "unit_scale_to_m": 1.0,
        "up_axis": "Z_UP",
        "metadata_kind": "glb_native",
        "glb_axis_transform": "gltf_yup_to_zup",
        "object_count": len(objects),
        "counts_by_category": counts,
    }
    return manifest, objects


def _load_model_objects(model_id: str) -> tuple[dict, list[dict]]:
    spec = _model_registry().get(model_id)
    if not spec:
        raise KeyError(f"unknown model_id: {model_id}")
    if spec.get("kind") == "glb_native":
        return _load_glb_native_model(model_id, spec)
    meta_path = MODELS_DIR / spec["metadata"]
    if not meta_path.exists():
        raise FileNotFoundError(str(meta_path))
    pag = _read_json(meta_path)
    project = pag.get("project", {}) if isinstance(pag, dict) else {}
    scale = _pag_units_scale(project.get("units"))
    units = project.get("units", "m")
    objects: list[dict] = []
    counts: dict[str, int] = {}

    def add_obj(obj: dict, category: str, *, linear: bool) -> None:
        guid = obj.get("guid")
        if not guid:
            return
        start = _vec_scaled(obj.get("start"), scale)
        end = _vec_scaled(obj.get("end"), scale)
        center = _vec_scaled(obj.get("center"), scale)
        if center is None and start is not None and end is not None:
            center = [(a + b) * 0.5 for a, b in zip(start, end)]
        bbox = None
        if obj.get("bbox"):
            raw = obj["bbox"]
            if isinstance(raw, list) and len(raw) == 2:
                bbox = [_vec_scaled(raw[0], scale), _vec_scaled(raw[1], scale)]
        if bbox is None and center is not None and not linear:
            bbox = _bbox_from_center(
                obj.get("center"),
                obj.get("width") or obj.get("overallWidth") or obj.get("diameter") or obj.get("largeDiameter") or 100.0,
                obj.get("height") or obj.get("overallHeight") or obj.get("diameter") or obj.get("largeDiameter") or 100.0,
                obj.get("length") or obj.get("faceToFace") or 100.0,
                scale,
            )
        if center is None and start is None and bbox is None:
            return
        rec = {
            "guid": str(guid),
            "category": category,
            "system": obj.get("systemName") or obj.get("system") or obj.get("systemType"),
            "zone": obj.get("level"),
            "center": center,
            "start": start,
            "end": end,
            "bbox": bbox,
            "is_linear": bool(start is not None and end is not None),
            "diameter_m": float(obj.get("diameter") or obj.get("largeDiameter") or obj.get("runDiameter") or 0.0) * scale,
            "source_key": obj.get("_source_key"),
        }
        objects.append(rec)
        counts[category] = counts.get(category, 0) + 1

    if isinstance(pag, dict):
        for key, category in LINEAR_PAG_KEYS:
            for item in pag.get(key, []) or []:
                item["_source_key"] = key
                add_obj(item, category, linear=True)
        for key, category in POINT_PAG_KEYS:
            for item in pag.get(key, []) or []:
                item["_source_key"] = key
                add_obj(item, category, linear=False)

    manifest = {
        "id": model_id,
        "model_id": model_id,
        "name": spec["name"],
        "glb": spec["glb"],
        "metadata": spec["metadata"],
        "manifest": spec["manifest"],
        "units": units,
        "unit_scale_to_m": scale,
        "up_axis": "Z_UP",
        "pag_to_glb_transform": [
            [scale, 0, 0, 0],
            [0, scale, 0, 0],
            [0, 0, scale, 0],
            [0, 0, 0, 1],
        ],
        "object_count": len(objects),
        "counts_by_category": counts,
    }
    glb_names = _glb_name_set(MODELS_DIR / spec["glb"])
    for rec in objects:
        rec["has_glb_mesh"] = rec["guid"] in glb_names
    return manifest, objects


def _model_manifest(model_id: str, *, persist: bool = False) -> dict:
    spec = _model_registry()[model_id]
    glb_path = MODELS_DIR / spec["glb"]
    meta_path = MODELS_DIR / spec["metadata"] if spec.get("metadata") else None
    manifest, objects = _load_model_objects(model_id)
    guids = {o["guid"] for o in objects}
    if spec.get("kind") == "glb_native":
        # GLB 자체가 메타데이터 소스이므로 모든 객체가 mesh와 직접 연결된다
        matched = len(guids)
    else:
        names = _glb_name_set(glb_path)
        matched = sum(1 for g in guids if g in names)
    mapping_ratio = matched / max(len(guids), 1)
    proxy_bounds = _model_object_bounds(objects)
    guid_mapping_status = "direct" if mapping_ratio >= 0.60 else "partial" if mapping_ratio >= 0.10 else "proxy_only"
    manifest.update({
        "glb_url": f"/models/{spec['glb']}",
        "metadata_url": f"/models/{spec['metadata']}" if spec.get("metadata") else None,
        "manifest_url": f"/models/{spec['manifest']}",
        "manifest_valid": glb_path.exists() and (meta_path is None or meta_path.exists()) and bool(objects),
        "guid_mapping_ratio": round(mapping_ratio, 4),
        "guid_mapping_status": guid_mapping_status,
        "guid_mapped_count": matched,
        "guid_total": len(guids),
        "fallback_geometry": "pag_proxy",
        "fallback_geometry_count": max(0, len(guids) - matched),
        "coverage_geometry_source": (
            "glb_guid_mesh" if mapping_ratio >= 0.60
            else "hybrid_glb_proxy" if mapping_ratio > 0.0
            else "pag_proxy"
        ),
        "glb_role": (
            "coverage_geometry" if mapping_ratio >= 0.60
            else "partial_coverage_geometry" if mapping_ratio > 0.0
            else "visual_context_only"
        ),
        "proxy_geometry_bounds": proxy_bounds,
        "glb_bytes": glb_path.stat().st_size if glb_path.exists() else 0,
        "metadata_bytes": meta_path.stat().st_size if (meta_path is not None and meta_path.exists()) else 0,
    })
    if persist and MODELS_DIR.exists():
        out = MODELS_DIR / spec["manifest"]
        out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _select_scan_payload(upload_id: str, preferred: str | None = None) -> Path | None:
    variants = {
        "mesh": UPLOAD_DIR / f"{upload_id}.lbm1.mesh",
        "detail": UPLOAD_DIR / f"{upload_id}.lbp4.test",
        "texture": UPLOAD_DIR / f"{upload_id}.lbp4.texture",
        "visual": UPLOAD_DIR / f"{upload_id}.lbp4.visual",
        "lbp2": UPLOAD_DIR / f"{upload_id}.lbp2",
    }
    if preferred and preferred in variants and variants[preferred].exists():
        return variants[preferred]
    for key in ("mesh", "detail", "texture", "visual", "lbp2"):
        if variants[key].exists():
            return variants[key]
    return None


def _parse_scan_payload(path: Path, *, include_points: bool = False, max_points: int = 250_000) -> dict:
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError("payload too small")
    magic = data[:4]
    out: dict = {
        "payload": path.name,
        "payload_bytes": len(data),
        "magic": magic.decode("ascii", errors="replace"),
        "points": None,
        "colors": None,
        "source_ids": None,
    }
    if magic in (b"LBP2", b"LBP3", b"LBP4"):
        _magic, flags, num_pts, num_poses, seq = struct.unpack("<4sIIII", data[:20])
        off = 20
        xyz = np.frombuffer(data, dtype="<f4", count=num_pts * 3, offset=off).reshape(-1, 3)
        off += num_pts * 3 * 4
        rgb = np.frombuffer(data, dtype=np.uint8, count=num_pts * 3, offset=off).reshape(-1, 3)
        off += num_pts * 3
        if magic == b"LBP3":
            off += num_pts * 3 * 4
        src_ids = None
        if magic == b"LBP4":
            off += num_pts * 3 * 4  # normals
            src_ids = np.frombuffer(data, dtype="<u4", count=num_pts, offset=off)
            off += num_pts * 4
            off += num_pts * 4  # component ids
        poses = np.frombuffer(data, dtype="<f4", count=num_poses * 12, offset=off).reshape(-1, 12)
        off += num_poses * 12 * 4
        K = np.frombuffer(data, dtype="<f4", count=9, offset=off).reshape(3, 3)
        out.update({"count": int(num_pts), "poses": poses.tolist(), "K": K.tolist(), "flags": int(flags), "seq": int(seq)})
        pts_for_bounds = xyz
        if include_points:
            step = max(1, int(np.ceil(num_pts / max_points)))
            sample = xyz[::step].astype(np.float32)
            out["points"] = sample.tolist()
            out["colors"] = rgb[::step].astype(np.uint8).tolist()
            if src_ids is not None:
                out["source_ids"] = src_ids[::step].astype(np.uint32).tolist()
    elif magic == b"LBM1":
        _magic, flags, num_v, num_f, num_poses, seq = struct.unpack("<4sIIIII", data[:24])
        off = 24
        verts = np.frombuffer(data, dtype="<f4", count=num_v * 3, offset=off).reshape(-1, 3)
        off += num_v * 3 * 4
        rgb = np.frombuffer(data, dtype=np.uint8, count=num_v * 3, offset=off).reshape(-1, 3)
        off += num_v * 3
        off += num_f * 3 * 4  # faces
        poses = np.frombuffer(data, dtype="<f4", count=num_poses * 12, offset=off).reshape(-1, 12)
        off += num_poses * 12 * 4
        K = np.frombuffer(data, dtype="<f4", count=9, offset=off).reshape(3, 3)
        out.update({
            "count": int(num_v),
            "faces": int(num_f),
            "poses": poses.tolist(),
            "K": K.tolist(),
            "flags": int(flags),
            "seq": int(seq),
            "glb_url": f"/api/audit/glb/{path.stem.replace('.lbm1', '')}" if path.name.endswith(".lbm1.mesh") else None,
        })
        pts_for_bounds = verts
        if include_points:
            step = max(1, int(np.ceil(num_v / max_points)))
            out["points"] = verts[::step].astype(np.float32).tolist()
            out["colors"] = rgb[::step].astype(np.uint8).tolist()
    else:
        raise ValueError(f"unsupported payload magic: {magic!r}")
    if pts_for_bounds.shape[0]:
        mn = pts_for_bounds.min(axis=0)
        mx = pts_for_bounds.max(axis=0)
        out["bounds"] = {"min": mn.astype(float).tolist(), "max": mx.astype(float).tolist()}
    return out


@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...), target_frames: int = 48):
    if STATE.worker is None:
        return JSONResponse({"ok": False, "error": "model not loaded"}, status_code=503)
    ext = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    if ext not in (".mp4", ".mov", ".webm", ".mkv", ".avi"):
        return JSONResponse({"ok": False, "error": f"unsupported ext {ext}"}, status_code=400)
    ts = int(time.time() * 1000)
    dest = UPLOAD_DIR / f"upload_{ts}{ext}"
    size = 0
    with dest.open("wb") as f:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            size += len(chunk)
    log.info("received upload %s (%.1f MB)", dest.name, size / 1e6)
    upload_id = dest.stem
    _jobs[upload_id] = {"status": "processing", "queued_at": time.time()}
    asyncio.get_event_loop().create_task(_process_upload_job(upload_id, dest, target_frames))
    return JSONResponse({"ok": True, "queued": True, "id": upload_id, "file": dest.name, "bytes": size})


async def _process_upload_job(upload_id: str, dest: Path, target_frames: int) -> None:
    try:
        t0 = time.time()
        _payload, _thumbs, info = await STATE.worker.process_video_file(dest, target_frames=target_frames)
        dt = time.time() - t0
        if _payload is not None:
            (UPLOAD_DIR / f"{upload_id}.lbp2").write_bytes(_payload)
            if _thumbs:
                (UPLOAD_DIR / f"{upload_id}.thumbs.json").write_text(
                    json.dumps([base64.b64encode(t).decode("ascii") for t in _thumbs])
                )
        model_id = next(iter(_model_registry()), "pxx")
        _jobs[upload_id] = {
            "status": "done" if _payload is not None else "error",
            "ok": _payload is not None,
            "id": upload_id,
            "file": dest.name,
            "coverage_url": f"/coverage.html?model={model_id}&upload={upload_id}",
            "report_url": f"/coverage-report.html?model={model_id}&upload={upload_id}",
            "coplay_url": f"/coplay?upload={upload_id}",
            "elapsed_s": round(dt, 2),
            **info,
        }
        log.info("job %s done in %.1fs ok=%s", upload_id, dt, _payload is not None)
        if _payload is not None:
            asyncio.get_event_loop().create_task(_build_coplay_for_upload(upload_id, dest))
    except Exception as e:
        log.exception("job %s failed", upload_id)
        _jobs[upload_id] = {"status": "error", "error": str(e)}


async def _render_demo_video(upload_id: str, raw_video: Path) -> Path | None:
    """Render the team-standard point-cloud demo video (H.264) for a raw upload.

    Phone uploads are often HEVC/4K which browsers cannot decode; the demo render
    re-projects the reconstruction into an H.264 follow+birdeye composite that the
    coplay PiP can play. Runs in the isolated .venv-lbdemo (micromamba) env.
    """
    script = ROOT.parent / "tools/render_demo_format.sh"
    prefix = ROOT.parent / ".venv-lbdemo"
    if not script.exists() or not prefix.exists():
        log.info("demo render skip %s: tool/env missing", upload_id)
        return None
    out_dir = UPLOAD_DIR / "_demo" / upload_id
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["bash", str(script), str(raw_video), str(out_dir), "8"]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=str(ROOT.parent),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        demo = _demo_video_path(upload_id)
        if proc.returncode == 0 and demo is not None:
            log.info("demo video rendered for %s → %s", upload_id, demo.name)
            return demo
        log.warning("demo render failed for %s: %s", upload_id, stdout.decode(errors="replace")[-400:])
    except Exception as e:
        log.warning("demo render error for %s: %s", upload_id, e)
    return None


async def _build_coplay_for_upload(upload_id: str, video_path: Path) -> None:
    """Build a per-upload coplay.html after GPU processing.

    Step 1: render the point-cloud demo video (PiP source must be the H.264 demo,
            not the raw HEVC upload). Step 2: build coplay.html with it.
    """
    if not _COPLAY_DTDX:
        log.info("coplay skip %s: no dtdx files configured", upload_id)
        return
    demo = await _render_demo_video(upload_id, video_path)
    render_video = demo or video_path  # fall back to raw if demo render unavailable
    out = UPLOAD_DIR / f"{upload_id}.coplay.html"
    cmd = [
        str(ROOT.parent / ".venv/bin/python"),
        str(ROOT.parent / "tools/build_coplay.py"),
        "--upload", upload_id,
        "--dtdx", *[str(d) for d in _COPLAY_DTDX],
        "--render-video", str(render_video),
        "--auto-pipe",
        "--out", str(out),
        "--base-url", "http://127.0.0.1:8767",
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(ROOT.parent),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode == 0:
            log.info("coplay built for %s → %s (video=%s)", upload_id, out.name, render_video.name)
        else:
            log.warning("coplay build failed for %s: %s", upload_id, stdout.decode(errors="replace")[-400:])
    except Exception as e:
        log.warning("coplay build error for %s: %s", upload_id, e)


@app.get("/api/jobs/{upload_id}/status")
async def job_status(upload_id: str):
    job = _jobs.get(upload_id)
    if job is None:
        return JSONResponse({"status": "unknown"}, status_code=404)
    return JSONResponse(job)


@app.post("/api/bim/upload")
async def upload_bim(file: UploadFile = File(...)):
    """Upload a BIM model file for projection overlay (glTF/GLB/OBJ)."""
    ext = Path(file.filename or "model.glb").suffix.lower() or ".glb"
    allowed = {".glb", ".gltf", ".obj"}
    if ext not in allowed:
        return JSONResponse({"ok": False, "error": f"unsupported ext {ext} (allowed: {sorted(allowed)})"},
                            status_code=400)
    ts = int(time.time() * 1000)
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(file.filename or "model").stem)[:48] or "model"
    bim_id = f"bim_{ts}_{safe_name}"
    dest = BIM_DIR / f"{bim_id}{ext}"
    size = 0
    with dest.open("wb") as f:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            size += len(chunk)
    log.info("BIM uploaded %s (%.1f MB)", dest.name, size / 1e6)
    return {"ok": True, "id": bim_id, "file": dest.name, "bytes": size, "ext": ext}


@app.get("/api/bim/list")
async def list_bim():
    """List uploaded BIM models, newest first."""
    items = []
    for ext in (".glb", ".gltf", ".obj"):
        for p in BIM_DIR.glob(f"bim_*{ext}"):
            stem = p.stem
            ts_ms = int(stem.split("_")[1]) if stem.count("_") >= 2 else 0
            items.append({
                "id": stem, "file": p.name, "bytes": p.stat().st_size,
                "ts_ms": ts_ms, "ext": ext,
            })
    items.sort(key=lambda x: x["ts_ms"], reverse=True)
    return {"bim": items}


@app.get("/api/bim/{bim_id}/file")
async def get_bim_file(bim_id: str):
    """Serve a BIM file by id."""
    if not bim_id.startswith("bim_") or "/" in bim_id or ".." in bim_id:
        return JSONResponse({"ok": False, "error": "bad id"}, status_code=400)
    for ext in (".glb", ".gltf", ".obj"):
        p = BIM_DIR / f"{bim_id}{ext}"
        if p.exists():
            media = {"glb": "model/gltf-binary", "gltf": "model/gltf+json",
                     "obj": "text/plain"}[ext[1:]]
            return FileResponse(p, media_type=media,
                                headers={"Cache-Control": "no-store"})
    return JSONResponse({"ok": False, "error": "not found"}, status_code=404)


@app.post("/api/bim/align")
async def align_bim(payload: dict):
    """Compute Sim(3) alignment from correspondences via umeyama.

    Body:
        {
          "correspondences": [
            {"bim": [x,y,z], "scan": [x,y,z]},
            ... (>= 3 pairs)
          ]
        }
    Response: { scale, rotation: [[..]*3], translation: [x,y,z], rmse, quality, n }
    """
    try:
        from lingbot_map.bim.alignment import solve_sim3_umeyama
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"alignment module missing: {e}"}, status_code=500)
    pairs = payload.get("correspondences", [])
    if len(pairs) < 3:
        return JSONResponse({"ok": False, "error": "need >=3 correspondences"}, status_code=400)
    try:
        src = np.array([p["bim"]  for p in pairs], dtype=np.float64)   # BIM-local
        dst = np.array([p["scan"] for p in pairs], dtype=np.float64)   # scan world
        result = solve_sim3_umeyama(src, dst)
        return {
            "ok": True,
            "scale": float(result.transform.scale),
            "rotation": result.transform.rotation.tolist(),
            "translation": result.transform.translation.tolist(),
            "rmse": float(result.rmse),
            "quality": result.quality,
            "n": int(result.n_correspondences),
            "per_point_residuals": result.per_point_residuals.tolist(),
        }
    except Exception as e:
        log.exception("alignment failed: %s", e)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.delete("/api/bim/{bim_id}")
async def delete_bim(bim_id: str):
    if not bim_id.startswith("bim_") or "/" in bim_id or ".." in bim_id:
        return JSONResponse({"ok": False, "error": "bad id"}, status_code=400)
    removed = []
    for ext in (".glb", ".gltf", ".obj"):
        p = BIM_DIR / f"{bim_id}{ext}"
        if p.exists():
            p.unlink(); removed.append(p.name)
    return {"ok": bool(removed), "removed": removed}


@app.get("/api/uploads")
async def list_uploads():
    """List past uploads with any persisted scan payload, newest first."""
    ids: set[str] = set()
    for pat in ("upload_*.lbp2", "upload_*.lbp4.*", "upload_*.lbm1.mesh", "upload_*.mp4", "upload_*.mov", "upload_*.webm", "upload_*.mkv", "upload_*.avi"):
        for p in UPLOAD_DIR.glob(pat):
            name = p.name
            if ".lbp4." in name:
                ids.add(name.split(".lbp4.", 1)[0])
            elif name.endswith(".lbm1.mesh"):
                ids.add(name.removesuffix(".lbm1.mesh"))
            else:
                ids.add(p.stem)
    items = []
    for upload_id in sorted(ids, key=lambda x: int(x.split("_")[-1]) if x.count("_") >= 1 and x.split("_")[-1].isdigit() else 0, reverse=True):
        # find matching video (any common ext)
        video = None
        for ext in (".mp4", ".mov", ".webm", ".mkv", ".avi"):
            cand = UPLOAD_DIR / f"{upload_id}{ext}"
            if cand.exists():
                video = cand.name
                break
        ts_ms = int(upload_id.split("_")[-1]) if upload_id.count("_") >= 1 else 0
        payloads = _upload_payloads(upload_id)
        items.append({
            "id": upload_id,
            "video": video,
            "payloads": payloads,
            "payload_bytes": max((v["bytes"] for v in payloads.values()), default=0),
            "ts_ms": ts_ms,
            "coverage_url": f"/coverage.html?model={_default_model_id()}&upload={upload_id}",
            "report_url": f"/coverage-report.html?model={_default_model_id()}&upload={upload_id}",
            "coplay_url": f"/coplay?upload={upload_id}",
        })
    return {"uploads": items}


def _upload_payloads(upload_id: str) -> dict:
    payloads = {}
    for key, p in {
        "lbp2": UPLOAD_DIR / f"{upload_id}.lbp2",
        "mesh": UPLOAD_DIR / f"{upload_id}.lbm1.mesh",
        "detail": UPLOAD_DIR / f"{upload_id}.lbp4.test",
        "texture": UPLOAD_DIR / f"{upload_id}.lbp4.texture",
        "visual": UPLOAD_DIR / f"{upload_id}.lbp4.visual",
        "mesh_glb": UPLOAD_DIR / f"{upload_id}.mesh.glb",
    }.items():
        if p.exists():
            payloads[key] = {"file": p.name, "bytes": p.stat().st_size}
    return payloads


def _load_upload_thumbs(upload_id: str, *, prefer_audit: bool = False) -> list[bytes] | None:
    candidates = []
    if prefer_audit:
        candidates.append(UPLOAD_DIR / f"{upload_id}.lbm1.thumbs.json")
        candidates.append(UPLOAD_DIR / f"{upload_id}.lbp4.thumbs.json")
    candidates.append(UPLOAD_DIR / f"{upload_id}.thumbs.json")
    tp = next((p for p in candidates if p.exists()), None)
    if tp is None:
        return None
    try:
        return [base64.b64decode(s) for s in json.loads(tp.read_text())]
    except Exception as e:
        log.warning("thumbs load failed for %s: %s", upload_id, e)
        return None


def _upload_video_path(upload_id: str) -> Path | None:
    for ext in (".mp4", ".mov", ".webm", ".mkv", ".avi"):
        p = UPLOAD_DIR / f"{upload_id}{ext}"
        if p.exists():
            return p
    return None


def _demo_video_path(upload_id: str) -> Path | None:
    """Browser-playable point-cloud demo render (H.264) for an upload, if built."""
    p = UPLOAD_DIR / "_demo" / upload_id / f"{upload_id}_demo_format.mp4"
    return p if p.exists() else None


def _upload_ts_ms(upload_id: str) -> int:
    try:
        return int(upload_id.split("_")[-1]) if upload_id.count("_") >= 1 else 0
    except Exception:
        return 0


def _video_timing_metadata(upload_id: str, sample_count: int = 0) -> dict:
    video = _upload_video_path(upload_id)
    if video is None:
        return {}
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return {"video_file": video.name}
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    duration_s = (total / fps) if total > 0 and fps > 0 else None
    out: dict = {
        "video_file": video.name,
        "video_frame_count": total,
        "video_fps": round(fps, 3) if fps > 0 else None,
        "video_duration_s": round(duration_s, 3) if duration_s is not None else None,
        "upload_ts_ms": _upload_ts_ms(upload_id),
    }
    if sample_count > 0 and total > 0:
        raw_indices = _uniform_video_indices(total, sample_count)
        out["raw_frame_indices"] = raw_indices
        if fps > 0:
            out["frame_timestamps_s"] = [round(idx / fps, 3) for idx in raw_indices]
    return out


def _scan_frame_count(upload_id: str) -> int:
    p = _select_scan_payload(upload_id, "mesh")
    if p is None:
        p = _select_scan_payload(upload_id)
    if p is not None:
        try:
            scan = _parse_scan_payload(p, include_points=False)
            return int(len(scan.get("poses") or []))
        except Exception as e:
            log.warning("scan frame count failed for %s: %s", upload_id, e)
    thumbs = _load_upload_thumbs(upload_id, prefer_audit=True)
    return len(thumbs) if thumbs else 0


def _uniform_video_indices(total_frames: int, sample_count: int) -> list[int]:
    if total_frames <= 0 or sample_count <= 0:
        return []
    n = min(sample_count, total_frames)
    if n == 1:
        return [total_frames // 2]
    return [int(round(i * (total_frames - 1) / (n - 1))) for i in range(n)]


def _extract_keyframes(
    upload_id: str,
    frame_ids: list[int],
    *,
    max_width: int = 1280,
    quality: int = 90,
) -> list[dict]:
    video = _upload_video_path(upload_id)
    if video is None:
        raise FileNotFoundError(f"video not found for {upload_id}")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video.name}")
    rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1.0)
    except Exception:
        pass
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    sample_count = _scan_frame_count(upload_id) or max(frame_ids, default=-1) + 1
    raw_indices = _uniform_video_indices(total, sample_count)
    rot_map = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    rot_code = rot_map.get(rotation)
    out_dir = UPLOAD_DIR / f"{upload_id}.keyframes"
    out_dir.mkdir(exist_ok=True)
    out: list[dict] = []
    try:
        for frame_id in sorted({int(x) for x in frame_ids if int(x) >= 0}):
            if frame_id >= len(raw_indices):
                continue
            raw_idx = int(raw_indices[frame_id])
            dest = out_dir / f"frame_{frame_id:06d}.jpg"
            if not dest.exists():
                cap.set(cv2.CAP_PROP_POS_FRAMES, raw_idx)
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    continue
                if rot_code is not None and rotation in (90, 270) and bgr.shape[1] > bgr.shape[0]:
                    bgr = cv2.rotate(bgr, rot_code)
                elif rot_code == cv2.ROTATE_180:
                    bgr = cv2.rotate(bgr, rot_code)
                h, w = bgr.shape[:2]
                if max_width > 0 and w > max_width:
                    scale = max_width / float(w)
                    bgr = cv2.resize(bgr, (max_width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
                if not ok:
                    continue
                dest.write_bytes(buf.tobytes())
            try:
                img = cv2.imread(str(dest))
                height, width = img.shape[:2] if img is not None else (None, None)
            except Exception:
                height, width = None, None
            out.append({
                "frame": frame_id,
                "raw_frame": raw_idx,
                "timestamp_s": round(raw_idx / fps, 3) if fps > 0 else None,
                "file": dest.name,
                "url": f"/api/uploads/{upload_id}/keyframes/{frame_id}.jpg",
                "width": width,
                "height": height,
                "bytes": dest.stat().st_size if dest.exists() else 0,
            })
    finally:
        cap.release()
    return out


@app.get("/api/models")
async def list_models():
    """List built-in reference models under /models with manifest validation."""
    items = []
    registry = _model_registry()
    for model_id in registry:
        try:
            items.append(_model_manifest(model_id, persist=True))
        except Exception as e:
            spec = registry[model_id]
            items.append({
                "model_id": model_id,
                "name": spec["name"],
                "glb": spec["glb"],
                "metadata": spec["metadata"],
                "manifest_valid": False,
                "error": str(e),
            })
    return {"items": items}


@app.get("/api/models/{model_id}/objects")
async def model_objects(model_id: str, limit: int = 5000):
    """Return PAG-derived proxy objects for a built-in model."""
    if model_id not in _model_registry():
        return JSONResponse({"ok": False, "error": "unknown model_id"}, status_code=404)
    try:
        manifest, objects = _load_model_objects(model_id)
        return {"ok": True, "manifest": _model_manifest(model_id), "objects": objects[:max(0, limit)]}
    except Exception as e:
        log.exception("model objects failed: %s", e)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/api/uploads/{upload_id}/video")
async def upload_video_file(upload_id: str, raw: bool = False):
    """Serve the point-cloud demo render (H.264, browser-playable) for coplay PiP.

    Falls back to the raw upload when the demo render is not ready or raw=true is
    requested. The raw phone video may be HEVC/4K and not decode in browsers.
    """
    p = None if raw else _demo_video_path(upload_id)
    if p is None:
        p = _upload_video_path(upload_id)
    if p is None:
        return JSONResponse({"error": "video not found"}, status_code=404)
    media = "video/mp4" if p.suffix.lower() == ".mp4" else "application/octet-stream"
    return FileResponse(p, media_type=media, filename=p.name, headers={"Cache-Control": "no-store"})


@app.get("/api/uploads/{upload_id}/scan")
async def upload_scan(
    upload_id: str,
    variant: str | None = None,
    include_points: bool = False,
    include_thumbs: bool = True,
    max_points: int = 120_000,
):
    """Return persisted scan metadata: poses, intrinsics, counts, thumbnails."""
    if _bad_id(upload_id, "upload_"):
        return JSONResponse({"ok": False, "error": "bad id"}, status_code=400)
    p = _select_scan_payload(upload_id, variant)
    if p is None:
        return JSONResponse({"ok": False, "error": "no scan payload found"}, status_code=404)
    try:
        scan = _parse_scan_payload(p, include_points=include_points, max_points=max(1_000, min(int(max_points), 500_000)))
    except Exception as e:
        log.exception("scan parse failed: %s", e)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    thumbs = _load_upload_thumbs(upload_id, prefer_audit=True) if include_thumbs else None
    sample_count = max(
        len(scan.get("poses") or []),
        len(thumbs) if thumbs else 0,
    )
    scan.update(_video_timing_metadata(upload_id, sample_count=sample_count))
    scan.update({
        "ok": True,
        "id": upload_id,
        "mesh_glb_url": f"/api/uploads/{upload_id}/mesh.glb" if (UPLOAD_DIR / f"{upload_id}.mesh.glb").exists() else None,
        "thumbs": [base64.b64encode(t).decode("ascii") for t in thumbs] if thumbs else [],
        "thumb_count": len(thumbs) if thumbs else 0,
    })
    return scan


@app.get("/api/uploads/{upload_id}/mesh.glb")
async def upload_scan_glb(upload_id: str):
    if _bad_id(upload_id, "upload_"):
        return JSONResponse({"ok": False, "error": "bad id"}, status_code=400)
    p = UPLOAD_DIR / f"{upload_id}.mesh.glb"
    if not p.exists():
        return JSONResponse({"ok": False, "error": "GLB mesh not found"}, status_code=404)
    return FileResponse(p, media_type="model/gltf-binary", filename=p.name, headers={"Cache-Control": "no-store"})


@app.get("/api/uploads/{upload_id}/keyframes")
async def upload_keyframes(upload_id: str, frames: str = "", max_width: int = 1280):
    if _bad_id(upload_id, "upload_"):
        return JSONResponse({"ok": False, "error": "bad id"}, status_code=400)
    try:
        if frames.strip():
            frame_ids = [int(x) for x in re.split(r"[, ]+", frames.strip()) if x != ""]
        else:
            frame_ids = list(range(min(_scan_frame_count(upload_id), 12)))
    except ValueError:
        return JSONResponse({"ok": False, "error": "frames must be comma-separated integers"}, status_code=400)
    try:
        items = _extract_keyframes(upload_id, frame_ids, max_width=max(0, min(int(max_width), 4096)))
        return {
            "ok": True,
            "id": upload_id,
            "count": len(items),
            "items": items,
            "dir": f"{upload_id}.keyframes",
        }
    except FileNotFoundError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=404)
    except Exception as e:
        log.exception("keyframe extraction failed: %s", e)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/api/uploads/{upload_id}/keyframes/{frame_idx}.jpg")
async def upload_keyframe_image(upload_id: str, frame_idx: int):
    if _bad_id(upload_id, "upload_"):
        return JSONResponse({"ok": False, "error": "bad id"}, status_code=400)
    if frame_idx < 0:
        return JSONResponse({"ok": False, "error": "bad frame"}, status_code=400)
    p = UPLOAD_DIR / f"{upload_id}.keyframes" / f"frame_{frame_idx:06d}.jpg"
    if not p.exists():
        try:
            items = _extract_keyframes(upload_id, [frame_idx])
        except FileNotFoundError as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=404)
        except Exception as e:
            log.exception("keyframe extraction failed: %s", e)
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
        if not items or not p.exists():
            return JSONResponse({"ok": False, "error": "keyframe not found"}, status_code=404)
    return FileResponse(p, media_type="image/jpeg", filename=p.name, headers={"Cache-Control": "no-store"})


def _viewer_points(points: np.ndarray) -> np.ndarray:
    out = points.astype(np.float64, copy=True)
    out[:, 1] *= -1.0
    out[:, 2] *= -1.0
    return out


def _apply_sim3(points: np.ndarray, transform: dict) -> np.ndarray:
    s = float(transform.get("scale", 1.0))
    R = np.asarray(transform.get("rotation", np.eye(3)), dtype=np.float64)
    t = np.asarray(transform.get("translation", [0, 0, 0]), dtype=np.float64)
    return s * (R @ points.T).T + t


def _bounds_summary_from_points(points: np.ndarray | list) -> dict | None:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.size == 0:
        return None
    arr = arr[np.isfinite(arr).all(axis=1)]
    if arr.size == 0:
        return None
    mn = arr.min(axis=0)
    mx = arr.max(axis=0)
    span = mx - mn
    center = (mn + mx) * 0.5
    return {
        "min": mn.astype(float).tolist(),
        "max": mx.astype(float).tolist(),
        "span": span.astype(float).tolist(),
        "center": center.astype(float).tolist(),
        "diagonal_m": float(np.linalg.norm(span)),
    }


def _scan_viewer_bounds(scan: dict) -> dict | None:
    bounds = scan.get("bounds") if isinstance(scan, dict) else None
    if not isinstance(bounds, dict) or "min" not in bounds or "max" not in bounds:
        return None
    mn = np.asarray(bounds.get("min"), dtype=np.float64)
    mx = np.asarray(bounds.get("max"), dtype=np.float64)
    if mn.shape != (3,) or mx.shape != (3,) or not np.isfinite(mn).all() or not np.isfinite(mx).all():
        return None
    # Viewer/picking coordinates mirror the payload Y/Z axes before scan-to-model Sim(3).
    viewer_min = np.array([mn[0], -mx[1], -mx[2]], dtype=np.float64)
    viewer_max = np.array([mx[0], -mn[1], -mn[2]], dtype=np.float64)
    return _bounds_summary_from_points(np.vstack([viewer_min, viewer_max]))


def _bounds_corners(bounds: dict | None) -> np.ndarray:
    if not bounds or "min" not in bounds or "max" not in bounds:
        return np.empty((0, 3), dtype=np.float64)
    mn = np.asarray(bounds.get("min"), dtype=np.float64)
    mx = np.asarray(bounds.get("max"), dtype=np.float64)
    if mn.shape != (3,) or mx.shape != (3,) or not np.isfinite(mn).all() or not np.isfinite(mx).all():
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray([
        [mn[0], mn[1], mn[2]],
        [mn[0], mn[1], mx[2]],
        [mn[0], mx[1], mn[2]],
        [mn[0], mx[1], mx[2]],
        [mx[0], mn[1], mn[2]],
        [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mn[2]],
        [mx[0], mx[1], mx[2]],
    ], dtype=np.float64)


def _model_object_bounds(objects: list[dict]) -> dict | None:
    pts: list[list[float]] = []
    for obj in objects:
        for key in ("center", "start", "end"):
            value = obj.get(key)
            if isinstance(value, list) and len(value) == 3:
                pts.append(value)
        bbox = obj.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 2:
            for value in bbox:
                if isinstance(value, list) and len(value) == 3:
                    pts.append(value)
    return _bounds_summary_from_points(pts)


def _viewer_pose(pose12) -> tuple[np.ndarray, np.ndarray]:
    c2w = np.asarray(pose12, dtype=np.float64)
    if c2w.shape[0] != 12:
        raise ValueError("pose must have 12 floats")
    R = np.array([
        [c2w[0], -c2w[1], -c2w[2]],
        [-c2w[4], c2w[5], c2w[6]],
        [-c2w[8], c2w[9], c2w[10]],
    ], dtype=np.float64)
    t = np.array([c2w[3], -c2w[7], -c2w[11]], dtype=np.float64)
    return R, t


def _aligned_camera_states(poses, alignment: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    A = np.asarray(alignment.get("rotation", np.eye(3)), dtype=np.float64)
    s = float(alignment.get("scale", 1.0))
    at = np.asarray(alignment.get("translation", [0, 0, 0]), dtype=np.float64)
    states = []
    for pose in poses or []:
        R, t = _viewer_pose(pose)
        states.append((A @ R, s * (A @ t) + at))
    return states


def _pose_descriptor(poses) -> dict:
    positions = []
    for pose in poses or []:
        try:
            _R, t = _viewer_pose(pose)
            positions.append(t)
        except Exception:
            continue
    if not positions:
        return {
            "pose_count": 0,
            "path_length_m": 0.0,
            "displacement_m": 0.0,
            "bounds": None,
        }
    pts = np.vstack(positions)
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1) if pts.shape[0] > 1 else np.zeros(0)
    return {
        "pose_count": int(pts.shape[0]),
        "path_length_m": float(steps.sum()),
        "displacement_m": float(np.linalg.norm(pts[-1] - pts[0])) if pts.shape[0] > 1 else 0.0,
        "bounds": _bounds_summary_from_points(pts),
    }


def _model_repetition_risk(objects: list[dict]) -> dict:
    counts: dict[str, int] = {}
    system_counts: dict[str, int] = {}
    for obj in objects:
        category = str(obj.get("category") or "Unknown")
        counts[category] = counts.get(category, 0) + 1
        system = obj.get("system")
        if system:
            key = f"{category}:{system}"
            system_counts[key] = system_counts.get(key, 0) + 1
    repeated_categories = sorted(
        ({"category": k, "count": v} for k, v in counts.items() if v >= 8),
        key=lambda item: item["count"],
        reverse=True,
    )[:8]
    repeated_systems = sorted(
        ({"group": k, "count": v} for k, v in system_counts.items() if v >= 8),
        key=lambda item: item["count"],
        reverse=True,
    )[:8]
    max_count = max(counts.values(), default=0)
    level = "high" if max_count >= 40 or len(repeated_categories) >= 3 else "medium" if max_count >= 12 else "low"
    return {
        "level": level,
        "object_count": len(objects),
        "max_category_count": max_count,
        "repeated_categories": repeated_categories,
        "repeated_systems": repeated_systems,
        "note": (
            "repeated model geometry can produce multiple plausible global positions"
            if level in ("high", "medium")
            else "repetition risk is low for this model subset"
        ),
    }


def _bounds_alignment_score(aligned_bounds: dict | None, model_bounds: dict | None) -> dict:
    if not aligned_bounds or not model_bounds:
        return {"score": 0.0, "center_distance_m": None, "span_ratio": None}
    ac = np.asarray(aligned_bounds.get("center"), dtype=np.float64)
    mc = np.asarray(model_bounds.get("center"), dtype=np.float64)
    if ac.shape != (3,) or mc.shape != (3,):
        return {"score": 0.0, "center_distance_m": None, "span_ratio": None}
    center_distance = float(np.linalg.norm(ac - mc))
    model_diag = max(float(model_bounds.get("diagonal_m") or 0.0), 1e-6)
    aligned_diag = max(float(aligned_bounds.get("diagonal_m") or 0.0), 1e-6)
    center_score = float(np.exp(-center_distance / model_diag))
    span_ratio = aligned_diag / model_diag
    span_score = float(np.exp(-abs(np.log(max(span_ratio, 1e-6)))))
    score = float(0.65 * center_score + 0.35 * span_score)
    return {
        "score": round(score, 4),
        "center_distance_m": center_distance,
        "span_ratio": span_ratio,
    }


def _candidate_from_alignment(
    *,
    candidate_id: str,
    source: str,
    alignment: dict,
    scan_bounds: dict | None,
    model_bounds: dict | None,
    reason: str,
    upload_id: str | None = None,
    generated_at: str | None = None,
    can_save: bool = False,
) -> dict:
    aligned_bounds = None
    if scan_bounds:
        corners = _bounds_corners(scan_bounds)
        if corners.size:
            aligned_bounds = _bounds_summary_from_points(_apply_sim3(corners, alignment))
    score_info = _bounds_alignment_score(aligned_bounds, model_bounds)
    quality = alignment.get("quality")
    base_score = float(score_info.get("score") or 0.0)
    if quality == "green":
        base_score = max(base_score, 0.78)
    elif quality == "yellow":
        base_score = max(base_score, 0.62)
    confidence = "high" if base_score >= 0.78 and quality == "green" else "medium" if base_score >= 0.55 else "low"
    return {
        "id": candidate_id,
        "source": source,
        "upload_id": upload_id,
        "generated_at": generated_at,
        "quality": quality,
        "score": round(base_score, 4),
        "confidence": confidence,
        "can_save": can_save,
        "requires_user_approval": source != "manual_pairs_preview",
        "reason": reason,
        "alignment": {
            k: alignment.get(k)
            for k in (
                "quality",
                "stable",
                "rmse_m",
                "max_leave_one_out_rmse_m",
                "correspondence_spread",
                "scale",
                "rotation",
                "translation",
                "n",
                "per_point_residuals",
                "pairs",
            )
            if k in alignment
        },
        "aligned_scan_bounds": aligned_bounds,
        "center_distance_m": score_info.get("center_distance_m"),
        "span_ratio": score_info.get("span_ratio"),
    }


def _previous_alignment_candidates(
    model_id: str,
    current_upload_id: str,
    scan_bounds: dict | None,
    model_bounds: dict | None,
    *,
    limit: int = 5,
) -> list[dict]:
    paths = set(UPLOAD_DIR.glob(f"upload_*.{model_id}.alignment.json"))
    paths.update(UPLOAD_DIR.glob("upload_*.alignment.json"))
    items = []
    for path in sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            doc = _read_json(path)
        except Exception:
            continue
        if not isinstance(doc, dict) or _json_model_id(doc) != model_id:
            continue
        upload_id = str(doc.get("upload_id") or path.name.split(".", 1)[0])
        if upload_id == current_upload_id:
            continue
        alignment = doc.get("alignment") if isinstance(doc.get("alignment"), dict) else None
        if not alignment or alignment.get("quality") not in ("green", "yellow"):
            continue
        candidate = _candidate_from_alignment(
            candidate_id=f"previous:{upload_id}",
            source="previous_upload_alignment",
            upload_id=upload_id,
            generated_at=doc.get("generated_at"),
            alignment=alignment,
            scan_bounds=scan_bounds,
            model_bounds=model_bounds,
            reason="validated alignment from another upload on the same model; use only as a prior and confirm with anchors",
            can_save=False,
        )
        items.append(candidate)
        if len(items) >= limit:
            break
    return items


def _model_surface_samples(objects: list[dict]) -> np.ndarray:
    pts = []
    for obj in objects:
        samples, _ts = _object_samples(obj)
        if samples.size:
            pts.append(samples)
    if not pts:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(pts)


def _object_surface_radius(obj: dict) -> float:
    """중심선/중심점 샘플 -> 표면 거리 보정용 반경 (placement 채점·coverage 공용)."""
    r = float(obj.get("diameter_m") or 0.0) * 0.5
    if obj.get("is_linear"):
        return max(0.0, min(r, 0.4))
    # bbox 기반 point 객체는 중심 거리에 외형이 일부 반영되므로 보수적으로
    return max(0.0, min(r * 0.5, 0.2))


def _model_surface_samples_with_radius(objects: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Sample points plus per-sample surface radius (centerline->surface correction)."""
    pts = []
    rads = []
    for obj in objects:
        samples, _ts = _object_samples(obj)
        if not samples.size:
            continue
        r = _object_surface_radius(obj)
        pts.append(samples)
        rads.append(np.full(len(samples), r, dtype=np.float64))
    if not pts:
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.float64)
    return np.vstack(pts), np.concatenate(rads)


def _corroborate_alignment(
    scan_points: np.ndarray,
    model_tree,
    alignment: dict,
    *,
    inlier_m: float = 0.30,
    max_points: int = 40_000,
    sample_radii: np.ndarray | None = None,
) -> dict:
    """Score a candidate transform by NN distance of aligned scan points to model samples.
    With sample_radii, centerline distances are reduced to surface distances.
    Verification only: never mutates the transform."""
    if scan_points.size == 0 or model_tree is None:
        return {"ok": False, "reason": "no scan points or model samples"}
    pts = scan_points
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts = pts[idx]
    aligned = _apply_sim3(pts, alignment)
    d, nn_idx = model_tree.query(aligned, k=1, workers=-1)
    if sample_radii is not None and len(sample_radii):
        # 표면 거리: 중심선 거리 d가 반경 r과 같을 때(표면 위)만 0. 내부/원거리 모두 벌점.
        d = np.abs(d - sample_radii[np.asarray(nn_idx, dtype=np.int64)])
    return {
        "ok": True,
        "n_points": int(len(pts)),
        "median_nn_m": round(float(np.median(d)), 4),
        "inlier_ratio": round(float((d <= inlier_m).mean()), 4),
        "inlier_threshold_m": inlier_m,
        "radius_corrected": bool(sample_radii is not None and len(sample_radii)),
    }


def _icp_refine_alignment(
    scan_points: np.ndarray,
    model_samples: np.ndarray,
    model_tree,
    alignment: dict,
    *,
    trim_m: float = 0.5,
    max_points: int = 20_000,
) -> dict | None:
    """Refine a prior Sim(3) with rigid ICP against model surface samples (scale frozen).
    Clutter is trimmed by keeping only seeded points within trim_m of the model."""
    from registration import icp_refine_rigid
    if scan_points.size == 0 or model_samples.size == 0 or model_tree is None:
        return None
    pts = scan_points
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts = pts[idx]
    seeded = _apply_sim3(pts, alignment)
    d, _ = model_tree.query(seeded, k=1, workers=-1)
    near = seeded[d <= trim_m]
    if len(near) < 300:
        return None
    R_i, t_i, _icp_rmse = icp_refine_rigid(near.astype(np.float64), model_samples.astype(np.float64))
    R_i = np.asarray(R_i, dtype=np.float64)
    t_i = np.asarray(t_i, dtype=np.float64)
    R0 = np.asarray(alignment.get("rotation", np.eye(3)), dtype=np.float64)
    t0 = np.asarray(alignment.get("translation", [0, 0, 0]), dtype=np.float64)
    refined = dict(alignment)
    refined["rotation"] = (R_i @ R0).tolist()
    refined["translation"] = (R_i @ t0 + t_i).tolist()
    refined["note"] = "icp_refined_from_prior"
    return refined


def _auto_place_candidates(
    upload_id: str,
    model_objects: list[dict],
    *,
    scale: float = 1.0,
    start_hint: list | None = None,
    direction_az: float | None = None,
    max_candidates: int = 3,
) -> tuple[list[dict], list[str]]:
    """중력 정렬(+metric scale) 기하 탐색으로 scan->model 배치 가설을 만든다.
    자동이 원칙이며, start_hint(모델 좌표 1점)는 탐색 범위를 좁히는 보조 수단이다.
    결과는 가설 후보(quality yellow/red)로, 명시적 저장 전에는 정합이 아니다."""
    warnings: list[str] = []
    scan_path = _select_scan_payload(upload_id, "detail")
    if scan_path is None:
        return [], ["scan payload not found"]
    scan = _parse_scan_payload(scan_path, include_points=True, max_points=120_000)
    pts_raw = np.asarray(scan.get("points") or [], dtype=np.float64)
    poses = scan.get("poses") or []
    if pts_raw.size == 0 or len(poses) < 5:
        return [], ["auto placement needs scan points and >=5 camera poses"]
    pts = _viewer_points(pts_raw)
    sub = pts[np.linspace(0, len(pts) - 1, min(6000, len(pts))).astype(int)]

    ups = []
    for p in poses:
        R, _t = _viewer_pose(p)
        ups.append(R @ np.array([0.0, 1.0, 0.0]))
    g = np.mean(ups, axis=0)
    gn = float(np.linalg.norm(g))
    if gn < 1e-6:
        return [], ["camera up axis unstable; cannot gravity-align"]
    g /= gn
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(g, z)
    c = float(g @ z)
    if np.linalg.norm(v) < 1e-8:
        R0 = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R0 = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))

    samples, sample_radii = _model_surface_samples_with_radius(model_objects)
    if samples.size == 0:
        return [], ["model has no geometry samples"]
    from scipy.spatial import cKDTree
    tree = cKDTree(samples)
    # M2: 중력 정렬 프레임에서 수평 대평면(바닥/천장) 클러터 제거 후 채점
    pts_g = (R0 @ sub.T).T
    zbin = np.round(pts_g[:, 2] / 0.05).astype(int)
    vals, counts = np.unique(zbin, return_counts=True)
    clutter = np.zeros(len(sub), dtype=bool)
    for zb_val, n in sorted(zip(vals.tolist(), counts.tolist()), key=lambda x: -x[1])[:3]:
        if n >= 0.10 * len(sub):
            clutter |= np.abs(zbin - zb_val) <= 1
    if clutter.any() and (len(sub) - int(clutter.sum())) >= 1500:
        sub_score = sub[~clutter]
        warnings.append(f"clutter planes removed for scoring: {int(clutter.sum())}/{len(sub)} pts")
    else:
        sub_score = sub
    scan_c = sub_score.mean(0)

    def rz(deg: float) -> np.ndarray:
        r = np.deg2rad(deg)
        cs, sn = np.cos(r), np.sin(r)
        return np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]])

    centers: list[np.ndarray] = []
    if start_hint is not None and len(start_hint) == 3:
        hx, hy, hz = (float(x) for x in start_hint)
        for dx in (-1.5, 0.0, 1.5):
            for dy in (-1.5, 0.0, 1.5):
                centers.append(np.array([hx + dx, hy + dy, hz]))
    else:
        centers_z = np.asarray([o["center"][2] for o in model_objects if o.get("center")], dtype=np.float64)
        if centers_z.size:
            hist, edges = np.histogram(centers_z, bins=24)
            zmode = float((edges[hist.argmax()] + edges[hist.argmax() + 1]) * 0.5)
        else:
            zs_all = samples[:, 2]
            hist, edges = np.histogram(zs_all, bins=24)
            zmode = float((edges[hist.argmax()] + edges[hist.argmax() + 1]) * 0.5)
        band = samples[np.abs(samples[:, 2] - zmode) < 2.5]
        if band.size == 0:
            band = samples
        xq = [float(np.quantile(band[:, 0], q)) for q in (0.15, 0.35, 0.5, 0.65, 0.85)]
        yq = [float(np.quantile(band[:, 1], q)) for q in (0.1, 0.3, 0.5, 0.7, 0.9)]
        for zc in (zmode - 1.0, zmode, zmode + 1.0):
            for cx in xq:
                for cy in yq:
                    centers.append(np.array([cx, cy, zc]))

    # M3: yaw 사전후보 — 모델 배관 방위 분포(길이 가중) vs scan 수평 주축
    yaws: list[float] = [float(a) for a in range(0, 360, 20)]
    lin_dirs = []
    for o in model_objects:
        if o.get("is_linear") and o.get("start") and o.get("end"):
            dvec = np.asarray(o["end"], dtype=np.float64) - np.asarray(o["start"], dtype=np.float64)
            L = float(np.linalg.norm(dvec[:2]))
            if L > 0.3:
                lin_dirs.append((float(np.degrees(np.arctan2(dvec[1], dvec[0])) % 180.0), L))
    if direction_az is not None and len(sub_score) >= 30:
        # 사용자가 지정한 촬영 방향으로 yaw를 고정한다(±refine). scan 수평 주축에 그 방향을
        # 맞춰 후보를 만들어 전역 yaw 스윕의 모호성을 제거한다(방향 미지정 시 기존 로직 사용).
        sg_xy = (R0 @ sub_score.T).T[:, :2]
        sg_xy = sg_xy - sg_xy.mean(0)
        evals, evecs = np.linalg.eigh(sg_xy.T @ sg_xy)
        # primary(최대 분산) 수평 주축 1개만 사용한다. 양 직교 축을 모두 쓰면 90도 모호성으로
        # 사용자 방향이 상쇄돼 무의미해진다. flip(±180)은 PCA 축의 부호 모호성 보정용.
        sax = float(np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1])) % 180.0)
        m = float(direction_az) % 180.0
        cand_yaws = set()
        for flip in (0.0, 180.0):
            base = (m - sax + flip) % 360.0
            for dd in (-8.0, -4.0, 0.0, 4.0, 8.0):
                cand_yaws.add(round((base + dd) % 360.0, 1))
        yaws = sorted(cand_yaws)
        warnings.append(f"yaw from user direction {round(m,1)}deg: {len(yaws)} candidates")
    elif lin_dirs and len(sub_score) >= 100:
        hist36 = np.zeros(36)
        for az, w in lin_dirs:
            hist36[int(az // 5) % 36] += w
        model_modes = [float(b * 5 + 2.5) for b in np.argsort(hist36)[::-1][:2] if hist36[b] > 0]
        sg_xy = (R0 @ sub_score.T).T[:, :2]
        sg_xy = sg_xy - sg_xy.mean(0)
        evals, evecs = np.linalg.eigh(sg_xy.T @ sg_xy)
        scan_axes = [float(np.degrees(np.arctan2(evecs[1, i], evecs[0, i])) % 180.0) for i in (1, 0)]
        cand_yaws: set[float] = set()
        for m in model_modes:
            for sax in scan_axes:
                for flip in (0.0, 180.0):
                    base = (m - sax + flip) % 360.0
                    for dd in (-10.0, 0.0, 10.0):
                        cand_yaws.add(round((base + dd) % 360.0, 1))
        if cand_yaws:
            yaws = sorted(cand_yaws)
            warnings.append(f"yaw prior from pipe azimuths: {len(yaws)} candidates")

    # 단안(up-to-scale) 재구성이라 절대 scale이 미지수다. start_hint(가이드 모드)에서는
    # scale도 함께 탐색해 배관 겹침(표면 거리)이 최소가 되는 배율을 찾는다. scan center를
    # 피벗으로 t를 맞춰 scale을 바꿔도 시작점은 힌트에 고정된다. 힌트 없는 경로는 기존대로 고정.
    if start_hint is not None and len(start_hint) == 3:
        scale_cands = sorted({round(scale * f, 4) for f in
                              (0.5, 0.65, 0.8, 0.9, 1.0, 1.15, 1.35, 1.6, 2.0)})
        warnings.append(f"scale search enabled (guided): {len(scale_cands)} candidates")
    else:
        scale_cands = [scale]

    scored: list[tuple[dict, dict, float]] = []
    for cgrid in centers:
        for a in yaws:
            R = rz(a) @ R0
            for s in scale_cands:
                t = cgrid - s * (R @ scan_c)
                al = {"scale": s, "rotation": R.tolist(), "translation": t.tolist()}
                sc = _corroborate_alignment(sub_score, tree, al, max_points=3000, sample_radii=sample_radii)
                if sc.get("ok"):
                    scored.append((sc, al, a))
    if not scored:
        return [], ["auto placement search produced no candidates"]
    scored.sort(key=lambda x: x[0]["median_nn_m"])

    picked: list[tuple[dict, dict, int]] = []
    for sc, al, a in scored:
        distinct = True
        for _sc2, al2, a2 in picked:
            dt = float(np.linalg.norm(np.asarray(al["translation"]) - np.asarray(al2["translation"])))
            da = min(abs(a - a2), 360 - abs(a - a2))
            if dt < 2.0 and da < 30:
                distinct = False
                break
        if distinct:
            picked.append((sc, al, a))
        if len(picked) >= max_candidates:
            break

    out: list[dict] = []
    for rank, (sc, al, _a) in enumerate(picked):
        cur = sc
        for _ in range(3):
            improved = False
            for d in ([0.4, 0, 0], [-0.4, 0, 0], [0, 0.4, 0], [0, -0.4, 0], [0, 0, 0.25], [0, 0, -0.25]):
                a2 = dict(al)
                a2["translation"] = (np.asarray(al["translation"]) + d).tolist()
                s2 = _corroborate_alignment(sub_score, tree, a2, max_points=3000, sample_radii=sample_radii)
                if s2["median_nn_m"] < cur["median_nn_m"]:
                    al, cur, improved = a2, s2, True
            if not improved:
                break
        final = _corroborate_alignment(sub_score, tree, al, max_points=20_000, sample_radii=sample_radii)
        # 표면 거리 척도 기준 캘리브레이션: +3m 오배치가 yellow를 통과하지 못하는 값
        quality = "yellow" if (final.get("median_nn_m", 9e9) <= 0.25 and final.get("inlier_ratio", 0.0) >= 0.50) else "red"
        alignment = {
            "quality": quality,
            "stable": False,
            "rmse_m": None,
            "scale": scale,
            "rotation": al["rotation"],
            "translation": al["translation"],
            "n": 0,
            "note": "auto_geometric_gravity_aligned",
            "pairs": [],
        }
        cand = _candidate_from_alignment(
            candidate_id=f"auto:{rank}",
            source="auto_geometric",
            alignment=alignment,
            scan_bounds=None,
            model_bounds=None,
            reason="gravity-aligned geometric hypothesis; preview and save explicitly",
            can_save=quality != "red",
        )
        cand["corroboration"] = final
        cand["corroborated"] = quality != "red"
        cand["score"] = round(max(0.0, 1.0 - float(final.get("median_nn_m", 1.0))), 4)
        out.append(cand)
    if len(out) >= 2:
        m0 = out[0]["corroboration"]["median_nn_m"]
        m1 = out[1]["corroboration"]["median_nn_m"]
        if abs(m0 - m1) < 0.05:
            warnings.append("auto placement ambiguous: multiple locations score similarly; add a start hint or a correspondence pair")
    return out, warnings


def _infer_image_size(K: np.ndarray) -> tuple[float, float]:
    cx = float(K[0, 2]) if K.shape == (3, 3) else 0.0
    cy = float(K[1, 2]) if K.shape == (3, 3) else 0.0
    return max(1.0, cx * 2.0), max(1.0, cy * 2.0)


def _visible_frames_for_samples(
    samples: np.ndarray,
    camera_states: list[tuple[np.ndarray, np.ndarray]],
    K: np.ndarray | None,
    *,
    max_depth_m: float = 15.0,
    margin_px: float = 32.0,
) -> list[int]:
    if samples.size == 0 or not camera_states or K is None or K.shape != (3, 3):
        return []
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    if fx <= 0 or fy <= 0:
        return []
    width, height = _infer_image_size(K)
    frames: list[int] = []
    for frame_idx, (R, t) in enumerate(camera_states):
        # R is camera-local to model-world. For row vectors, local = world_delta @ R.
        local = (samples - t[None, :]) @ R
        depth = -local[:, 2]
        in_depth = (depth > 0.05) & (depth <= max_depth_m)
        if not np.any(in_depth):
            continue
        u = fx * (local[:, 0] / np.maximum(depth, 1e-9)) + cx
        v = fy * (local[:, 1] / np.maximum(depth, 1e-9)) + cy
        in_frame = (
            in_depth
            & (u >= -margin_px)
            & (u <= width + margin_px)
            & (v >= -margin_px)
            & (v <= height + margin_px)
        )
        if np.any(in_frame):
            frames.append(frame_idx)
    return frames


def _nearest_camera_frames_for_points(
    points: np.ndarray,
    camera_states: list[tuple[np.ndarray, np.ndarray]],
    *,
    max_frames: int = 12,
) -> list[int]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0 or not camera_states:
        return []
    pts = np.atleast_2d(pts)
    centers = np.asarray([t for _R, t in camera_states], dtype=np.float64)
    if centers.size == 0:
        return []
    dists = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
    nearest = dists.argmin(axis=1)
    unique, counts = np.unique(nearest, return_counts=True)
    order = np.argsort(-counts)
    selected = unique[order][:max(1, int(max_frames))]
    return sorted(int(x) for x in selected.tolist())


def _object_samples(obj: dict, n_linear: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Return sample points and [0..1] linear parameters for segment coverage."""
    if obj.get("start") is not None and obj.get("end") is not None:
        start = np.asarray(obj["start"], dtype=np.float64)
        end = np.asarray(obj["end"], dtype=np.float64)
        ts = np.linspace(0.0, 1.0, n_linear)
        pts = (1 - ts[:, None]) * start[None, :] + ts[:, None] * end[None, :]
        return pts, ts
    pts = []
    if obj.get("center") is not None:
        pts.append(obj["center"])
    if obj.get("bbox") is not None:
        mn = np.asarray(obj["bbox"][0], dtype=np.float64)
        mx = np.asarray(obj["bbox"][1], dtype=np.float64)
        pts.extend([
            mn, mx,
            [mn[0], mn[1], mx[2]], [mn[0], mx[1], mn[2]], [mx[0], mn[1], mn[2]],
            [mn[0], mx[1], mx[2]], [mx[0], mn[1], mx[2]], [mx[0], mx[1], mn[2]],
            ((mn + mx) * 0.5).tolist(),
        ])
    if not pts:
        pts.append([0, 0, 0])
    arr = np.asarray(pts, dtype=np.float64)
    return arr, np.zeros(arr.shape[0], dtype=np.float64)


def _linear_hit_metrics(ts: np.ndarray, hit: np.ndarray, *, bins: int = 8) -> dict:
    params = np.asarray(ts, dtype=np.float64)
    hits = np.asarray(hit, dtype=bool)
    if params.size == 0 or hits.size == 0:
        return {
            "sample_count": 0,
            "hit_sample_count": 0,
            "coverage_ratio": 0.0,
            "segment_coverage_ratio": 0.0,
            "geometry_consistency": 0.0,
            "segment_hit_bins": 0,
            "segment_total_bins": int(bins),
            "linear_hit_span_ratio": 0.0,
            "linear_longest_gap_ratio": 1.0,
        }
    bins = max(1, int(bins))
    sample_count = int(hits.size)
    hit_count = int(hits.sum())
    coverage = float(hit_count / max(sample_count, 1))
    if hit_count == 0:
        return {
            "sample_count": sample_count,
            "hit_sample_count": 0,
            "coverage_ratio": coverage,
            "segment_coverage_ratio": 0.0,
            "geometry_consistency": 0.0,
            "segment_hit_bins": 0,
            "segment_total_bins": bins,
            "linear_hit_span_ratio": 0.0,
            "linear_longest_gap_ratio": 1.0,
        }
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.clip(np.digitize(params, edges[1:-1], right=False), 0, bins - 1)
    hit_bins = np.unique(bin_ids[hits])
    segment_coverage = float(hit_bins.size / bins)
    hit_params = np.sort(np.clip(params[hits], 0.0, 1.0))
    hit_span = float(hit_params[-1] - hit_params[0]) if hit_params.size >= 2 else 0.0
    gaps = np.diff(np.concatenate(([0.0], hit_params, [1.0])))
    longest_gap = float(gaps.max()) if gaps.size else 1.0
    # A long linear object should have evidence distributed along its length;
    # a dense cluster on one end should not look geometrically complete.
    geometry_consistency = float(min(coverage, segment_coverage))
    return {
        "sample_count": sample_count,
        "hit_sample_count": hit_count,
        "coverage_ratio": coverage,
        "segment_coverage_ratio": segment_coverage,
        "geometry_consistency": geometry_consistency,
        "segment_hit_bins": int(hit_bins.size),
        "segment_total_bins": bins,
        "linear_hit_span_ratio": hit_span,
        "linear_longest_gap_ratio": longest_gap,
    }


def _correspondence_spread(points: np.ndarray) -> dict:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return {"extent_m": 0.0, "rms_radius_m": 0.0, "rank": 0, "singular_values": []}
    centered = pts - pts.mean(axis=0)
    extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    rms = float(np.sqrt((centered ** 2).sum(axis=1).mean())) if pts.shape[0] else 0.0
    if pts.shape[0] >= 2:
        singular = np.linalg.svd(centered, compute_uv=False)
        rel = singular / max(float(singular[0]), 1e-9)
        rank = int((rel > 0.08).sum())
    else:
        singular = np.zeros(0, dtype=np.float64)
        rank = 0
    return {
        "extent_m": extent,
        "rms_radius_m": rms,
        "rank": rank,
        "singular_values": singular.astype(float).tolist(),
    }


def _solve_scan_to_model_alignment(pairs: list[dict]) -> dict:
    if len(pairs) < 4:
        return {
            "quality": "red",
            "stable": False,
            "rmse_m": None,
            "scale": 1.0,
            "rotation": np.eye(3).tolist(),
            "translation": [0, 0, 0],
            "n": len(pairs),
            "note": "need >=4 correspondences; using identity for preview",
        }
    from lingbot_map.bim.alignment import solve_sim3_umeyama
    src = np.asarray([p["scan"] for p in pairs], dtype=np.float64)
    dst = np.asarray([p["model"] for p in pairs], dtype=np.float64)
    src_spread = _correspondence_spread(src)
    dst_spread = _correspondence_spread(dst)
    spread_ok = (
        src_spread["extent_m"] >= 0.15
        and dst_spread["extent_m"] >= 0.50
        and src_spread["rank"] >= 2
        and dst_spread["rank"] >= 2
    )
    result = solve_sim3_umeyama(src, dst, rmse_green=0.10, rmse_yellow=0.25)
    loo_errors = []
    if len(pairs) >= 5:
        for i in range(len(pairs)):
            keep = [j for j in range(len(pairs)) if j != i]
            r = solve_sim3_umeyama(src[keep], dst[keep], rmse_green=0.10, rmse_yellow=0.25)
            pred = r.transform.apply(src[i])
            loo_errors.append(float(np.linalg.norm(pred - dst[i])))
    max_loo = max(loo_errors) if loo_errors else float(result.rmse)
    quality = "red" if result.quality == "review" else result.quality
    stable = bool(max_loo <= max(0.25, float(result.rmse) * 2.5))
    if not stable and quality == "green":
        quality = "yellow"
    if not stable and quality == "yellow":
        quality = "red"
    spread_note = None
    if not spread_ok:
        quality = "red"
        stable = False
        spread_note = "correspondence points are too clustered or near-collinear; distribute points across the scan/model area"
    return {
        "quality": quality,
        "stable": stable,
        "rmse_m": float(result.rmse),
        "max_leave_one_out_rmse_m": max_loo,
        "correspondence_spread": {
            "ok": spread_ok,
            "scan": src_spread,
            "model": dst_spread,
            "note": spread_note,
        },
        "scale": float(result.transform.scale),
        "rotation": result.transform.rotation.tolist(),
        "translation": result.transform.translation.tolist(),
        "n": int(result.n_correspondences),
        "per_point_residuals": result.per_point_residuals.tolist(),
    }


def _normalize_alignment_pairs(raw_pairs) -> list[dict]:
    if raw_pairs is None:
        return []
    if not isinstance(raw_pairs, list):
        raise ValueError("alignment.pairs must be a list")
    pairs = []
    for i, pair in enumerate(raw_pairs):
        if not isinstance(pair, dict):
            raise ValueError(f"alignment pair #{i + 1} must be an object")
        scan = pair.get("scan")
        model = pair.get("model")
        if not isinstance(scan, list) or not isinstance(model, list) or len(scan) != 3 or len(model) != 3:
            raise ValueError(f"alignment pair #{i + 1} needs scan/model 3D coordinates")
        try:
            scan_v = [float(x) for x in scan]
            model_v = [float(x) for x in model]
        except Exception as e:
            raise ValueError(f"alignment pair #{i + 1} contains non-numeric coordinates") from e
        if not np.isfinite(scan_v).all() or not np.isfinite(model_v).all():
            raise ValueError(f"alignment pair #{i + 1} contains non-finite coordinates")
        pairs.append({"scan": scan_v, "model": model_v})
    return pairs


@app.get("/api/uploads/{upload_id}/coverage")
async def get_coverage(upload_id: str, model_id: str | None = None):
    if _bad_id(upload_id, "upload_"):
        return JSONResponse({"ok": False, "error": "bad id"}, status_code=400)
    doc = _load_scoped_upload_json(upload_id, "coverage", model_id)
    if doc is None:
        return JSONResponse({"ok": False, "status": "needs_analysis", "error": "coverage not found"}, status_code=404)
    alignment = doc.get("alignment") if isinstance(doc, dict) else None
    if isinstance(alignment, dict) and alignment.get("quality") == "red":
        return JSONResponse(
            {
                "ok": False,
                "status": "needs_alignment",
                "error": "coverage was generated with red alignment and must be regenerated after valid correspondence alignment",
                "alignment": alignment,
            },
            status_code=409,
        )
    return JSONResponse(doc, headers={"Cache-Control": "no-store"})


@app.get("/api/uploads/{upload_id}/alignment")
async def get_alignment(upload_id: str, model_id: str | None = None):
    if _bad_id(upload_id, "upload_"):
        return JSONResponse({"ok": False, "error": "bad id"}, status_code=400)
    doc = _load_scoped_upload_json(upload_id, "alignment", model_id)
    if doc is None:
        return JSONResponse({"ok": False, "status": "needs_alignment", "error": "alignment not found"}, status_code=404)
    return JSONResponse(doc, headers={"Cache-Control": "no-store"})


@app.get("/api/uploads/{upload_id}/alignment/candidates")
async def get_alignment_candidates(upload_id: str, model_id: str | None = None):
    if _bad_id(upload_id, "upload_"):
        return JSONResponse({"ok": False, "error": "bad id"}, status_code=400)
    doc = _load_scoped_upload_json(upload_id, "alignment_candidates", model_id or _default_model_id())
    if doc is None:
        return JSONResponse(
            {"ok": False, "status": "missing", "error": "alignment candidates not found"},
            status_code=404,
        )
    return JSONResponse(doc, headers={"Cache-Control": "no-store"})


@app.post("/api/uploads/{upload_id}/alignment/candidates")
async def alignment_candidates(upload_id: str, payload: dict):
    if _bad_id(upload_id, "upload_"):
        return JSONResponse({"ok": False, "error": "bad id"}, status_code=400)
    model_id = payload.get("model_id") or _default_model_id()
    if model_id not in _model_registry():
        return JSONResponse({"ok": False, "error": "unknown model_id"}, status_code=404)
    dry_run = bool(payload.get("dry_run"))
    try:
        input_pairs = _normalize_alignment_pairs((payload.get("alignment") or {}).get("pairs", []))
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    try:
        manifest = _model_manifest(model_id)
        if not manifest.get("manifest_valid"):
            return JSONResponse(
                {"ok": False, "status": "model_manifest_invalid", "error": "model manifest invalid", "model": manifest},
                status_code=409,
            )
        _manifest, model_objects = _load_model_objects(model_id)
        model_bounds = _model_object_bounds(model_objects)
        repetition = _model_repetition_risk(model_objects)
        scan_path = _select_scan_payload(upload_id, payload.get("variant") or "detail")
        if scan_path is None:
            return JSONResponse({"ok": False, "status": "needs_scan", "error": "scan payload not found"}, status_code=404)
        scan = _parse_scan_payload(scan_path, include_points=False)
        scan_bounds = _scan_viewer_bounds(scan)
        descriptor = _pose_descriptor(scan.get("poses") or [])
        candidates: list[dict] = []
        warnings: list[str] = []

        if len(input_pairs) >= 4:
            alignment = _solve_scan_to_model_alignment(input_pairs)
            alignment["pairs"] = input_pairs
            candidates.append(_candidate_from_alignment(
                candidate_id="manual:pairs",
                source="manual_pairs_preview",
                alignment=alignment,
                scan_bounds=scan_bounds,
                model_bounds=model_bounds,
                reason="current correspondence pairs solve scan-to-model Sim(3); save alignment before coverage analysis",
                can_save=alignment.get("quality") in ("green", "yellow"),
            ))
            status = "auto_alignment_candidate" if alignment.get("quality") in ("green", "yellow") else "needs_alignment"
            next_action = "save_alignment" if alignment.get("quality") in ("green", "yellow") else "add_better_alignment_pairs"
        else:
            if len(input_pairs) == 0:
                warnings.append("no anchor pairs provided; video-only global localization is intentionally not auto-accepted")
            else:
                warnings.append("1-3 anchor pairs are useful priors but not enough to solve a stable Sim(3) alignment")
            if payload.get("auto_place"):
                auto_cands, auto_warnings = _auto_place_candidates(
                    upload_id,
                    model_objects,
                    scale=float(payload.get("scale") or 1.0),
                    start_hint=payload.get("start_hint"),
                    direction_az=(float(payload["direction_az"])
                                  if payload.get("direction_az") is not None else None),
                    max_candidates=int(payload.get("max_candidates") or 3),
                )
                candidates.extend(auto_cands)
                warnings.extend(auto_warnings)
            candidates.extend(_previous_alignment_candidates(
                model_id,
                upload_id,
                scan_bounds,
                model_bounds,
                limit=int(payload.get("max_candidates") or 5),
            ))
            # Option B (FR-A4/A3): corroborate prior candidates against scan geometry,
            # ICP-refine when it improves, and promote can_save under strict gates.
            if candidates:
                scan_pts = None
                try:
                    scan_full = _parse_scan_payload(scan_path, include_points=True, max_points=150_000)
                    raw_pts = np.asarray(scan_full.get("points") or [], dtype=np.float64)
                    if raw_pts.size:
                        scan_pts = _viewer_points(raw_pts)
                except Exception as e:
                    log.warning("candidate corroboration scan load failed: %s", e)
                model_tree = None
                model_samples = np.empty((0, 3))
                model_radii = None
                if scan_pts is not None and scan_pts.size:
                    model_samples, model_radii = _model_surface_samples_with_radius(model_objects)
                    try:
                        from scipy.spatial import cKDTree
                        model_tree = cKDTree(model_samples) if model_samples.size else None
                    except Exception:
                        model_tree = None
                anchor_src = np.asarray([p["scan"] for p in input_pairs], dtype=np.float64) if input_pairs else None
                anchor_dst = np.asarray([p["model"] for p in input_pairs], dtype=np.float64) if input_pairs else None
                for cand in candidates:
                    if cand.get("source") == "auto_geometric":
                        continue  # 자동 배치 후보는 _auto_place_candidates에서 이미 검증됨
                    cal = cand.get("alignment") or {}
                    if model_tree is None:
                        continue
                    score = _corroborate_alignment(scan_pts, model_tree, cal, sample_radii=model_radii)
                    if score.get("ok"):
                        refined = _icp_refine_alignment(scan_pts, model_samples, model_tree, cal)
                        if refined is not None:
                            r_score = _corroborate_alignment(scan_pts, model_tree, refined, sample_radii=model_radii)
                            if r_score.get("ok") and r_score["median_nn_m"] < score["median_nn_m"]:
                                cal = {**cal, "rotation": refined["rotation"], "translation": refined["translation"], "note": refined["note"]}
                                cand["alignment"] = cal
                                cand["icp_refined"] = True
                                score = r_score
                    cand["corroboration"] = score
                    corroborated = (
                        bool(score.get("ok"))
                        and score.get("median_nn_m", 1e9) <= 0.20
                        and score.get("inlier_ratio", 0.0) >= 0.50
                    )
                    cand["corroborated"] = corroborated
                    anchor_ok = False
                    if anchor_src is not None and len(anchor_src):
                        pred = _apply_sim3(anchor_src, cal)
                        res = np.linalg.norm(pred - anchor_dst, axis=1)
                        cand["anchor_residuals_m"] = [round(float(x), 4) for x in res]
                        anchor_ok = bool(np.all(res <= 0.30))
                    # repeated-geometry models (high risk) still require at least one anchor
                    if corroborated and (anchor_ok or repetition["level"] != "high"):
                        cand["can_save"] = True
                        cand["reason"] = "prior alignment corroborated by scan geometry" + (" + anchors" if anchor_ok else "")
                saveable = [c for c in candidates if c.get("can_save")]
                if saveable:
                    status = "auto_alignment_candidate"
                    next_action = "save_candidate_alignment"
                    warnings.append("corroborated candidate available; explicit save still required")
                else:
                    top_score = candidates[0]["score"]
                    close = [c for c in candidates if top_score - c["score"] <= 0.15]
                    status = "ambiguous_alignment" if len(close) > 1 or repetition["level"] == "high" else "auto_alignment_candidate"
                    next_action = "add_anchor_pair_or_confirm_candidate"
                    warnings.append("candidate alignments are preview-only until confirmed by user anchors")
            else:
                status = "needs_anchor"
                next_action = "add_anchor_pair"
        if repetition["level"] in ("high", "medium"):
            warnings.append("model contains repeated object groups; ambiguous candidates must not be auto-applied")

        doc = {
            "ok": True,
            "status": status,
            "next_action": next_action,
            "upload_id": upload_id,
            "model_id": model_id,
            "model": {
                "model_id": manifest.get("model_id"),
                "name": manifest.get("name"),
                "manifest_valid": manifest.get("manifest_valid"),
                "guid_mapping_ratio": manifest.get("guid_mapping_ratio"),
            },
            "scan": {
                "payload": scan_path.name,
                "magic": scan.get("magic"),
                "point_count": scan.get("count"),
                "pose_count": len(scan.get("poses") or []),
                "bounds": scan_bounds,
                "trajectory": descriptor,
            },
            "repetition_risk": repetition,
            "anchor_count": len(input_pairs),
            "candidates": sorted(candidates, key=lambda c: c.get("score", 0.0), reverse=True),
            "warnings": warnings,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "dry_run": dry_run,
        }
        if not dry_run:
            (UPLOAD_DIR / f"{upload_id}.{model_id}.alignment_candidates.json").write_text(
                json.dumps(doc, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return doc
    except Exception as e:
        log.exception("alignment candidate generation failed: %s", e)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/api/uploads/{upload_id}/coverage/status")
async def coverage_status(upload_id: str, model_id: str | None = None):
    if _bad_id(upload_id, "upload_"):
        return JSONResponse({"ok": False, "error": "bad id"}, status_code=400)
    model_id = model_id or _default_model_id()
    if model_id not in _model_registry():
        return JSONResponse({"ok": False, "error": "unknown model_id"}, status_code=404)
    warnings: list[str] = []
    try:
        model = _model_manifest(model_id)
    except Exception as e:
        return JSONResponse(
            {"ok": False, "status": "model_manifest_invalid", "error": str(e), "model_id": model_id},
            status_code=409,
        )
    if model.get("coverage_geometry_source") == "pag_proxy":
        warnings.append("GLB GUID mapping is low; coverage coloring uses PAG proxy geometry and GLB is visual context only")
    try:
        _manifest, model_objects = _load_model_objects(model_id)
        model_bounds = _model_object_bounds(model_objects)
    except Exception as e:
        model_bounds = None
        warnings.append(f"model bounds unavailable: {e}")

    payloads = _upload_payloads(upload_id)
    scan_path = _select_scan_payload(upload_id)
    scan_status: dict = {"ok": False, "payload": None}
    scan_bounds = None
    if scan_path is None:
        warnings.append("scan payload not found")
    else:
        try:
            scan = _parse_scan_payload(scan_path, include_points=False)
            scan_bounds = _scan_viewer_bounds(scan)
            thumbs = _load_upload_thumbs(upload_id, prefer_audit=True) or []
            scan_status = {
                "ok": True,
                "payload": scan_path.name,
                "magic": scan.get("magic"),
                "point_count": scan.get("count"),
                "pose_count": len(scan.get("poses") or []),
                "thumb_count": len(thumbs),
                "mesh_glb": (UPLOAD_DIR / f"{upload_id}.mesh.glb").exists(),
                "has_source_ids": scan.get("magic") == "LBP4",
            }
            if not scan_status["mesh_glb"]:
                warnings.append("scan GLB not found; point cloud can still be used for correspondence picking")
            if not scan_status["has_source_ids"]:
                warnings.append("scan source frame ids not available; coverage will use nearest camera pose fallback")
            if not scan_status["thumb_count"]:
                warnings.append("thumbnail sidecar not found")
        except Exception as e:
            warnings.append(f"scan parse failed: {e}")
            scan_status = {"ok": False, "payload": scan_path.name, "error": str(e)}

    alignment_doc = _load_scoped_upload_json(upload_id, "alignment", model_id)
    alignment = alignment_doc.get("alignment") if isinstance(alignment_doc, dict) else None
    if not alignment:
        alignment_status = {"state": "missing", "quality": None, "ready": False}
    else:
        alignment_status = {
            "state": "valid" if alignment.get("quality") in ("green", "yellow") else "invalid",
            "quality": alignment.get("quality"),
            "ready": alignment.get("quality") in ("green", "yellow"),
            "n": alignment.get("n"),
            "stable": alignment.get("stable"),
            "rmse_m": alignment.get("rmse_m"),
            "max_leave_one_out_rmse_m": alignment.get("max_leave_one_out_rmse_m"),
            "spread_ok": (alignment.get("correspondence_spread") or {}).get("ok"),
        }
        if not alignment_status["ready"]:
            warnings.append("valid scan-to-model alignment is required before coverage analysis")

    aligned_scan_bounds = None
    if alignment and alignment_status["ready"] and scan_bounds:
        corners = _bounds_corners(scan_bounds)
        if corners.size:
            aligned_scan_bounds = _bounds_summary_from_points(_apply_sim3(corners, alignment))

    coordinate_state = "aligned_model_world" if alignment_status["ready"] else "raw_scan_local"
    coordinate_status = {
        "state": coordinate_state,
        "scan_space": "model_world" if alignment_status["ready"] else "lingbot_local_reconstruction",
        "model_space": "bim_model_world",
        "model_up_axis": "Z_UP",
        "scene_up_axis": "Z_UP",
        "scene_ground_plane": "XY",
        "viewer_scan_axis": "x,-y,-z",
        "raw_scan_bounds": scan_bounds,
        "aligned_scan_bounds": aligned_scan_bounds,
        "model_bounds": model_bounds,
        "alignment_required": not alignment_status["ready"],
        "diagnosis": (
            "scan has been transformed into model coordinates"
            if alignment_status["ready"]
            else "raw scan and BIM model use different origins/axes until correspondence alignment is saved"
        ),
    }
    if scan_status.get("ok") and not alignment_status["ready"]:
        warnings.append("raw scan coordinates are local reconstruction; model/path overlay requires valid alignment")

    coverage_doc = _load_scoped_upload_json(upload_id, "coverage", model_id)
    coverage_alignment = coverage_doc.get("alignment") if isinstance(coverage_doc, dict) else None
    if not coverage_doc:
        coverage_state = "missing"
        coverage_ready = False
    elif isinstance(coverage_alignment, dict) and coverage_alignment.get("quality") == "red":
        coverage_state = "stale_red_alignment"
        coverage_ready = False
        warnings.append("saved coverage was generated with red alignment and must be regenerated")
    else:
        coverage_state = "ready"
        coverage_ready = True

    candidates_doc = _load_scoped_upload_json(upload_id, "alignment_candidates", model_id)
    if isinstance(candidates_doc, dict):
        auto_alignment_status = {
            "state": candidates_doc.get("status"),
            "candidate_count": len(candidates_doc.get("candidates") or []),
            "anchor_count": candidates_doc.get("anchor_count"),
            "repetition_risk": (candidates_doc.get("repetition_risk") or {}).get("level"),
            "generated_at": candidates_doc.get("generated_at"),
        }
    else:
        auto_alignment_status = {
            "state": "missing",
            "candidate_count": 0,
            "anchor_count": 0,
            "repetition_risk": None,
            "generated_at": None,
        }
    if auto_alignment_status.get("repetition_risk") == "high":
        warnings.append("model has many repeated/similar objects; automatic video-to-model mapping requires user anchors")

    if not model.get("manifest_valid"):
        next_action = "fix_model_manifest"
        overall = "model_manifest_invalid"
    elif not scan_status.get("ok"):
        next_action = "upload_or_process_video"
        overall = "needs_scan"
    elif not alignment_status["ready"]:
        next_action = "add_alignment_pairs"
        overall = "needs_alignment"
    elif not coverage_ready:
        next_action = "run_coverage_analysis"
        overall = "needs_analysis"
    else:
        next_action = "review_coverage"
        overall = "ready"

    keyframe_dir = UPLOAD_DIR / f"{upload_id}.keyframes"
    keyframe_count = len(list(keyframe_dir.glob("frame_*.jpg"))) if keyframe_dir.exists() else 0
    checks = [
        {
            "id": "model_manifest",
            "label": "모델 manifest",
            "state": "pass" if model.get("manifest_valid") else "block",
            "detail": "valid" if model.get("manifest_valid") else "invalid",
        },
        {
            "id": "model_guid_mapping",
            "label": "GUID/기준 형상",
            "state": "pass" if model.get("coverage_geometry_source") in ("glb_guid_mesh", "hybrid_glb_proxy") else "warn",
            "detail": (
                "GLB GUID mesh"
                if model.get("coverage_geometry_source") == "glb_guid_mesh"
                else f"하이브리드: {model.get('guid_mapped_count', 0)}/{model.get('guid_total', 0)} GLB mesh + 나머지 proxy"
                if model.get("coverage_geometry_source") == "hybrid_glb_proxy"
                else "PAG proxy 기준; GLB는 시각 참고"
            ),
        },
        {
            "id": "scan_payload",
            "label": "scan payload",
            "state": "pass" if scan_status.get("ok") else "block",
            "detail": (
                f"{scan_status.get('magic')} · {scan_status.get('point_count', 0):,} pts"
                if scan_status.get("ok") else "scan 없음"
            ),
        },
        {
            "id": "pose_thumbnails",
            "label": "pose/thumb",
            "state": "pass" if scan_status.get("pose_count") and scan_status.get("thumb_count") else "warn",
            "detail": f"{scan_status.get('pose_count', 0)} / {scan_status.get('thumb_count', 0)}",
        },
        {
            "id": "scan_overlay",
            "label": "scan overlay",
            "state": "pass" if scan_status.get("mesh_glb") else "warn",
            "detail": "scan GLB 있음" if scan_status.get("mesh_glb") else "scan GLB 없음; RGB 점군으로 대응점 선택",
        },
        {
            "id": "source_frames",
            "label": "source frame",
            "state": "pass" if scan_status.get("has_source_ids") else "warn",
            "detail": "source_ids" if scan_status.get("has_source_ids") else "nearest pose fallback",
        },
        {
            "id": "alignment",
            "label": "scan-to-model 정합",
            "state": "pass" if alignment_status["ready"] else "block",
            "detail": (
                f"{alignment_status.get('quality')} · rmse {alignment_status.get('rmse_m')}"
                if alignment_status["ready"]
                else "green/yellow 정합 필요"
            ),
        },
        {
            "id": "auto_mapping_risk",
            "label": "자동 매핑 위험",
            "state": "warn" if auto_alignment_status.get("repetition_risk") == "high" else "pass",
            "detail": auto_alignment_status.get("repetition_risk") or "low/unknown",
        },
        {
            "id": "coverage_result",
            "label": "coverage 결과",
            "state": (
                "pass" if coverage_ready else
                "block" if coverage_state == "stale_red_alignment" else
                "pending"
            ),
            "detail": coverage_state,
        },
        {
            "id": "keyframes",
            "label": "evidence keyframe",
            "state": "pass" if keyframe_count else "pending",
            "detail": f"{keyframe_count} files" if keyframe_count else "필요 시 on-demand 생성",
        },
    ]

    return {
        "ok": True,
        "status": overall,
        "next_action": next_action,
        "model": {
            "model_id": model.get("model_id"),
            "name": model.get("name"),
            "manifest_valid": model.get("manifest_valid"),
            "guid_mapping_ratio": model.get("guid_mapping_ratio"),
            "guid_mapping_status": model.get("guid_mapping_status"),
            "fallback_geometry_count": model.get("fallback_geometry_count"),
            "coverage_geometry_source": model.get("coverage_geometry_source"),
            "glb_role": model.get("glb_role"),
        },
        "upload": {
            "id": upload_id,
            "video": (_upload_video_path(upload_id) or Path("")).name or None,
            "payloads": payloads,
        },
        "scan": scan_status,
        "alignment": alignment_status,
        "coordinate": coordinate_status,
        "auto_alignment": auto_alignment_status,
        "coverage": {
            "state": coverage_state,
            "ready": coverage_ready,
            "status_counts": coverage_doc.get("status_counts") if isinstance(coverage_doc, dict) else None,
            "generated_at": coverage_doc.get("generated_at") if isinstance(coverage_doc, dict) else None,
        },
        "checks": checks,
        "warnings": warnings,
    }


@app.get("/api/uploads/{upload_id}/coverage/review")
async def get_coverage_review(upload_id: str, model_id: str | None = None):
    if _bad_id(upload_id, "upload_"):
        return JSONResponse({"ok": False, "error": "bad id"}, status_code=400)
    model_id = model_id or _default_model_id()
    if model_id not in _model_registry():
        return JSONResponse({"ok": False, "error": "unknown model_id"}, status_code=404)
    return _load_coverage_review(upload_id, model_id)


@app.post("/api/uploads/{upload_id}/coverage/review")
async def save_coverage_review(upload_id: str, payload: dict):
    if _bad_id(upload_id, "upload_"):
        return JSONResponse({"ok": False, "error": "bad id"}, status_code=400)
    model_id = payload.get("model_id") or _default_model_id()
    if model_id not in _model_registry():
        return JSONResponse({"ok": False, "error": "unknown model_id"}, status_code=404)
    replace = bool(payload.get("replace"))
    review = _empty_coverage_review(upload_id, model_id) if replace else _load_coverage_review(upload_id, model_id)

    object_reviews = payload.get("object_reviews")
    if isinstance(object_reviews, dict):
        for guid, value in object_reviews.items():
            key = str(guid)
            if value is None:
                review["object_reviews"].pop(key, None)
                continue
            entry = _sanitize_review_entry(value)
            if entry:
                review["object_reviews"][key] = entry

    frame_reviews = payload.get("frame_reviews")
    if isinstance(frame_reviews, dict):
        for frame, value in frame_reviews.items():
            try:
                key = str(int(frame))
            except (TypeError, ValueError):
                continue
            if value is None:
                review["frame_reviews"].pop(key, None)
                continue
            entry = _sanitize_review_entry(value)
            if entry:
                review["frame_reviews"][key] = entry

    if "notes" in payload:
        review["notes"] = str(payload.get("notes") or "")[:2000]
    review["ok"] = True
    review["upload_id"] = upload_id
    review["model_id"] = model_id
    review["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _coverage_review_path(upload_id, model_id).write_text(
        json.dumps(review, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return review


@app.get("/api/uploads/{upload_id}/coverage/report")
async def coverage_quality_report(upload_id: str, model_id: str | None = None):
    if _bad_id(upload_id, "upload_"):
        return JSONResponse({"ok": False, "error": "bad id"}, status_code=400)
    model_id = model_id or _default_model_id()
    if model_id not in _model_registry():
        return JSONResponse({"ok": False, "error": "unknown model_id"}, status_code=404)

    blockers: list[dict] = []
    warnings: list[dict] = []
    try:
        model = _model_manifest(model_id)
        if not model.get("manifest_valid"):
            blockers.append({"id": "model_manifest", "message": "model manifest is invalid"})
    except Exception as e:
        return JSONResponse(
            {"ok": False, "status": "model_manifest_invalid", "error": str(e), "model_id": model_id},
            status_code=409,
        )

    coverage_doc = _load_scoped_upload_json(upload_id, "coverage", model_id)
    alignment_doc = _load_scoped_upload_json(upload_id, "alignment", model_id)
    coverage_alignment = coverage_doc.get("alignment") if isinstance(coverage_doc, dict) else None
    saved_alignment = alignment_doc.get("alignment") if isinstance(alignment_doc, dict) else None
    alignment = coverage_alignment if isinstance(coverage_alignment, dict) else saved_alignment

    if not isinstance(coverage_doc, dict):
        blockers.append({"id": "coverage_missing", "message": "coverage analysis result is missing"})
        objects: list[dict] = []
        status_counts: dict = {}
    elif isinstance(coverage_alignment, dict) and coverage_alignment.get("quality") == "red":
        blockers.append({"id": "stale_red_alignment", "message": "coverage was generated with red alignment"})
        objects = coverage_doc.get("objects") or []
        status_counts = coverage_doc.get("status_counts") or {}
    else:
        objects = coverage_doc.get("objects") or []
        status_counts = coverage_doc.get("status_counts") or {}

    alignment_ready = isinstance(alignment, dict) and alignment.get("quality") in ("green", "yellow")
    if not alignment_ready:
        blockers.append({"id": "alignment", "message": "green/yellow scan-to-model alignment is required"})

    scan_path = _select_scan_payload(upload_id)
    pose_count = 0
    thumb_count = 0
    if scan_path is None:
        blockers.append({"id": "scan_payload", "message": "scan payload is missing"})
    else:
        try:
            scan = _parse_scan_payload(scan_path, include_points=False)
            pose_count = len(scan.get("poses") or [])
            thumb_count = len(_load_upload_thumbs(upload_id, prefer_audit=True) or [])
            if not pose_count or not thumb_count:
                warnings.append({"id": "pose_thumb", "message": "pose/thumb data is incomplete"})
        except Exception as e:
            blockers.append({"id": "scan_payload", "message": f"scan parse failed: {e}"})

    observed = [o for o in objects if o.get("status") == "observed"]
    likely = [o for o in objects if o.get("status") == "likely_observed"]
    uncertain = [o for o in objects if o.get("status") == "uncertain"]
    observed_count = len(observed)
    observed_with_evidence = sum(1 for o in observed if o.get("evidence_thumb_ids") or o.get("evidence_keyframes"))
    observed_support_ge3 = sum(1 for o in observed if int(o.get("support_frame_count") or len(o.get("support_frames") or [])) >= 3)
    observed_evidence_ratio = observed_with_evidence / observed_count if observed_count else 0.0
    observed_support_ratio = observed_support_ge3 / observed_count if observed_count else 0.0
    review_doc = _load_coverage_review(upload_id, model_id)
    object_reviews = review_doc.get("object_reviews") if isinstance(review_doc.get("object_reviews"), dict) else {}
    frame_reviews = review_doc.get("frame_reviews") if isinstance(review_doc.get("frame_reviews"), dict) else {}
    observed_reviewed = 0
    observed_review_accepted = 0
    observed_review_rejected = 0
    for obj in observed:
        rec = object_reviews.get(str(obj.get("guid")))
        if not isinstance(rec, dict) or rec.get("accepted") is None:
            continue
        observed_reviewed += 1
        if rec.get("accepted") is True:
            observed_review_accepted += 1
        else:
            observed_review_rejected += 1
    observed_review_ratio = observed_review_accepted / observed_count if observed_count else 0.0
    frame_reviewed = 0
    frame_review_accepted = 0
    frame_review_rejected = 0
    for rec in frame_reviews.values():
        if not isinstance(rec, dict) or rec.get("accepted") is None:
            continue
        frame_reviewed += 1
        if rec.get("accepted") is True:
            frame_review_accepted += 1
        else:
            frame_review_rejected += 1
    frame_review_ratio = frame_review_accepted / frame_reviewed if frame_reviewed else 0.0

    if isinstance(coverage_doc, dict) and not observed_count:
        warnings.append({"id": "observed_empty", "message": "no observed object exists yet"})

    automated_gate_passed = (
        not blockers
        and observed_count > 0
        and observed_evidence_ratio >= 0.90
        and observed_support_ratio >= 0.90
    )
    manual_gate_passed = (
        automated_gate_passed
        and observed_review_ratio >= 0.90
        and frame_reviewed > 0
        and frame_review_ratio >= 0.90
    )
    if blockers:
        report_status = "blocked"
    elif manual_gate_passed:
        report_status = "ready_for_manual_review"
    elif automated_gate_passed:
        report_status = "ready_for_manual_review"
    else:
        report_status = "needs_evidence_review"

    criteria = [
        {
            "id": "keyframe_location_90",
            "label": "keyframe 위치 90%",
            "state": "pass" if alignment_ready and pose_count else "block",
            "metric": {"pose_count": pose_count, "thumb_count": thumb_count},
            "detail": "정합된 경로/프레임을 모델 좌표에서 검수 가능" if alignment_ready else "정합 전에는 위치 90% 검수 불가",
        },
        {
            "id": "observed_evidence_90",
            "label": "observed evidence 90%",
            "state": (
                "pass" if observed_count and observed_evidence_ratio >= 0.90 else
                "pending" if not observed_count else
                "warn"
            ),
            "metric": {
                "observed_count": observed_count,
                "observed_with_evidence": observed_with_evidence,
                "ratio": round(observed_evidence_ratio, 4),
            },
            "detail": "observed 객체 대부분이 evidence frame에 연결됨",
        },
        {
            "id": "support_frames",
            "label": "support frame",
            "state": (
                "pass" if observed_count and observed_support_ratio >= 0.90 else
                "pending" if not observed_count else
                "warn"
            ),
            "metric": {
                "observed_count": observed_count,
                "support_ge3": observed_support_ge3,
                "ratio": round(observed_support_ratio, 4),
            },
            "detail": "observed 객체는 최소 3개 support frame 기준을 만족해야 함",
        },
        {
            "id": "overclaim_guard",
            "label": "과대판정 방지",
            "state": "pass" if alignment_ready else "block",
            "metric": {
                "likely_observed": len(likely),
                "uncertain": len(uncertain),
                "status_counts": status_counts,
            },
            "detail": "uncertain은 observed로 승격하지 않고 별도 검토",
        },
        {
            "id": "manual_review",
            "label": "수동 evidence 검수",
            "state": (
                "pass" if manual_gate_passed else
                "warn" if observed_review_rejected or frame_review_rejected else
                "pending"
            ),
            "metric": {
                "required": True,
                "observed_reviewed": observed_reviewed,
                "observed_accepted": observed_review_accepted,
                "observed_rejected": observed_review_rejected,
                "observed_acceptance_ratio": round(observed_review_ratio, 4),
                "frame_reviewed": frame_reviewed,
                "frame_accepted": frame_review_accepted,
                "frame_rejected": frame_review_rejected,
                "frame_acceptance_ratio": round(frame_review_ratio, 4),
            },
            "detail": "최종 90% 통과는 observed 객체와 keyframe 위치 수동 검수 90% 이상이 필요",
        },
    ]

    return {
        "ok": True,
        "upload_id": upload_id,
        "model_id": model_id,
        "status": report_status,
        "automated_gate_passed": automated_gate_passed,
        "manual_gate_passed": manual_gate_passed,
        "manual_review_required": True,
        "passed_90": manual_gate_passed,
        "summary": {
            "objects": len(objects),
            "status_counts": status_counts,
            "observed_count": observed_count,
            "observed_with_evidence": observed_with_evidence,
            "observed_evidence_ratio": round(observed_evidence_ratio, 4),
            "observed_support_ge3": observed_support_ge3,
            "observed_support_ratio": round(observed_support_ratio, 4),
            "observed_reviewed": observed_reviewed,
            "observed_review_accepted": observed_review_accepted,
            "observed_review_rejected": observed_review_rejected,
            "observed_review_acceptance_ratio": round(observed_review_ratio, 4),
            "frame_reviewed": frame_reviewed,
            "frame_review_accepted": frame_review_accepted,
            "frame_review_rejected": frame_review_rejected,
            "frame_review_acceptance_ratio": round(frame_review_ratio, 4),
            "alignment_quality": alignment.get("quality") if isinstance(alignment, dict) else None,
            "coverage_generated_at": coverage_doc.get("generated_at") if isinstance(coverage_doc, dict) else None,
            "review_updated_at": review_doc.get("updated_at"),
        },
        "criteria": criteria,
        "blockers": blockers,
        "warnings": warnings,
    }


@app.post("/api/uploads/{upload_id}/alignment")
async def save_alignment(upload_id: str, payload: dict):
    if _bad_id(upload_id, "upload_"):
        return JSONResponse({"ok": False, "error": "bad id"}, status_code=400)
    model_id = payload.get("model_id") or _default_model_id()
    if model_id not in _model_registry():
        return JSONResponse({"ok": False, "error": "unknown model_id"}, status_code=404)
    try:
        input_pairs = _normalize_alignment_pairs((payload.get("alignment") or {}).get("pairs", []))
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    if len(input_pairs) < 4:
        candidate_id = payload.get("candidate_id")
        if candidate_id:
            # Option B: explicit save of a corroborated prior candidate (can_save=true only)
            cand_doc = _load_scoped_upload_json(upload_id, "alignment_candidates", model_id)
            cand = next(
                (c for c in (cand_doc or {}).get("candidates", []) if c.get("id") == candidate_id),
                None,
            )
            if not cand or not cand.get("can_save") or not isinstance(cand.get("alignment"), dict):
                return JSONResponse(
                    {
                        "ok": False,
                        "status": "candidate_not_saveable",
                        "error": "candidate not found, or not corroborated for saving",
                    },
                    status_code=409,
                )
            try:
                manifest = _model_manifest(model_id)
                if not manifest.get("manifest_valid"):
                    return JSONResponse(
                        {"ok": False, "status": "model_manifest_invalid", "error": "model manifest invalid", "model": manifest},
                        status_code=409,
                    )
                alignment = dict(cand["alignment"])
                alignment.setdefault("pairs", [])
                alignment["source"] = cand.get("source")
                alignment["candidate_id"] = candidate_id
                if cand.get("corroboration"):
                    alignment["corroboration"] = cand["corroboration"]
                alignment_doc = {
                    "ok": True,
                    "upload_id": upload_id,
                    "model_id": model_id,
                    "model": manifest,
                    "alignment": alignment,
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
                (UPLOAD_DIR / f"{upload_id}.{model_id}.alignment.json").write_text(
                    json.dumps(alignment_doc, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                (UPLOAD_DIR / f"{upload_id}.alignment.json").write_text(
                    json.dumps(alignment_doc, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return alignment_doc
            except Exception as e:
                log.exception("candidate alignment save failed: %s", e)
                return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
        return JSONResponse(
            {
                "ok": False,
                "status": "needs_alignment",
                "error": "alignment save requires at least 4 scan/model correspondence pairs",
            },
            status_code=409,
        )
    try:
        manifest = _model_manifest(model_id)
        if not manifest.get("manifest_valid"):
            return JSONResponse(
                {"ok": False, "status": "model_manifest_invalid", "error": "model manifest invalid", "model": manifest},
                status_code=409,
            )
        alignment = _solve_scan_to_model_alignment(input_pairs)
        alignment["pairs"] = input_pairs
        alignment_doc = {
            "ok": True,
            "upload_id": upload_id,
            "model_id": model_id,
            "model": manifest,
            "alignment": alignment,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        (UPLOAD_DIR / f"{upload_id}.{model_id}.alignment.json").write_text(
            json.dumps(alignment_doc, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (UPLOAD_DIR / f"{upload_id}.alignment.json").write_text(
            json.dumps(alignment_doc, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return alignment_doc
    except Exception as e:
        log.exception("alignment save failed: %s", e)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/uploads/{upload_id}/coverage/analyze")
async def analyze_coverage(upload_id: str, payload: dict):
    """Generate conservative model-object observation evidence from scan proximity."""
    if _bad_id(upload_id, "upload_"):
        return JSONResponse({"ok": False, "error": "bad id"}, status_code=400)
    model_id = payload.get("model_id") or _default_model_id()
    if model_id not in _model_registry():
        return JSONResponse({"ok": False, "error": "unknown model_id"}, status_code=404)
    opts = payload.get("coverage_options", {}) or {}
    dist_thresh = float(opts.get("distance_threshold_m", 0.10))
    min_support = int(opts.get("min_support_frames", 3))
    min_consistency = float(opts.get("min_geometry_consistency", 0.6))
    require_visibility = bool(opts.get("require_visibility", True))
    visibility_max_depth_m = float(opts.get("visibility_max_depth_m", 15.0))
    visibility_margin_px = float(opts.get("visibility_margin_px", 32.0))
    preferred_variant = payload.get("variant") or "detail"
    try:
        input_pairs = _normalize_alignment_pairs((payload.get("alignment") or {}).get("pairs", []))
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    try:
        manifest, objects = _load_model_objects(model_id)
        manifest = _model_manifest(model_id)
        if not manifest.get("manifest_valid"):
            return JSONResponse(
                {"ok": False, "status": "model_manifest_invalid", "error": "model manifest invalid", "model": manifest},
                status_code=409,
            )
        scan_path = _select_scan_payload(upload_id, preferred_variant)
        if scan_path is None:
            return JSONResponse({"ok": False, "error": "scan payload not found"}, status_code=404)
        scan = _parse_scan_payload(scan_path, include_points=True, max_points=500_000)
        scan_points = np.asarray(scan.get("points") or [], dtype=np.float64)
        if scan_points.size == 0:
            return JSONResponse({"ok": False, "error": "scan points not available"}, status_code=400)
        scan_points = _viewer_points(scan_points)
        alignment_source = "request_pairs"
        if input_pairs:
            if len(input_pairs) < 4:
                return JSONResponse(
                    {
                        "ok": False,
                        "status": "needs_alignment",
                        "error": "coverage analysis requires at least 4 correspondence pairs when pairs are provided",
                    },
                    status_code=409,
                )
            alignment = _solve_scan_to_model_alignment(input_pairs)
            alignment["pairs"] = input_pairs
        else:
            alignment_doc = _load_scoped_upload_json(upload_id, "alignment", model_id)
            alignment = alignment_doc.get("alignment") if isinstance(alignment_doc, dict) else None
            if not isinstance(alignment, dict):
                return JSONResponse(
                    {
                        "ok": False,
                        "status": "needs_alignment",
                        "error": "coverage analysis requires saved alignment or at least 4 correspondence pairs",
                    },
                    status_code=409,
                )
            alignment_source = "saved_alignment"
        if alignment.get("quality") == "red":
            return JSONResponse(
                {
                    "ok": False,
                    "status": "needs_alignment",
                    "error": "coverage analysis requires non-red scan/model alignment quality",
                    "alignment": alignment,
                },
                status_code=409,
            )
        scan_model = _apply_sim3(scan_points, alignment)
        poses = scan.get("poses") or []
        K = np.asarray(scan.get("K"), dtype=np.float64) if scan.get("K") is not None else None
        camera_states = _aligned_camera_states(poses, alignment) if alignment["quality"] != "red" else []

        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(scan_model)
            tree_mode = "scipy.cKDTree"
        except Exception:
            tree = None
            tree_mode = "numpy_fallback"

        source_ids = scan.get("source_ids")
        source_ids_arr = np.asarray(source_ids, dtype=np.uint32) if source_ids is not None else None

        results = []
        status_counts: dict[str, int] = {}
        for obj in objects:
            samples, ts = _object_samples(obj)
            if tree is not None:
                dists, idx = tree.query(samples, k=1, workers=-1)
            else:
                # Fallback for small inputs only; still keeps the API functional.
                diff = samples[:, None, :] - scan_model[None, :, :]
                all_d = np.linalg.norm(diff, axis=2)
                idx = all_d.argmin(axis=1)
                dists = all_d[np.arange(samples.shape[0]), idx]
            # M1b: 객체 샘플은 중심선/중심점이므로 표면 반경만큼 허용 거리를 가산
            hit = dists <= (dist_thresh + _object_surface_radius(obj))
            coverage = float(hit.mean()) if hit.size else 0.0
            distribution_metrics = {
                "sample_count": int(hit.size),
                "hit_sample_count": int(hit.sum()) if hit.size else 0,
                "segment_hit_bins": None,
                "segment_total_bins": None,
                "linear_hit_span_ratio": None,
                "linear_longest_gap_ratio": None,
            }
            if obj.get("is_linear"):
                linear_metrics = _linear_hit_metrics(ts, hit)
                coverage = linear_metrics["coverage_ratio"]
                segment_coverage = linear_metrics["segment_coverage_ratio"]
                geometry_consistency = linear_metrics["geometry_consistency"]
                distribution_metrics.update({
                    "sample_count": linear_metrics["sample_count"],
                    "hit_sample_count": linear_metrics["hit_sample_count"],
                    "segment_hit_bins": linear_metrics["segment_hit_bins"],
                    "segment_total_bins": linear_metrics["segment_total_bins"],
                    "linear_hit_span_ratio": linear_metrics["linear_hit_span_ratio"],
                    "linear_longest_gap_ratio": linear_metrics["linear_longest_gap_ratio"],
                })
            else:
                segment_coverage = coverage
                geometry_consistency = 1.0 if coverage > 0 else 0.0
            support_frames: list[int] = []
            support_frame_source = "none"
            if source_ids_arr is not None and len(idx):
                support_frames = sorted({int(source_ids_arr[int(i)]) for i, ok in zip(idx, hit) if ok})
                support_frame_source = "source_ids"
            elif len(idx) and np.any(hit):
                hit_indices = np.asarray(idx, dtype=np.int64)[hit]
                hit_points = scan_model[hit_indices]
                support_frames = _nearest_camera_frames_for_points(hit_points, camera_states, max_frames=12)
                support_frame_source = "nearest_camera_pose" if support_frames else "none"
            visibility_frames = _visible_frames_for_samples(
                samples,
                camera_states,
                K,
                max_depth_m=visibility_max_depth_m,
                margin_px=visibility_margin_px,
            )
            if require_visibility and alignment["quality"] != "red":
                visible_set = set(visibility_frames)
                support_frames = [f for f in support_frames if f in visible_set]
            support_count = len(support_frames)
            support_factor = min(1.0, support_count / max(1, min_support))
            alignment_factor = 1.0 if alignment["quality"] == "green" else 0.8 if alignment["quality"] == "yellow" else 0.0
            visibility_factor = 1.0 if visibility_frames or not require_visibility else 0.0
            evidence_score = float(alignment_factor * visibility_factor * (
                0.40 * coverage + 0.25 * segment_coverage + 0.15 * geometry_consistency + 0.20 * support_factor
            ))
            if evidence_score >= 0.75 and support_count >= min_support:
                confidence = "high"
            elif evidence_score >= 0.45:
                confidence = "medium"
            elif evidence_score > 0:
                confidence = "low"
            else:
                confidence = "none"
            if alignment["quality"] == "red" and coverage >= 0.30:
                status = "uncertain"
            elif alignment["quality"] != "red" and require_visibility and not visibility_frames:
                status = "out_of_scope"
            elif (
                coverage >= 0.70
                and segment_coverage >= min_consistency
                and support_count >= min_support
                and geometry_consistency >= min_consistency
                and alignment["quality"] in ("green", "yellow")
            ):
                status = "observed"
            elif coverage >= 0.30 or segment_coverage >= 0.30:
                status = "likely_observed" if alignment["quality"] in ("green", "yellow") else "uncertain"
            elif support_count == 0:
                status = "not_observed"
            else:
                status = "uncertain"
            status_counts[status] = status_counts.get(status, 0) + 1
            results.append({
                "guid": obj["guid"],
                "category": obj["category"],
                "system": obj.get("system"),
                "zone": obj.get("zone"),
                "status": status,
                "coverage_ratio": round(coverage, 4),
                "segment_coverage_ratio": round(segment_coverage, 4),
                "nearest_distance_median_m": float(np.median(dists)) if len(dists) else None,
                "nearest_distance_max_m": float(np.max(dists)) if len(dists) else None,
                "geometry_consistency": round(float(geometry_consistency), 4),
                "sample_count": distribution_metrics["sample_count"],
                "hit_sample_count": distribution_metrics["hit_sample_count"],
                "segment_hit_bins": distribution_metrics["segment_hit_bins"],
                "segment_total_bins": distribution_metrics["segment_total_bins"],
                "linear_hit_span_ratio": (
                    round(float(distribution_metrics["linear_hit_span_ratio"]), 4)
                    if distribution_metrics["linear_hit_span_ratio"] is not None else None
                ),
                "linear_longest_gap_ratio": (
                    round(float(distribution_metrics["linear_longest_gap_ratio"]), 4)
                    if distribution_metrics["linear_longest_gap_ratio"] is not None else None
                ),
                "visibility_frames": visibility_frames,
                "support_frames": support_frames,
                "support_frame_source": support_frame_source,
                "support_frame_count": support_count,
                "visibility_frame_count": len(visibility_frames),
                "evidence_score": round(evidence_score, 4),
                "confidence": confidence,
                "evidence_thumb_ids": support_frames[:8],
                "evidence_keyframes": [f"/api/uploads/{upload_id}/keyframes/{f}.jpg" for f in support_frames[:8]],
                "center": obj.get("center"),
                "start": obj.get("start"),
                "end": obj.get("end"),
                "bbox": obj.get("bbox"),
            })
        coverage_doc = {
            "ok": True,
            "upload_id": upload_id,
            "model": manifest,
            "alignment": alignment,
            "options": {
                "distance_threshold_m": dist_thresh,
                "min_support_frames": min_support,
                "min_geometry_consistency": min_consistency,
                "require_visibility": require_visibility,
                "visibility_max_depth_m": visibility_max_depth_m,
                "visibility_margin_px": visibility_margin_px,
                "tree": tree_mode,
                "scan_payload": scan_path.name,
                "scan_points_sampled": int(scan_points.shape[0]),
                "alignment_source": alignment_source,
            },
            "status_counts": status_counts,
            "objects": results,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        out = UPLOAD_DIR / f"{upload_id}.{model_id}.coverage.json"
        out.write_text(json.dumps(coverage_doc, ensure_ascii=False, indent=2), encoding="utf-8")
        (UPLOAD_DIR / f"{upload_id}.coverage.json").write_text(json.dumps(coverage_doc, ensure_ascii=False, indent=2), encoding="utf-8")
        if alignment_source == "request_pairs":
            alignment_doc = {
                "ok": True,
                "upload_id": upload_id,
                "model_id": model_id,
                "model": manifest,
                "alignment": alignment,
                "generated_at": coverage_doc["generated_at"],
            }
            (UPLOAD_DIR / f"{upload_id}.{model_id}.alignment.json").write_text(
                json.dumps(alignment_doc, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (UPLOAD_DIR / f"{upload_id}.alignment.json").write_text(
                json.dumps(alignment_doc, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return coverage_doc
    except Exception as e:
        log.exception("coverage analysis failed: %s", e)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


def _load_audit_results() -> dict:
    if not AUDIT_RESULTS.exists():
        return {"summary": {"passed": False, "note": "audit_results.json not found"}, "items": []}
    return json.loads(AUDIT_RESULTS.read_text())


@app.get("/api/audit/results")
async def audit_results():
    return _load_audit_results()


@app.post("/api/audit/replay/{upload_id}")
async def replay_audit_upload(upload_id: str, variant: str = "mesh"):
    """Replay a generated audit payload without replacing the original upload."""
    if not upload_id.startswith("upload_") or "/" in upload_id or ".." in upload_id:
        return JSONResponse({"ok": False, "error": "bad id"}, status_code=400)
    audit = _load_audit_results()
    allowed = {it.get("id") for it in audit.get("items", [])}
    if upload_id not in allowed:
        return JSONResponse({"ok": False, "error": "not in audit results"}, status_code=404)
    variant_suffix = {
        "mesh": "lbm1.mesh",
        "texture": "texture",
        "visual": "visual",
        "detail": "test",
    }
    if variant not in variant_suffix:
        return JSONResponse({"ok": False, "error": "variant must be mesh, texture, visual, or detail"}, status_code=400)
    suffix = variant_suffix[variant]
    p = UPLOAD_DIR / f"{upload_id}.{suffix}" if variant == "mesh" else UPLOAD_DIR / f"{upload_id}.lbp4.{suffix}"
    if not p.exists():
        return JSONResponse({"ok": False, "error": f"{variant} payload not found"}, status_code=404)
    payload = p.read_bytes()
    thumbs = _load_upload_thumbs(upload_id, prefer_audit=True)
    await broadcast_point_cloud(payload, thumbs)
    return {"ok": True, "id": upload_id, "variant": variant, "bytes": len(payload),
            "thumbs": len(thumbs) if thumbs else 0,
            "viewers": len(STATE.viewer_sockets)}


@app.get("/api/audit/glb/{upload_id}")
async def audit_glb(upload_id: str):
    """Download the generated scan mesh as GLB."""
    if not upload_id.startswith("upload_") or "/" in upload_id or ".." in upload_id:
        return JSONResponse({"ok": False, "error": "bad id"}, status_code=400)
    audit = _load_audit_results()
    allowed = {it.get("id") for it in audit.get("items", [])}
    if upload_id not in allowed:
        return JSONResponse({"ok": False, "error": "not in audit results"}, status_code=404)
    p = UPLOAD_DIR / f"{upload_id}.mesh.glb"
    if not p.exists():
        return JSONResponse({"ok": False, "error": "GLB mesh not found"}, status_code=404)
    return FileResponse(
        p,
        media_type="model/gltf-binary",
        filename=p.name,
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/uploads/{upload_id}/replay")
async def replay_upload(upload_id: str):
    """Re-broadcast a persisted payload to all WS viewers."""
    # guard against path traversal
    if not upload_id.startswith("upload_") or "/" in upload_id or ".." in upload_id:
        return JSONResponse({"ok": False, "error": "bad id"}, status_code=400)
    p = UPLOAD_DIR / f"{upload_id}.lbp2"
    if not p.exists():
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    payload = p.read_bytes()
    thumbs = _load_upload_thumbs(upload_id)
    await broadcast_point_cloud(payload, thumbs)
    return {"ok": True, "id": upload_id, "bytes": len(payload),
            "thumbs": len(thumbs) if thumbs else 0,
            "viewers": len(STATE.viewer_sockets)}


@app.get("/preview.jpg")
async def preview_jpg():
    if STATE.latest_jpeg is None:
        # 1x1 black placeholder
        ok, buf = cv2.imencode(".jpg", np.zeros((1, 1, 3), dtype=np.uint8))
        return StreamingResponse(iter([buf.tobytes()]), media_type="image/jpeg")
    return StreamingResponse(iter([STATE.latest_jpeg]), media_type="image/jpeg",
                             headers={"Cache-Control": "no-store"})


@app.get("/preview.mjpeg")
async def preview_mjpeg():
    boundary = b"--frame"
    async def gen():
        last_version = -1
        while True:
            try:
                await asyncio.wait_for(STATE.frame_event.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
            if STATE.latest_jpeg is None or STATE.latest_jpeg_version == last_version:
                continue
            last_version = STATE.latest_jpeg_version
            yield (boundary + b"\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(STATE.latest_jpeg)).encode() + b"\r\n\r\n"
                   + STATE.latest_jpeg + b"\r\n")
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/health")
async def health():
    return {
        "ok": True,
        "frames_received": STATE.frames_received,
        "uptime_s": round(time.time() - STATE.start_ts, 1),
        "model_loaded": STATE.worker is not None,
        "viewers": len(STATE.viewer_sockets),
        "last_infer_age_s": round(time.time() - STATE.last_infer_ts, 1) if STATE.last_infer_ts else None,
        "last_infer_points": STATE.last_infer_points,
    }


async def broadcast_point_cloud(payload: bytes, thumbs: list[bytes] | None = None) -> None:
    STATE.last_infer_ts = time.time()
    STATE.last_payload = payload
    STATE.last_thumbs = thumbs
    # LBP2/LBP3/LBP4: magic + flags + num_pts + num_poses + seq
    if len(payload) >= 20 and payload[:4] in (b"LBP2", b"LBP3", b"LBP4"):
        import struct
        _, _flags, num_pts, _np, _seq = struct.unpack("<4sIIII", payload[:20])
        STATE.last_infer_points = num_pts
    # JSON sidecar with per-frame thumbnails (base64-encoded JPEGs). We send
    # this BEFORE the binary so the viewer can cache thumbs and apply them as
    # the camera frustums are built from the point-cloud payload.
    thumbs_msg = None
    if thumbs:
        thumbs_msg = json.dumps({
            "type": "thumbs",
            "images": [base64.b64encode(t).decode("ascii") for t in thumbs],
        })
    dead = []
    for ws in list(STATE.viewer_sockets):
        try:
            if thumbs_msg is not None:
                await ws.send_text(thumbs_msg)
            await ws.send_bytes(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        STATE.viewer_sockets.discard(ws)


@app.websocket("/ws")
async def ws_viewer(ws: WebSocket):
    await ws.accept()
    STATE.viewer_sockets.add(ws)
    log.info("viewer connected (total=%d)", len(STATE.viewer_sockets))
    # replay last payload so late-joining viewers see the most recent result
    if STATE.last_payload is not None:
        try:
            if STATE.last_thumbs:
                await ws.send_text(json.dumps({
                    "type": "thumbs",
                    "images": [base64.b64encode(t).decode("ascii") for t in STATE.last_thumbs],
                }))
            await ws.send_bytes(STATE.last_payload)
        except Exception as e:
            log.warning("replay to new viewer failed: %s", e)
    try:
        while True:
            # keep-alive: read pings (we don't actually need client messages)
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                await ws.send_text('{"type":"ping"}')
    except WebSocketDisconnect:
        pass
    finally:
        STATE.viewer_sockets.discard(ws)
        log.info("viewer disconnected (total=%d)", len(STATE.viewer_sockets))


@app.post("/offer")
async def offer(req: Request):
    params = await req.json()
    offer_desc = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection()
    STATE.pcs.add(pc)
    log.info(f"new PeerConnection (total={len(STATE.pcs)})")

    # phone -> server: a DataChannel will be created by the phone for control/stats.
    # We grab it via @pc.on("datachannel").
    server_dc = {"chan": None}

    @pc.on("datachannel")
    def on_datachannel(channel):
        log.info(f"datachannel from client: {channel.label}")
        server_dc["chan"] = channel

        @channel.on("message")
        def on_msg(msg):
            log.info(f"client says: {msg!r}")

    @pc.on("connectionstatechange")
    async def on_state():
        log.info(f"pc state: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed"):
            STATE.pcs.discard(pc)
            await pc.close()

    @pc.on("track")
    def on_track(track):
        log.info(f"track received: {track.kind} ({track.id})")
        if track.kind == "video":
            # need a DataChannel to push stats — wait briefly then start.
            async def start_consumer():
                for _ in range(40):
                    if server_dc["chan"] is not None:
                        break
                    await asyncio.sleep(0.05)
                FrameConsumer(track, server_dc["chan"])
            asyncio.create_task(start_consumer())

    await pc.setRemoteDescription(offer_desc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return JSONResponse({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})


@app.on_event("startup")
async def on_startup():
    model_path = os.environ.get("LINGBOT_MODEL", "ckpts/lingbot-map-long.pt")
    if not Path(model_path).exists():
        log.warning("LINGBOT_MODEL not found at %s — running in transport-only mode", model_path)
        return
    out_mode = os.environ.get("LINGBOT_OUTPUT_MODE", "points")  # "points" | "mesh"
    log.info("starting inference worker (model=%s, output_mode=%s)", model_path, out_mode)
    try:
        STATE.worker = InferenceWorker(model_path, broadcast_point_cloud, output_mode=out_mode)
        STATE.worker.start()
    except Exception as e:
        # GPU OOM 등으로 워커가 못 떠도 coverage 웹 기능은 동작해야 한다 (업로드 처리만 비활성)
        log.error("inference worker init failed (%s); serving without video processing", e)
        STATE.worker = None


@app.on_event("shutdown")
async def on_shutdown():
    if STATE.worker is not None:
        await STATE.worker.stop()
    await asyncio.gather(*(pc.close() for pc in list(STATE.pcs)))
    STATE.pcs.clear()


# ---------------- self-signed cert ----------------

def ensure_cert(cert: Path, key: Path) -> None:
    if cert.exists() and key.exists():
        return
    log.info("generating self-signed cert (10y) for sannet use")
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-sha256",
        "-days", "3650", "-nodes",
        "-keyout", str(key), "-out", str(cert),
        "-subj", "/CN=lingbot-realtime",
        "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:0.0.0.0",
    ], check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--no-ssl", action="store_true", help="HTTP only (localhost development)")
    ap.add_argument("--cert", default=str(ROOT / "cert.pem"))
    ap.add_argument("--key", default=str(ROOT / "key.pem"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    if args.no_ssl:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    else:
        ensure_cert(Path(args.cert), Path(args.key))
        uvicorn.run(
            app, host=args.host, port=args.port,
            ssl_certfile=args.cert, ssl_keyfile=args.key,
            log_level="info",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

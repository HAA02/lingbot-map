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


UPLOAD_DIR = ROOT / "_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
BIM_DIR = ROOT / "_bim"
BIM_DIR.mkdir(exist_ok=True)


@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...), target_frames: int = 32):
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
    t0 = time.time()
    _payload, _thumbs, info = await STATE.worker.process_video_file(dest, target_frames=target_frames)
    dt = time.time() - t0
    # persist payload + thumbnails alongside the video so it can be replayed later
    upload_id = dest.stem  # e.g. "upload_1716329123456"
    if _payload is not None:
        (UPLOAD_DIR / f"{upload_id}.lbp2").write_bytes(_payload)
        if _thumbs:
            (UPLOAD_DIR / f"{upload_id}.thumbs.json").write_text(
                json.dumps([base64.b64encode(t).decode("ascii") for t in _thumbs])
            )
    return JSONResponse({
        "ok": _payload is not None,
        "id": upload_id,
        "file": dest.name,
        "bytes": size,
        "elapsed_s": round(dt, 2),
        **info,
    })


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
    """List past uploads with persisted point-cloud payloads, newest first."""
    items = []
    for p in sorted(UPLOAD_DIR.glob("upload_*.lbp2"),
                    key=lambda x: x.stat().st_mtime, reverse=True):
        upload_id = p.stem
        # find matching video (any common ext)
        video = None
        for ext in (".mp4", ".mov", ".webm", ".mkv", ".avi"):
            cand = UPLOAD_DIR / f"{upload_id}{ext}"
            if cand.exists():
                video = cand.name
                break
        ts_ms = int(upload_id.split("_")[-1]) if upload_id.count("_") >= 1 else 0
        items.append({
            "id": upload_id,
            "video": video,
            "payload_bytes": p.stat().st_size,
            "ts_ms": ts_ms,
        })
    return {"uploads": items}


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
    thumbs: list[bytes] | None = None
    tp = UPLOAD_DIR / f"{upload_id}.thumbs.json"
    if tp.exists():
        try:
            thumbs = [base64.b64decode(s) for s in json.loads(tp.read_text())]
        except Exception as e:
            log.warning("thumbs load failed for %s: %s", upload_id, e)
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
    # LBP2: magic + flags + num_pts + num_poses + seq
    if len(payload) >= 20 and payload[:4] == b"LBP2":
        import struct
        _, _flags, num_pts, _np, _seq = struct.unpack("<4sIIII", payload[:20])
        STATE.last_infer_points = num_pts
    # JSON sidecar with per-frame thumbnails (base64-encoded JPEGs). We send
    # this BEFORE the binary so the viewer can cache thumbs and apply them as
    # the camera frustums are built from the LBP2 payload.
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
    log.info("starting inference worker (model=%s)", model_path)
    STATE.worker = InferenceWorker(model_path, broadcast_point_cloud)
    STATE.worker.start()


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

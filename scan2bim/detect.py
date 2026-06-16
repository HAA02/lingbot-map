"""FR-2.3 object recognition — open-vocabulary detection on the (crisp) video
keyframes. The monocular POINT CLOUD is too noisy to match BIM structure, but
the 2D frames are sharp: OWL-ViT detects pipes/ducts/AC/lights/windows/columns
by text prompt. Non-repetitive objects (window, AC, light) become disambiguation
anchors — back-project their image position via the camera pose → 3D, match to
the model's same-type objects to pin location/yaw (where repeated pipes can't).

Heavy deps (transformers, OWL-ViT weights) are imported lazily.
"""
from __future__ import annotations

from pathlib import Path

# default vocabulary: distinctive (non-repetitive) anchors first
DEFAULT_PROMPTS = [
    "red pipe", "duct", "ceiling air conditioner", "ceiling light",
    "structural column", "window", "door",
]
_MODEL_ID = "google/owlv2-base-patch16-ensemble"
_cache = {}


def _load(model_id=_MODEL_ID):
    if model_id not in _cache:
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        proc = Owlv2Processor.from_pretrained(model_id)
        mdl = Owlv2ForObjectDetection.from_pretrained(model_id).eval()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        mdl = mdl.to(dev)
        _cache[model_id] = (proc, mdl, dev)
    return _cache[model_id]


def detect(image_paths, prompts=None, *, threshold: float = 0.15, model_id=_MODEL_ID) -> list:
    """Detect prompted objects per image.

    Returns [{path, detections:[{label, score, bbox[x0,y0,x1,y1], cx, cy}]}].
    cx,cy = box center in pixels (for back-projection via the camera pose).
    """
    import torch
    from PIL import Image
    prompts = prompts or DEFAULT_PROMPTS
    proc, mdl, dev = _load(model_id)
    out = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        inp = proc(text=[prompts], images=img, return_tensors="pt").to(dev)
        with torch.inference_mode():
            res = mdl(**inp)
        tgt = torch.tensor([img.size[::-1]]).to(dev)
        r = proc.post_process_grounded_object_detection(res, threshold=threshold, target_sizes=tgt)[0]
        dets = []
        for s, l, b in zip(r["scores"].tolist(), r["labels"].tolist(), r["boxes"].tolist()):
            x0, y0, x1, y1 = [float(v) for v in b]
            dets.append({"label": prompts[l], "score": round(float(s), 3),
                         "bbox": [round(x0), round(y0), round(x1), round(y1)],
                         "cx": round((x0 + x1) / 2, 1), "cy": round((y0 + y1) / 2, 1)})
        dets.sort(key=lambda d: -d["score"])
        out.append({"path": str(p), "detections": dets})
    return out


def summarize(results) -> dict:
    """Count detections by label across frames (which anchors are available)."""
    from collections import Counter
    c = Counter()
    for r in results:
        for d in r["detections"]:
            c[d["label"]] += 1
    return dict(c)

"""Scoring container: one photo in, face crop + raw model numbers out.

Runs the same code as the pipeline (sinemart.crop, sinemart.embed), so a
photo scored here lands exactly where the pipeline would put it. It knows
nothing about taste: the scorer app applies the taste space (R2
models/<key>/taste.json) and the current weights (D1) itself, so a retrain
or a reloaded model never needs a new image.

    POST /embed   body: the photo's bytes
                  -> {"face": bool, "ok": bool, "reasons": [...],
                      "crop": base64 JPEG, "p_female": float,
                      "emb": {"m1": [768 numbers], "m6": [...]}}
    GET  /health  -> {"ready": bool}

Photos are kept in memory only: nothing is written to disk or logged.

    python server.py --prep [KEY]  # at image build: download weights (one model, or all),
                                   # cache the gender prompts
    python server.py           # serve on :8080
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.environ.get("PIPELINE_DIR", "/app/pipeline"))

MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
# App key -> sinemart.embed model name: Model 6 (the leaderboard's ranker
# since 2026-10-27) and Model 1. More can be added here (and to the image)
# without touching the app's maths.
KEYS = dict(kv.split(":") for kv in os.environ.get("SCORER_MODELS", "m1:siglip2,m6:clipl").split(","))
MAX_BYTES = 20 * 1024 * 1024

state: dict = {"ready": False}


def prep(only: str | None = None) -> None:
    """Image build step: fetch the models' weights and cache the gender
    prompts, so the running container needs no internet. One model per
    Docker layer keeps each layer under ~2 GB: the registry timed out
    committing a single 3.25 GB layer (2026-10-27)."""
    import cv2  # noqa: F401 - fail the build, not the running box, if a library is missing
    import mediapipe  # noqa: F401

    from sinemart.crop import Detector, Segmenter  # noqa: F401
    from sinemart.embed import load
    from sinemart.gender import PROMPTS

    for key, name in KEYS.items():
        if only in (None, key):
            load(name)
    if only not in (None, "m1"):
        print("prepared", only)
        return
    import torch
    from transformers import AutoModel, AutoProcessor

    from sinemart.embed import MODELS
    proc = AutoProcessor.from_pretrained(MODELS["siglip2"])
    model = AutoModel.from_pretrained(MODELS["siglip2"]).eval()
    out = {}
    with torch.inference_mode():
        for g, prompts in PROMPTS.items():
            x = proc(text=prompts, padding="max_length", max_length=64, return_tensors="pt")
            t = model.get_text_features(**x)
            t = t if isinstance(t, torch.Tensor) else t.pooler_output
            out[g] = torch.nn.functional.normalize(t.float(), dim=-1).numpy()
    np.savez(os.path.join(MODELS_DIR, "gender_text.npz"), **out)
    print("prepared", only or ", ".join(KEYS))


def warm() -> None:
    try:
        _warm()
    except Exception as e:  # noqa: BLE001 - say so on /health instead of "warming up" forever
        state["error"] = f"{type(e).__name__}: {e}"
        print("failed to start:", state["error"], flush=True)


def _warm() -> None:
    from sinemart.crop import Detector, Segmenter
    from sinemart.embed import load

    state["detector"] = Detector(os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx"))
    state["segmenter"] = Segmenter(os.path.join(MODELS_DIR, "selfie_segmenter.tflite"))
    state["encode"] = {key: load(name) for key, name in KEYS.items()}
    state["text"] = dict(np.load(os.path.join(MODELS_DIR, "gender_text.npz")))
    state["lock"] = threading.Lock()  # one photo at a time: the models aren't thread-safe
    state["ready"] = True
    print("ready:", ", ".join(KEYS), flush=True)


def embed(photo: bytes) -> dict:
    import cv2
    from PIL import Image

    from sinemart.crop import process
    from sinemart.embed import per_person
    from sinemart.gender import p_female

    # cv2.imdecode doesn't apply the EXIF rotation that the pipeline's
    # cv2.imread does; read through an in-memory file to match it exactly.
    path = f"/dev/shm/photo-{threading.get_ident()}"
    try:
        with open(path, "wb") as f:
            f.write(photo)
        img = cv2.imread(path)
    finally:
        if os.path.exists(path):
            os.remove(path)
    if img is None:
        return {"face": False, "ok": False, "reasons": ["unreadable"]}
    with state["lock"]:
        crop, q = process(img, state["detector"], state["segmenter"])
        if crop is None:
            return {"face": False, "ok": False, "reasons": q.reasons}
        # The pipeline saves crops as JPEG 90 and embeds the saved file: do
        # the same round trip so the numbers match.
        jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
        im = Image.open(io.BytesIO(jpg)).convert("RGB")
        emb = {}
        for key, encode in state["encode"].items():
            v = encode([im])
            emb[key] = per_person([0], v)[1][0].astype(float).round(7).tolist()
    pf = None
    if "m1" in emb:  # gender guess from Model 1's image numbers, as sinemart.gender does
        pf = float(p_female(np.array([emb["m1"]], np.float32), state["text"]["f"], state["text"]["m"])[0])
    return {"face": True, "ok": q.ok, "reasons": q.reasons, "crop": base64.b64encode(jpg).decode(),
            "p_female": pf, "emb": emb}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            return self._send(200, {"ready": state["ready"], "models": list(KEYS), "error": state.get("error")})
        self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path != "/embed":
            return self._send(404, {"error": "not found"})
        if state.get("error"):
            return self._send(500, {"error": "failed to start"})
        if not state["ready"]:
            return self._send(503, {"error": "warming up"})
        n = int(self.headers.get("content-length") or 0)
        if not 0 < n <= MAX_BYTES:
            return self._send(413, {"error": "photo missing or too large"})
        try:
            self._send(200, embed(self.rfile.read(n)))
        except Exception as e:  # noqa: BLE001 - report, never crash the server
            self._send(500, {"error": type(e).__name__})

    def log_message(self, *args):  # no request logs: nothing about photos leaves the box
        pass


if __name__ == "__main__":
    if "--prep" in sys.argv:
        rest = sys.argv[sys.argv.index("--prep") + 1:]
        prep(rest[0] if rest else None)
        sys.exit()
    server = ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT", "8080"))), Handler)
    threading.Thread(target=warm, daemon=True).start()  # answer /health while the models load
    server.serve_forever()

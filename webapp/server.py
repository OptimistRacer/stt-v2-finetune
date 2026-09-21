"""Local web UI for the Urdu TTS voice: type Urdu text, get speech back.

Runs on this machine because the model and GPU do. Nothing leaves the machine.

The model is loaded once at startup and held in memory -- loading it per request would
add several seconds to every click, and it only costs a few hundred MB of VRAM.

Usage:
    python webapp/server.py
    python webapp/server.py --host 0.0.0.0 --port 8090   # reachable from the LAN
"""
import argparse
import base64
import io
import socket
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synthesize import MODEL, PAUSE_S, split_sentences

app = FastAPI(title="Urdu TTS")
STATE: dict = {}
HERE = Path(__file__).resolve().parent
MAX_CHARS = 5000


class Request(BaseModel):
    text: str
    split: bool = True
    speaking_rate: float = 0.9
    seed: int = 1234


def get_tts():
    if "tts" not in STATE:
        from transformers import pipeline

        print(f"[server] loading voice from {STATE['model_dir']} ...")
        t0 = time.time()
        STATE["tts"] = pipeline("text-to-speech", STATE["model_dir"],
                                device=0 if torch.cuda.is_available() else -1)
        print(f"[server] voice ready in {time.time() - t0:.1f}s")
    return STATE["tts"]


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (HERE / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "model_dir": STATE["model_dir"], "loaded": "tts" in STATE,
            "cuda": torch.cuda.is_available()}


@app.post("/synthesize")
def synthesize(req: Request):
    text = (req.text or "").strip()
    if not text:
        return JSONResponse({"error": "Type some Urdu text first."}, status_code=400)
    if len(text) > MAX_CHARS:
        return JSONResponse({"error": f"Text is too long ({len(text)} characters, limit {MAX_CHARS})."},
                            status_code=400)

    tts = get_tts()
    # Sentence-splitting matters: on a 7-sentence paragraph it halved the word error
    # against feeding the whole thing in at once (0.067 vs 0.135).
    chunks = split_sentences(text) if req.split else [text]

    t0 = time.time()
    pieces, sr = [], 16000
    original_rate = tts.model.speaking_rate
    try:
        tts.model.speaking_rate = req.speaking_rate
        for i, chunk in enumerate(chunks):
            torch.manual_seed(req.seed + i)
            out = tts(chunk)
            sr = out["sampling_rate"]
            pieces.append(out["audio"].squeeze().astype(np.float32))
            if i < len(chunks) - 1:
                pieces.append(np.zeros(int(PAUSE_S * sr), dtype=np.float32))
    finally:
        tts.model.speaking_rate = original_rate

    audio = np.concatenate(pieces)
    elapsed = time.time() - t0

    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, rate=sr, data=audio)
    duration = len(audio) / sr
    return {
        "audio": base64.b64encode(buf.getvalue()).decode("ascii"),
        "duration_s": round(duration, 1),
        "elapsed_s": round(elapsed, 1),
        "speed": round(duration / elapsed, 1) if elapsed else None,
        "n_sentences": len(chunks),
        "sample_rate": sr,
    }


if __name__ == "__main__":
    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default=MODEL)
    p.add_argument("--host", default="127.0.0.1")
    # Same Windows port caveat as the STT app: 8000 is often unbindable because
    # Hyper-V/WSL reserve port blocks, and it fails as a permission error.
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--preload", action="store_true", help="Load the voice now rather than on first request")
    a = p.parse_args()

    STATE.update(model_dir=a.model_dir)
    if not Path(a.model_dir).exists():
        sys.exit(f"No model at {a.model_dir} -- run export_model.py first")
    if a.preload:
        get_tts()

    port = a.port
    for candidate in [a.port, 8090, 7861, 8766, 5001, 9001]:
        s = socket.socket()
        try:
            s.bind((a.host, candidate))
            port = candidate
            break
        except OSError:
            continue
        finally:
            s.close()
    if port != a.port:
        print(f"[server] port {a.port} unavailable, using {port} instead")

    print(f"\n  Urdu TTS running at  http://{a.host}:{port}\n")
    uvicorn.run(app, host=a.host, port=port, log_level="warning")

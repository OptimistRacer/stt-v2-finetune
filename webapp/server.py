"""Local web UI for the Urdu speech models: transcription and synthesis in one app.

Runs on this machine because the models and GPU do. Nothing leaves the machine --
uploads go to a temp file, get transcribed, and are deleted.

Each model is loaded on first use and then held in memory; loading per request would
add seconds to every click. Whisper costs ~2GB of VRAM, the voice a few hundred MB.

The STT half imports from the stt-v2-finetune project rather than duplicating it.

Usage:
    python webapp/server.py
    python webapp/server.py --host 0.0.0.0 --port 8095   # reachable from the LAN
"""
import argparse
import base64
import io
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, r"C:\Users\Sistems-AI\stt-v2-finetune")

from synthesize import MODEL as TTS_MODEL, PAUSE_S, split_sentences
from scripts.transcribe import CODESWITCH_PROMPT, DEFAULT_MODEL as STT_MODEL, _chunked, to_srt
from src.postprocess.codeswitch import CodeSwitcher

app = FastAPI(title="Urdu Speech")
STATE: dict = {}
HERE = Path(__file__).resolve().parent
MAX_CHARS = 5000


class SpeakRequest(BaseModel):
    text: str
    split: bool = True
    speaking_rate: float = 0.9
    seed: int = 1234


def get_tts():
    if "tts" not in STATE:
        from transformers import pipeline

        print(f"[server] loading voice from {STATE['tts_dir']} ...")
        t0 = time.time()
        STATE["tts"] = pipeline("text-to-speech", STATE["tts_dir"],
                                device=0 if torch.cuda.is_available() else -1)
        print(f"[server] voice ready in {time.time() - t0:.1f}s")
    return STATE["tts"]


def get_stt():
    if "stt" not in STATE:
        from faster_whisper import WhisperModel

        print(f"[server] loading transcriber from {STATE['stt_dir']} ...")
        t0 = time.time()
        STATE["stt"] = WhisperModel(STATE["stt_dir"], device=STATE["device"],
                                    compute_type=STATE["compute_type"])
        print(f"[server] transcriber ready in {time.time() - t0:.1f}s")
    return STATE["stt"]


def _to_wav(src: Path) -> Path:
    """Browser mic recordings arrive as webm/ogg. ffmpeg-decode to 16k mono wav so the
    decoder never has to guess at a container it may not support."""
    dst = src.with_suffix(".conv.wav")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", str(dst)],
        check=True,
    )
    return dst


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (HERE / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "cuda": torch.cuda.is_available(),
            "stt": {"dir": STATE["stt_dir"], "loaded": "stt" in STATE},
            "tts": {"dir": STATE["tts_dir"], "loaded": "tts" in STATE}}


@app.post("/transcribe")
async def transcribe_endpoint(
    audio: UploadFile = File(...),
    codeswitch: str = Form("false"),
    beam_size: int = Form(5),
):
    want_codeswitch = str(codeswitch).lower() in ("true", "1", "on", "yes")
    tmpdir = Path(tempfile.mkdtemp(prefix="urdu_stt_"))
    try:
        raw = tmpdir / (audio.filename or "upload.bin")
        with open(raw, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        try:
            wav = _to_wav(raw)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return JSONResponse({"error": "Could not decode that audio file."}, status_code=400)

        model = get_stt()
        t0 = time.time()
        if want_codeswitch:
            # Chunked, else the prompt only conditions the first 30s window and has no
            # effect on anything longer -- see scripts/transcribe.py.
            segs, duration = _chunked(model, str(wav), "ur", beam_size, False,
                                      CODESWITCH_PROMPT, 60)
        else:
            out, info = model.transcribe(str(wav), language="ur", beam_size=beam_size)
            segs = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in out]
            duration = float(getattr(info, "duration", 0.0) or 0.0)
        elapsed = time.time() - t0

        if want_codeswitch:
            cs = CodeSwitcher()
            for s in segs:
                s["text"] = cs.apply(s["text"])

        return {
            "text": " ".join(s["text"] for s in segs).strip(),
            "segments": segs,
            "srt": to_srt(segs),
            "duration_s": round(duration, 1),
            "elapsed_s": round(elapsed, 1),
            "speed": round(duration / elapsed, 1) if elapsed else None,
            "n_segments": len(segs),
            "codeswitch": want_codeswitch,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.post("/synthesize")
def synthesize(req: SpeakRequest):
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
    p.add_argument("--tts_dir", default=TTS_MODEL)
    p.add_argument("--stt_dir", default=STT_MODEL)
    p.add_argument("--host", default="127.0.0.1")
    # 8080 is the STT app and 8090 is the agentic_qa dashboard, so default clear of
    # both. 8000 is often unbindable on Windows too: Hyper-V/WSL reserve port blocks
    # and binding fails as a permission error rather than "in use".
    p.add_argument("--port", type=int, default=8095)
    p.add_argument("--device", default="cuda")
    p.add_argument("--compute_type", default="float16")
    p.add_argument("--preload", action="store_true", help="Load both models now rather than on first use")
    a = p.parse_args()

    STATE.update(tts_dir=a.tts_dir, stt_dir=a.stt_dir, device=a.device, compute_type=a.compute_type)
    for label, path in [("voice", a.tts_dir), ("transcriber", a.stt_dir)]:
        if not Path(path).exists():
            sys.exit(f"No {label} model at {path}")
    if a.preload:
        get_tts()
        get_stt()

    port = a.port
    for candidate in [a.port, 8095, 7861, 8766, 5001, 9001]:
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

    print(f"\n  Urdu Speech (transcribe + speak) running at  http://{a.host}:{port}\n")
    uvicorn.run(app, host=a.host, port=port, log_level="warning")

"""Local web UI for the Urdu STT model: upload a file or record from the mic.

Runs on this machine because the model and GPU do -- a hosted page could not reach
either. Nothing leaves the machine; uploads go to a temp file, get transcribed, and are
deleted.

The model is loaded once at startup and held in memory. Loading it per request would add
~10s to every transcription, and holding it costs ~2GB of VRAM that is otherwise idle.

Usage:
    python webapp/server.py
    python webapp/server.py --host 0.0.0.0 --port 8000   # reachable from the LAN
"""
import argparse
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from faster_whisper import WhisperModel

from scripts.transcribe import CODESWITCH_PROMPT, DEFAULT_MODEL, _chunked, to_srt
from src.postprocess.codeswitch import CodeSwitcher

app = FastAPI(title="Urdu STT")
STATE: dict = {}
HERE = Path(__file__).resolve().parent


def get_model() -> WhisperModel:
    if "model" not in STATE:
        print(f"[server] loading model from {STATE['model_dir']} ...")
        t0 = time.time()
        STATE["model"] = WhisperModel(STATE["model_dir"], device=STATE["device"],
                                      compute_type=STATE["compute_type"])
        print(f"[server] model ready in {time.time() - t0:.1f}s")
    return STATE["model"]


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
    return {"ok": True, "model_dir": STATE["model_dir"], "loaded": "model" in STATE}


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
        except subprocess.CalledProcessError:
            return JSONResponse({"error": "Could not decode that audio file."}, status_code=400)

        model = get_model()
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

        text = " ".join(s["text"] for s in segs).strip()
        return {
            "text": text,
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


if __name__ == "__main__":
    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default=DEFAULT_MODEL)
    p.add_argument("--host", default="127.0.0.1")
    # 8000 is frequently unbindable on Windows -- Hyper-V/WSL reserve blocks of ports and
    # binding one fails with WinError 10013 (permission denied) rather than "in use",
    # which reads like a firewall problem and is easy to misdiagnose.
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--device", default="cuda")
    p.add_argument("--compute_type", default="float16")
    p.add_argument("--preload", action="store_true", help="Load the model now rather than on first request")
    a = p.parse_args()

    STATE.update(model_dir=a.model_dir, device=a.device, compute_type=a.compute_type)
    if not Path(a.model_dir).exists():
        sys.exit(f"No model at {a.model_dir} -- run scripts/run_day5_export.py first")
    if a.preload:
        get_model()

    port = a.port
    for candidate in [a.port, 8080, 7860, 8765, 5000, 9000]:
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

    print(f"\n  Urdu STT running at  http://{a.host}:{port}\n")
    uvicorn.run(app, host=a.host, port=port, log_level="warning")

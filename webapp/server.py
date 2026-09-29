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
import atexit
import base64
import io
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, r"C:\Users\Sistems-AI\stt-v2-finetune")

from synthesize import MODEL as TTS_MODEL, PAUSE_S, VOICES, split_sentences
from text_normalize import describe, has_sayable, normalize_for_tts
from scripts.transcribe import CODESWITCH_PROMPT, DEFAULT_MODEL as STT_MODEL, _chunked, to_srt
from src.postprocess.codeswitch import CodeSwitcher

app = FastAPI(title="Urdu Speech")
STATE: dict = {}
HERE = Path(__file__).resolve().parent
MAX_CHARS = 5000
BASE_WHISPER = "large-v3"
LANGUAGES = {"ur": "Urdu", "ar": "Arabic"}


class SpeakRequest(BaseModel):
    text: str
    split: bool = True
    normalize: bool = True
    speaking_rate: float | None = None
    seed: int = 1234
    language: str = "ur"
    engine: str = "meta"  # "meta" (default MMS voices) or "omnivoice" (extra voice)


# Playback settings measured per voice, by synthesizing held-out sentences and scoring
# what a recognizer hears back. Urdu's are already baked into its exported config
# (0.33/0.5/0.9, 0.132 -> 0.085). Arabic's are applied here: 0.196 -> 0.107.
ARABIC_PLAYBACK = {"noise_scale": 0.5, "noise_scale_duration": 0.5, "speaking_rate": 0.9}

# Extra voice: OmniVoice won the blind listening test for both Urdu and Arabic, and a
# recognizer heard every word of its test sentences (WER 0.0 vs Meta's 5.8 Urdu / 13.2
# Arabic). It needs Python 3.11 and its own packages, so it runs as a local helper
# process in its own venv (candidates/omnivoice_worker.py), started on first use. Meta
# stays the default. OmniVoice's weights are CC-BY-NC: demos yes, commercial use no.
OMNIVOICE_DIR = HERE.parent / "candidates"
OMNIVOICE_PY = OMNIVOICE_DIR / "venv_omnivoice" / "Scripts" / "python.exe"
OMNIVOICE_URL = "http://127.0.0.1:8096"
ENGINES = ("meta", "omnivoice")
# The page's speed menu is in Meta's units, where 0.9 is "Normal".
META_NORMAL_RATE = 0.9


def _omnivoice_up() -> bool:
    try:
        with urllib.request.urlopen(OMNIVOICE_URL + "/health", timeout=2) as r:
            return r.status == 200
    except OSError:
        return False


def ensure_omnivoice(timeout_s: float = 180) -> None:
    """Start the OmniVoice helper if it isn't running, and wait until it's ready."""
    if _omnivoice_up():
        return
    proc = STATE.get("omnivoice_proc")
    if proc is None or proc.poll() is not None:
        print("[server] starting OmniVoice helper ...")
        log = open(OMNIVOICE_DIR / "omnivoice_worker.log", "a", encoding="utf-8")
        STATE["omnivoice_proc"] = subprocess.Popen(
            [str(OMNIVOICE_PY), "omnivoice_worker.py", "--port", "8096"],
            cwd=OMNIVOICE_DIR, stdout=log, stderr=subprocess.STDOUT)
        # Don't leave it holding GPU memory after the app exits.
        atexit.register(STATE["omnivoice_proc"].terminate)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _omnivoice_up():
            print("[server] OmniVoice helper ready")
            return
        if STATE["omnivoice_proc"].poll() is not None:
            break
        time.sleep(1)
    raise RuntimeError("The OmniVoice voice didn't start; see candidates/omnivoice_worker.log.")


def omnivoice_say(text: str, language: str, speed: float) -> tuple[np.ndarray, int]:
    body = json.dumps({"text": text, "language": language, "speed": speed}).encode()
    req = urllib.request.Request(OMNIVOICE_URL + "/speak", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        audio, sr = sf.read(io.BytesIO(r.read()), dtype="float32")
    return audio, sr


def get_tts(language: str = "ur"):
    """Urdu uses the locally exported voice (tuned playback baked into its config);
    Arabic uses facebook/mms-tts-ara. One pipeline per language, loaded on first use."""
    key = f"tts_{language}"
    if key not in STATE:
        from transformers import pipeline

        source = STATE["tts_dir"] if language == "ur" else VOICES[language]
        print(f"[server] loading {language} voice from {source} ...")
        t0 = time.time()
        tts = pipeline("text-to-speech", source, device=0 if torch.cuda.is_available() else -1)
        if language == "ar":
            for name, value in ARABIC_PLAYBACK.items():
                setattr(tts.model, name, value)
        STATE[key] = tts
        print(f"[server] {language} voice ready in {time.time() - t0:.1f}s")
    return STATE[key]


def get_stt(language: str = "ur"):
    """Urdu uses the fine-tune (18.45% WER, vs 22.77% for base large-v3 on the same
    test set); Arabic has no fine-tune, so it uses base large-v3, which is already
    cached. Each is loaded once, on first use of that language."""
    key = "stt_ur" if language == "ur" else "stt_base"
    if key not in STATE:
        from faster_whisper import WhisperModel

        source = STATE["stt_dir"] if key == "stt_ur" else BASE_WHISPER
        print(f"[server] loading {language} transcriber from {source} ...")
        t0 = time.time()
        STATE[key] = WhisperModel(source, device=STATE["device"], compute_type=STATE["compute_type"])
        print(f"[server] {language} transcriber ready in {time.time() - t0:.1f}s")
    return STATE[key]


def _to_wav(src: Path) -> Path:
    """Browser mic recordings arrive as webm/ogg. ffmpeg-decode to 16k mono wav so the
    decoder never has to guess at a container it may not support."""
    dst = src.with_suffix(".conv.wav")
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", str(dst)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or f"ffmpeg exited {r.returncode}")
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
    language: str = Form("ur"),
    vad: str = Form("true"),
):
    if language not in LANGUAGES:
        return JSONResponse({"error": f"Unsupported language {language!r}."}, status_code=400)
    # The code-switch prompt and dictionary are Urdu vocabulary; they'd be meaningless
    # or harmful applied to Arabic.
    want_codeswitch = language == "ur" and str(codeswitch).lower() in ("true", "1", "on", "yes")
    # VAD on by default: measured 18.23% vs 18.45% word error over the 336-clip Urdu
    # test set (25 fewer errors in 11,644 words; better in 93% of bootstrap resamples).
    # Small but free, and dropping non-speech should help more on real recordings than
    # on these clean test clips. Toggleable because aggressive trimming can clip quiet
    # speech.
    want_vad = str(vad).lower() in ("true", "1", "on", "yes")
    tmpdir = Path(tempfile.mkdtemp(prefix="urdu_stt_"))
    try:
        raw = tmpdir / (audio.filename or "upload.bin")
        with open(raw, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        try:
            wav = _to_wav(raw)
        except FileNotFoundError:
            print("[server] ffmpeg not found on PATH -- install it (winget install Gyan.FFmpeg) "
                  "and restart this server so it picks up the updated PATH")
            return JSONResponse(
                {"error": "Could not decode that audio file (ffmpeg is not available on the server)."},
                status_code=400)
        except RuntimeError as e:
            print(f"[server] ffmpeg could not decode {audio.filename!r}: {e}")
            return JSONResponse({"error": "Could not decode that audio file."}, status_code=400)

        model = get_stt(language)
        t0 = time.time()
        if want_codeswitch:
            # Chunked, else the prompt only conditions the first 30s window and has no
            # effect on anything longer -- see scripts/transcribe.py.
            segs, duration = _chunked(model, str(wav), language, beam_size, want_vad,
                                      CODESWITCH_PROMPT, 60)
        else:
            out, info = model.transcribe(str(wav), language=language, beam_size=beam_size,
                                         vad_filter=want_vad)
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
            "language": language,
            "vad": want_vad,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.post("/synthesize")
def synthesize(req: SpeakRequest):
    if req.language not in VOICES:
        return JSONResponse({"error": f"No voice for language {req.language!r}."}, status_code=400)
    if req.engine not in ENGINES:
        return JSONResponse({"error": f"Unknown voice {req.engine!r}."}, status_code=400)
    text = (req.text or "").strip()
    if not text:
        return JSONResponse({"error": "Type some text first."}, status_code=400)
    if len(text) > MAX_CHARS:
        return JSONResponse({"error": f"Text is too long ({len(text)} characters, limit {MAX_CHARS})."},
                            status_code=400)

    # Digits and English words are unsayable as written -- "1500" came out as "5000",
    # "podcast" vanished. Spelling them out halved the error on such sentences
    # (0.385 -> 0.167), and whatever still can't be said is reported back so the page
    # can warn instead of quietly dropping words.
    note = ""
    if req.normalize:
        text, report = normalize_for_tts(text, language=req.language)
        note = describe(report)

    # Sentence-splitting matters: on a 7-sentence paragraph it halved the word error
    # against feeding the whole thing in at once (0.067 vs 0.135).
    chunks = (split_sentences(text, language=req.language) if req.split
              else [c for c in [text] if has_sayable(c, language=req.language)])
    if not chunks:
        script = "Urdu" if req.language == "ur" else "Arabic"
        return JSONResponse({"error": f"Nothing here the voice can say -- it speaks {script} script only."},
                            status_code=400)

    t0 = time.time()
    pieces, sr = [], 16000
    if req.engine == "omnivoice":
        try:
            ensure_omnivoice()
            speed = (req.speaking_rate or META_NORMAL_RATE) / META_NORMAL_RATE
            for i, chunk in enumerate(chunks):
                audio, sr = omnivoice_say(chunk, req.language, speed)
                pieces.append(np.asarray(audio, dtype=np.float32).squeeze())
                if i < len(chunks) - 1:
                    pieces.append(np.zeros(int(PAUSE_S * sr), dtype=np.float32))
        except (RuntimeError, OSError) as e:
            print(f"[server] OmniVoice failed: {e}")
            return JSONResponse({"error": f"The OmniVoice voice failed: {e}"}, status_code=500)
    else:
        tts = get_tts(req.language)
        original_rate = tts.model.speaking_rate
        try:
            if req.speaking_rate is not None:
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
        "spoken_text": text,
        "note": note,
        "language": req.language,
        "engine": req.engine,
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

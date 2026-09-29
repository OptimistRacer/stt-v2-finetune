"""OmniVoice as a small local service, so the web app (main Python 3.14 environment)
can use it without installing OmniVoice there. OmniVoice lives in its own Python 3.11
venv; installs have broken the main environment before.

webapp/server.py starts this on first use of the OmniVoice voice:

    venv_omnivoice\\Scripts\\python.exe omnivoice_worker.py --port 8096

POST /speak  {"text": ..., "language": "ur"|"ar", "speed": 1.0}  -> audio/wav (24 kHz)
GET  /health -> {"ok": true}

Listens on 127.0.0.1 only. The voice is locked per language to the sample the user
picked in the blind listening test (out/omnivoice/<lang>_0.wav, OmniVoice's own
synthetic "auto" voice, not a real person's), so every sentence and every request
sounds like the same speaker.
"""
import argparse
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

import numpy as np
import soundfile as sf
import torch
from omnivoice import OmniVoice

HERE = Path(__file__).parent
SENTENCES = json.loads((HERE / "sentences.json").read_text(encoding="utf-8"))
SAMPLE_RATE = 24000
LANGS = ("ur", "ar")

MODEL = None
PROMPTS = {}
GPU_LOCK = Lock()  # one generation at a time on the GPU


def voice_prompt(lang):
    """Locked voice for a language: the listening-test sample, regenerated the same way
    (auto mode, seed 0) if the file is missing."""
    if lang not in PROMPTS:
        ref_text = SENTENCES[lang][0]
        ref_path = HERE / "out" / "omnivoice" / f"{lang}_0.wav"
        if ref_path.exists():
            wav, sr = sf.read(ref_path, dtype="float32")
        else:
            torch.manual_seed(0)
            wav, sr = MODEL.generate(text=ref_text, language=lang)[0], SAMPLE_RATE
        PROMPTS[lang] = MODEL.create_voice_clone_prompt(ref_audio=(torch.from_numpy(wav), sr),
                                                        ref_text=ref_text)
    return PROMPTS[lang]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, b'{"ok": true}', "application/json")
        else:
            self._send(404, b"", "text/plain")

    def do_POST(self):
        if self.path != "/speak":
            return self._send(404, b"", "text/plain")
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            lang = req.get("language", "ur")
            if lang not in LANGS:
                raise ValueError(f"unsupported language {lang!r}")
            with GPU_LOCK:
                audio = MODEL.generate(text=req["text"], language=lang,
                                       voice_clone_prompt=voice_prompt(lang),
                                       speed=float(req.get("speed") or 1.0))[0]
            buf = io.BytesIO()
            sf.write(buf, np.asarray(audio, dtype=np.float32), SAMPLE_RATE, format="WAV")
            self._send(200, buf.getvalue(), "audio/wav")
        except Exception as e:  # report to the app instead of dropping the connection
            self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")


def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8096)
    args = ap.parse_args()
    MODEL = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
    for lang in LANGS:
        voice_prompt(lang)
    print(f"[omnivoice] ready on 127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

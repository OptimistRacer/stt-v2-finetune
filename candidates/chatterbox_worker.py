"""Our Arabic voice as a small local service, so the web app (main Python 3.14
environment) can use it without installing Chatterbox there. It runs in venv_chatterbox
(Python 3.11), same arrangement as the OmniVoice and Parvaaz helpers.

webapp/server.py starts this on first use of the Arabic voice:

    venv_chatterbox\\Scripts\\python.exe chatterbox_worker.py --port 8098

POST /speak  {"text": ..., "speed": 0.9, "seed": 0}  -> audio/wav (24 kHz)
GET  /health -> {"ok": true}

Listens on 127.0.0.1 only. The voice and settings are the round-3 blind-test winner
(4.50 / 5, ../arabic_voice/voice.json): Chatterbox multilingual cloning a synthetic
reference voice designed with VoxCPM2, exaggeration 0.5, cfg_weight 0.3, temperature 0.6.
Chatterbox has no speed setting. Varying cfg_weight was tried for the app's speed menu and
did not change the pace reliably (0.2 / 0.3 / 0.45 gave 4.8 / 5.4 / 4.8 s for one sentence),
so this voice always uses its tested setting and the menu applies to the Urdu voice only.
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
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

VOICE_DIR = Path(__file__).resolve().parent.parent / "arabic_voice"
SETTINGS = json.loads((VOICE_DIR / "voice.json").read_text(encoding="utf-8"))["settings"]
REFERENCE = str(VOICE_DIR / "reference.wav")
PEAK = 0.9

MODEL = None
GPU_LOCK = Lock()  # one generation at a time on the GPU


def speak(text: str, speed: float, seed: int) -> np.ndarray:
    with GPU_LOCK, torch.no_grad():
        torch.manual_seed(seed)
        wav = MODEL.generate(text, language_id=SETTINGS["language_id"], audio_prompt_path=REFERENCE,
                             exaggeration=SETTINGS["exaggeration"], cfg_weight=SETTINGS["cfg_weight"],
                             temperature=SETTINGS["temperature"])
    x = wav.squeeze(0).cpu().numpy().astype(np.float32)
    return x * min(1.0, PEAK / (float(np.abs(x).max()) or 1.0))


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
            audio = speak(req["text"], float(req.get("speed") or 0.9), int(req.get("seed", 0)))
            buf = io.BytesIO()
            sf.write(buf, audio, MODEL.sr, format="WAV")
            self._send(200, buf.getvalue(), "audio/wav")
        except Exception as e:  # report to the app instead of dropping the connection
            self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")


def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8098)
    args = ap.parse_args()
    MODEL = ChatterboxMultilingualTTS.from_pretrained(device="cuda", t3_model="v3")
    speak("مرحبا", 0.9, 0)  # warm up so the first real request isn't the slow one
    print(f"[chatterbox] ready on 127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

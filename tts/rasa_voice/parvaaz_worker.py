"""The Parvaaz voice as a small local service, so the web app (main Python 3.14 environment)
can use it without installing Parler-TTS there. It runs in candidates/venv_parler
(Python 3.11, transformers 4.46.1), same arrangement as the OmniVoice helper.

webapp/server.py starts this on first use of the Parvaaz voice:

    ..\\candidates\\venv_parler\\Scripts\\python.exe parvaaz_worker.py --port 8097

POST /speak  {"text": ..., "speed": 0.9, "tone": "conversational", "seed": 0}  -> audio/wav (44.1 kHz)
GET  /health -> {"ok": true}

Listens on 127.0.0.1 only. Parler-TTS has no speed knob; speed is asked for in the
description, in the words the voice was trained on (rasa_voice/describe_rasa.py). The
app's speed menu is in Meta's units (0.8 slower / 0.9 normal / 1.0 faster). A fixed seed
per request keeps repeated requests sounding the same.
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
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

# Default v1 (parvaaz_voice); the app passes --model_dir to pick another export (e.g. v2).
MODEL_DIR = Path(__file__).resolve().parent.parent / "parvaaz_voice"
VOICE = None
TONES = {"conversational": "in a natural, conversational tone", "reading": "in a calm, informative reading voice",
         "news": "like a news presenter", "story": "like a storyteller narrating a book",
         "happy": "in a happy, cheerful tone", "sad": "in a sad, sorrowful tone"}

MODEL = TOK = DESC_TOK = None
GPU_LOCK = Lock()  # one generation at a time on the GPU


def rate_words(speed: float) -> str:
    if speed <= 0.85:
        return "slightly slowly"
    if speed >= 0.95:
        return "slightly fast"
    return "at a moderate speed"


def speak(text: str, speed: float, tone: str, seed: int) -> np.ndarray:
    desc = VOICE["description_template"].format(tone=TONES.get(tone, TONES["conversational"]), rate=rate_words(speed))
    d = DESC_TOK(desc, return_tensors="pt").to("cuda:0")
    p = TOK(text, return_tensors="pt").to("cuda:0")
    with GPU_LOCK, torch.no_grad():
        torch.manual_seed(seed)
        wav = MODEL.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                             prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
    return wav.cpu().numpy().squeeze().astype(np.float32)


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
            audio = speak(req["text"], float(req.get("speed") or 0.9), req.get("tone", "conversational"),
                          int(req.get("seed", 0)))
            buf = io.BytesIO()
            sf.write(buf, audio, MODEL.config.sampling_rate, format="WAV")
            self._send(200, buf.getvalue(), "audio/wav")
        except Exception as e:  # report to the app instead of dropping the connection
            self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")


def main():
    global MODEL, TOK, DESC_TOK, MODEL_DIR, VOICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--model_dir", default=str(MODEL_DIR))
    args = ap.parse_args()
    MODEL_DIR = Path(args.model_dir)
    VOICE = json.loads((MODEL_DIR / "voice.json").read_text(encoding="utf-8"))
    MODEL = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_DIR).to("cuda:0").eval()
    TOK = AutoTokenizer.from_pretrained(MODEL_DIR)
    DESC_TOK = AutoTokenizer.from_pretrained(MODEL.config.text_encoder._name_or_path)
    speak("سلام", 0.9, "conversational", 0)  # warm up so the first real request isn't the slow one
    print(f"[parvaaz] ready on 127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

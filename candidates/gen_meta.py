"""The current voices -- exactly as the app serves them -- reading the same test
sentences, so the candidates are compared against what users hear today.

Urdu: urdu_tts_final (tuned playback baked into its config). Arabic: facebook/mms-tts-ara
with webapp/server.py's ARABIC_PLAYBACK. Sentence-by-sentence input, as the app does.

    python gen_meta.py        (main Python environment)
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import pipeline

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from synthesize import VOICES  # noqa: E402

SENTENCES = json.loads((HERE / "sentences.json").read_text(encoding="utf-8"))
ARABIC_PLAYBACK = {"noise_scale": 0.5, "noise_scale_duration": 0.5, "speaking_rate": 0.9}


def main():
    out = HERE / "out" / "meta"
    out.mkdir(parents=True, exist_ok=True)
    for lang in ("ur", "ar"):
        tts = pipeline("text-to-speech", VOICES[lang], device=0)
        if lang == "ar":
            for name, value in ARABIC_PLAYBACK.items():
                setattr(tts.model, name, value)
        for i, text in enumerate(SENTENCES[lang]):
            torch.manual_seed(0)
            r = tts(text)
            sf.write(out / f"{lang}_{i}.wav", np.asarray(r["audio"]).squeeze(), r["sampling_rate"])
            print(f"[meta] {lang}_{i}", flush=True)
    print("[meta] done", flush=True)


if __name__ == "__main__":
    main()

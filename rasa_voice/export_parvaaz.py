"""Export a Parvaaz checkpoint as a clean, standalone model folder for serving.

A training checkpoint (parvaaz_train/checkpoint-N) is ~8 GB because it also holds the
optimizer state; a served model needs only the weights, config, tokenizer and generation
config. Writes ../parvaaz_voice/ (safetensors), plus ATTRIBUTION.md (CC-BY-4.0 requires
crediting Rasa wherever the voice is used) and voice.json (which checkpoint, and the
description wording the voice was trained on).

    ..\\candidates\\venv_parler\\Scripts\\python.exe export_parvaaz.py --step 300
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "parvaaz_voice"

ATTRIBUTION = """# Parvaaz voice -- attribution and licences

This text-to-speech voice is a fine-tune of **Indic Parler-TTS** (`ai4bharat/indic-parler-tts`,
Apache License 2.0) on {data} of the **Rasa** dataset by **AI4Bharat**
(`ai4bharat/Rasa`, Creative Commons Attribution 4.0, CC BY 4.0). The voice is Rasa's male
Urdu speaker.

CC BY 4.0 allows commercial use provided the source is credited. Suggested credit wherever
the voice is used (app "about" page, product documentation):

> Voice trained on the Rasa dataset by AI4Bharat (CC BY 4.0), fine-tuned from
> Indic Parler-TTS by AI4Bharat and Hugging Face (Apache 2.0).

Training code: Parler-TTS (Apache 2.0) and DataSpeech (MIT) by Hugging Face.
Have the exact wording, and any voice-likeness questions, confirmed before a commercial launch.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=300)
    ap.add_argument("--run", choices=["v1", "v2"], default="v1",
                    help="v1: male-only training (parvaaz_train); v2: male + female (parvaaz_v2_train)")
    a = ap.parse_args()
    global OUT
    if a.run == "v2":
        OUT = HERE.parent / "parvaaz_voice_v2"  # next to v1, so the app keeps v1 until switched
    src = HERE / "eval_models" / (f"v2_step{a.step}" if a.run == "v2" else f"step{a.step}")
    run_dir = "parvaaz_v2_train" if a.run == "v2" else "parvaaz_train"
    data = ("the Urdu male and female speakers" if a.run == "v2" else "the Urdu male speaker")
    if not (src / "pytorch_model.bin").exists():
        raise SystemExit(f"No checkpoint at {src} -- run make_eval_models.ps1 first")

    model = ParlerTTSForConditionalGeneration.from_pretrained(src, torch_dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(src)
    if OUT.exists():
        shutil.rmtree(OUT)
    model.save_pretrained(OUT, safe_serialization=True)
    tok.save_pretrained(OUT)
    shutil.copy(src / "preprocessor_config.json", OUT / "preprocessor_config.json")
    (OUT / "ATTRIBUTION.md").write_text(ATTRIBUTION.format(data=data), encoding="utf-8")
    (OUT / "voice.json").write_text(json.dumps({
        "name": "Parvaaz",
        "checkpoint": f"rasa_voice/{run_dir}/checkpoint-{a.step}",
        "language": "ur",
        "description_template": ("Parvaaz speaks {tone}, {rate}, with a moderate-pitched voice that is slightly "
                                 "expressive and animated. The recording is clean and very close-sounding."),
        "note": ("Trained on Urdu script only. The app spells out numbers and maps known English words to "
                 "Urdu script first; unknown English words are read as written (correct in the one case checked)."),
    }, indent=1), encoding="utf-8")
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e9
    print(f"[export] step {a.step} -> {OUT} ({size:.2f} GB)")


if __name__ == "__main__":
    main()

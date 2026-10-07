"""Speak 20 held-out Rasa sentences (Rasa's test split -- never trained on) with one
Parvaaz checkpoint, each described in its own style, so a checkpoint is judged on new
text and on whether the style words actually change the delivery.

    ..\\candidates\\venv_parler\\Scripts\\python.exe gen_heldout.py step300 eval_models\\step300

Writes eval_out/<tag>/<id>.wav and eval_out/heldout_texts.json (the shared text list).
Gentle on the GPU: rests after a sentence while the card is above 65 C.
"""
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
PICK = {"CONV": 6, "WIKI": 4, "NEWS": 3, "BOOK": 3, "HAPPY": 2, "SAD": 2}
STYLE = {"CONV": "in a natural, conversational tone", "WIKI": "in a calm, informative reading voice",
         "NEWS": "like a news presenter", "BOOK": "like a storyteller narrating a book",
         "HAPPY": "in a happy, cheerful tone", "SAD": "in a sad, sorrowful tone"}
DESC = ("Parvaaz speaks {style}, at a moderate speed, with a moderate-pitched voice that is slightly expressive "
        "and animated. The recording is clean and very close-sounding.")


def gpu_temp():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def texts():
    path = HERE / "eval_out" / "heldout_texts.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    test = load_dataset(str(HERE / "rasa_male_hf"), split="test").remove_columns("audio")
    rng = random.Random(11)
    chosen = []
    for style, n in PICK.items():
        pool = [r for r in test if r["style"] == style and 3.0 <= r["duration"] <= 10.0]
        chosen += [{"id": r["id"], "style": style, "text": r["text"]} for r in rng.sample(pool, n)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chosen, ensure_ascii=False, indent=1), encoding="utf-8")
    return chosen


def main():
    tag, model_dir = sys.argv[1], sys.argv[2]
    out = HERE / "eval_out" / tag
    out.mkdir(parents=True, exist_ok=True)
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_dir).to("cuda:0").eval()
    tok = AutoTokenizer.from_pretrained(model_dir)
    desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    for item in texts():
        wav_path = out / f"{item['id']}.wav"
        if wav_path.exists():
            continue
        torch.manual_seed(int(os.environ.get("TTS_SEED", 0)))  # re-draw to check rankings
        d = desc_tok(DESC.format(style=STYLE[item["style"]]), return_tensors="pt").to("cuda:0")
        p = tok(item["text"], return_tensors="pt").to("cuda:0")
        t0 = time.time()
        with torch.no_grad():
            wav = model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                 prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
        wav = wav.cpu().numpy().squeeze().astype(np.float32)
        sf.write(wav_path, wav, model.config.sampling_rate)
        print(f"[{tag}] {item['id']} {item['style']}: {len(wav) / model.config.sampling_rate:.1f}s in {time.time() - t0:.1f}s", flush=True)
        start = time.time()
        while (t := gpu_temp()) is not None and t > 65 and time.time() - start < 20:
            time.sleep(1)
    print(f"[{tag}] done", flush=True)


if __name__ == "__main__":
    main()

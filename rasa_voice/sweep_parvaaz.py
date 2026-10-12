"""Settings sweep for the Parvaaz voice: which description wording, sampling temperature
and checkpoint give the clearest pronunciation (and, judged by ear afterwards, the best
sound). The user asked for clearer pronunciation and better voice quality (2026-10-12).

Each config reads the 6 shared test sentences and the 20 held-out Rasa sentences, with two
seeds, into eval_out/sweep/<config>_s<seed>/<id>.wav; sweep_score.py scores them.

    ..\\candidates\\venv_parler\\Scripts\\python.exe sweep_parvaaz.py
"""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
OUT = HERE / "eval_out" / "sweep"
VOICE = ("Parvaaz speaks in a natural, conversational tone, {rate}, with a moderate-pitched voice that is slightly "
         "expressive and animated. {quality}")
QUALITY = {"clean": "The recording is clean and very close-sounding.",
           "vclean": "The recording is very clean and very close-sounding.",
           "hq": "The recording is of very high quality, with the speaker's voice sounding clear and very close up."}
# name: (checkpoint step, quality key, temperature, rate words)
CONFIGS = {
    "c0_base":        (654, "clean", 1.0, "at a moderate speed"),   # what the app does today, on step 654
    "c1_vclean":      (654, "vclean", 1.0, "at a moderate speed"),
    "c2_t07":         (654, "clean", 0.7, "at a moderate speed"),
    "c3_vclean_t07":  (654, "vclean", 0.7, "at a moderate speed"),
    "c4_slow_t07":    (654, "vclean", 0.7, "slightly slowly"),
    "c5_hq_t07":      (654, "hq", 0.7, "at a moderate speed"),
    "c6_s300_t07":    (300, "vclean", 0.7, "at a moderate speed"),
}
SEEDS = (0, 1)


def gpu_temp():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def texts():
    std = json.loads((HERE.parent / "candidates" / "sentences.json").read_text(encoding="utf-8"))["ur"]
    held = json.loads((HERE / "eval_out" / "heldout_texts.json").read_text(encoding="utf-8"))
    return [(f"std{i}", t) for i, t in enumerate(std)] + [(h["id"], h["text"]) for h in held]


def main():
    items = texts()
    models = {}
    for name, (step, quality, temp, rate) in CONFIGS.items():
        if step not in models:
            models.clear()
            torch.cuda.empty_cache()
            src = HERE / "eval_models" / f"step{step}"
            m = ParlerTTSForConditionalGeneration.from_pretrained(src).to("cuda:0").eval()
            models[step] = (m, AutoTokenizer.from_pretrained(src), AutoTokenizer.from_pretrained(m.config.text_encoder._name_or_path))
        model, tok, desc_tok = models[step]
        d = desc_tok(VOICE.format(rate=rate, quality=QUALITY[quality]), return_tensors="pt").to("cuda:0")
        for seed in SEEDS:
            out = OUT / f"{name}_s{seed}"
            out.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            for cid, text in items:
                path = out / f"{cid}.wav"
                if path.exists():
                    continue
                p = tok(text, return_tensors="pt").to("cuda:0")
                torch.manual_seed(seed)
                with torch.no_grad():
                    wav = model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                         prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask,
                                         do_sample=True, temperature=temp)
                sf.write(path, wav.cpu().numpy().squeeze().astype(np.float32), model.config.sampling_rate)
                start = time.time()
                while (t := gpu_temp()) is not None and t > 65 and time.time() - start < 20:
                    time.sleep(1)
            print(f"[sweep] {name} seed {seed}: {len(items)} clips in {time.time() - t0:.0f}s", flush=True)
    print("[sweep] done", flush=True)


if __name__ == "__main__":
    main()

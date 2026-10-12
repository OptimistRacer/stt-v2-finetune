"""Settings sweep for the Chatterbox Arabic voice (commercially usable, MIT): which sampling
temperature, expressiveness (exaggeration), text adherence (cfg_weight) and designed voice
give the clearest pronunciation and best sound. The user preferred Chatterbox over VoxCPM2
and asked for clearer pronunciation and better voice quality (2026-10-12).

Reference voices are the synthetic designed voices from gen_arabic_retry.py
(out/arabic_refs/design_<k>.wav), so no real person's voice is used.

    venv_chatterbox\\Scripts\\python.exe sweep_arabic.py
Writes out/arabic_sweep/<config>_s<seed>/std<i>.wav; score with
    python ..\\rasa_voice\\sweep_score.py --dir out\\arabic_sweep --lang ar
"""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

HERE = Path(__file__).parent
SENTENCES = json.loads((HERE / "sentences.json").read_text(encoding="utf-8"))["ar"]
REFS = HERE / "out" / "arabic_refs"
OUT = HERE / "out" / "arabic_sweep"
# name: (designed voice, exaggeration, cfg_weight, temperature)
CONFIGS = {
    "a0_v1_retry":   (1, 0.5, 0.3, 0.8),   # what the retry blind test used
    "a1_v1_t06":     (1, 0.5, 0.3, 0.6),
    "a2_v1_calm":    (1, 0.3, 0.3, 0.6),
    "a3_v1_cfg05":   (1, 0.5, 0.5, 0.6),
    "a4_v3_t06":     (3, 0.5, 0.3, 0.6),
    "a5_v3_calm05":  (3, 0.3, 0.5, 0.6),
}
SEEDS = (0, 1)
PEAK = 0.9


def gpu_temp():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def main():
    model = ChatterboxMultilingualTTS.from_pretrained(device="cuda", t3_model="v3")
    for name, (voice, exag, cfg, temp) in CONFIGS.items():
        ref = str(REFS / f"design_{voice}.wav")
        for seed in SEEDS:
            out = OUT / f"{name}_s{seed}"
            out.mkdir(parents=True, exist_ok=True)
            for i, text in enumerate(SENTENCES):
                path = out / f"std{i}.wav"
                if path.exists():
                    continue
                torch.manual_seed(seed)
                wav = model.generate(text, language_id="ar", audio_prompt_path=ref, exaggeration=exag,
                                     cfg_weight=cfg, temperature=temp)
                x = wav.squeeze(0).cpu().numpy().astype(np.float32)
                sf.write(path, x * min(1.0, PEAK / (float(np.abs(x).max()) or 1.0)), model.sr)
                start = time.time()
                while (t := gpu_temp()) is not None and t > 65 and time.time() - start < 20:
                    time.sleep(1)
            print(f"[arabic_sweep] {name} seed {seed} done", flush=True)
    print("[arabic_sweep] done", flush=True)


if __name__ == "__main__":
    main()

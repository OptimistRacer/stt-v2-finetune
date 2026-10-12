"""Second try at a commercially usable Arabic voice (2026-10-12).

In the blind test the two commercially licensed Arabic models sounded worst (VoxCPM2 1.0,
Chatterbox 1.83 of 5; Meta 3.0) although both pronounced every word right. The September
samples were each model's random default voice at VoxCPM2's fastest setting (10 steps),
spoke in about half Meta's time, and VoxCPM2 peaked at the clipping point. This retries
with a chosen, consistent male voice, more refinement, a calmer pace and no clipping.

    venv_voxcpm\\Scripts\\python.exe gen_arabic_retry.py voxcpm        # designs 3 voices, reads the 6 sentences in each
    venv_chatterbox\\Scripts\\python.exe gen_arabic_retry.py chatterbox  # reads them with designed voice 1 as reference

VoxCPM2 "voice design" makes a synthetic voice from a written description (no real
person's voice), which is then reused as a reference clip so every sentence has the same
voice. Writes out/voxcpm_v<k>/ar_<i>.wav, out/chatterbox_v1/ar_<i>.wav and
out/arabic_refs/design_<k>.wav (+ .txt transcript).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).parent
SENTENCES = json.loads((HERE / "sentences.json").read_text(encoding="utf-8"))["ar"]
REFS = HERE / "out" / "arabic_refs"
REF_TEXT = "أهلا وسهلا بكم، يسعدنا أن نكون معكم اليوم في هذه الحلقة الجديدة."
DESIGNS = {
    1: "A middle-aged Arab man with a warm, deep, natural voice, speaking calmly at a relaxed pace",
    2: "A young Arab man with a friendly, clear voice, speaking naturally at an unhurried conversational pace",
    3: "A mature Arab male presenter with a rich, steady voice, speaking slowly and clearly",
}
STEPS = 30          # VoxCPM2 refinement steps (card: "higher for better result"); September used 10
PEAK = 0.9          # keep clear of clipping


def gpu_temp():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def cool_down():
    start = time.time()
    while (t := gpu_temp()) is not None and t > 65 and time.time() - start < 20:
        time.sleep(1)


def save(path: Path, wav, sr: int):
    x = np.asarray(wav, dtype=np.float32).squeeze()
    peak = float(np.abs(x).max()) or 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x * min(1.0, PEAK / peak), sr)
    return len(x) / sr


def run_voxcpm():
    from voxcpm import VoxCPM
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    sr = model.tts_model.sample_rate
    for k, design in DESIGNS.items():
        ref = REFS / f"design_{k}.wav"
        if not ref.exists():
            torch.manual_seed(k)
            d = save(ref, model.generate(text=f"({design}){REF_TEXT}", cfg_value=2.0, inference_timesteps=STEPS), sr)
            (REFS / f"design_{k}.txt").write_text(REF_TEXT, encoding="utf-8")
            print(f"[voxcpm] designed voice {k}: {d:.1f}s reference", flush=True)
        for i, text in enumerate(SENTENCES):
            torch.manual_seed(0)
            # "Ultimate cloning": reference clip + its exact transcript keeps the same voice.
            wav = model.generate(text=text, prompt_wav_path=str(ref), prompt_text=REF_TEXT, reference_wav_path=str(ref),
                                 cfg_value=2.0, inference_timesteps=STEPS)
            d = save(HERE / "out" / f"voxcpm_v{k}" / f"ar_{i}.wav", wav, sr)
            print(f"[voxcpm_v{k}] ar_{i}: {d:.1f}s (GPU {gpu_temp()}C)", flush=True)
            cool_down()


def run_chatterbox():
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model = ChatterboxMultilingualTTS.from_pretrained(device="cuda", t3_model="v3")
    # One Chatterbox variant per designed voice (the user preferred Chatterbox's delivery).
    for k in DESIGNS:
        ref = REFS / f"design_{k}.wav"
        if not ref.exists():
            raise SystemExit("Run the voxcpm step first: it creates the reference voices.")
        for i, text in enumerate(SENTENCES):
            out = HERE / "out" / f"chatterbox_v{k}" / f"ar_{i}.wav"
            if out.exists():
                continue
            torch.manual_seed(0)
            # cfg_weight below the 0.5 default slows and steadies the pacing (Chatterbox's own advice).
            wav = model.generate(text, language_id="ar", audio_prompt_path=str(ref), exaggeration=0.5, cfg_weight=0.3)
            d = save(out, wav.squeeze(0).cpu().numpy(), model.sr)
            print(f"[chatterbox_v{k}] ar_{i}: {d:.1f}s (GPU {gpu_temp()}C)", flush=True)
            cool_down()


if __name__ == "__main__":
    {"voxcpm": run_voxcpm, "chatterbox": run_chatterbox}[sys.argv[1]]()
    print(f"[{sys.argv[1]}] done", flush=True)

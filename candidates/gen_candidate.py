"""Read the shared test sentences with one candidate voice model. Run it with that
model's own venv, e.g.

    venv_omnivoice\\Scripts\\python.exe gen_candidate.py omnivoice
    venv_voxcpm\\Scripts\\python.exe gen_candidate.py voxcpm
    venv_chatterbox\\Scripts\\python.exe gen_candidate.py chatterbox
    venv_parler\\Scripts\\python.exe gen_candidate.py parvaaz_step1000 D:/.../checkpoint-1000-epoch-1

The "parvaaz_<tag>" form loads our fine-tuned Indic Parler-TTS (rasa_voice/) from the
folder given as the second argument and writes out/parvaaz_<tag>/.

Writes out/<model>/<lang>_<i>.wav. No reference clip yet, so every model uses its own
built-in or described voice; voice cloning waits for consented reference clips.

Gentle on the GPU, as with training: after each sentence, if the card is above
TARGET_C it rests until it cools (capped), so the fans don't ramp.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).parent
SENTENCES = json.loads((HERE / "sentences.json").read_text(encoding="utf-8"))
TARGET_C = 65
MAX_REST_S = 20
# Which languages each model supports (from its model card).
LANGS = {"omnivoice": ["ur", "ar"], "voxcpm": ["ar"], "chatterbox": ["ar"],
         # Indic Parler-TTS (Apache-2.0): its model card names these two as Urdu voices.
         "parler_divjot": ["ur"], "parler_gurpreet": ["ur"]}
LANG_NAMES = {"ur": "Urdu", "ar": "Arabic"}


def gpu_temperature():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def cool_down():
    torch.cuda.synchronize()
    start = time.time()
    temp = gpu_temperature()
    while temp is not None and temp > TARGET_C and time.time() - start < MAX_REST_S:
        time.sleep(1)
        temp = gpu_temperature()


def load(name):
    if name == "omnivoice":
        from omnivoice import OmniVoice
        model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)

        def say(text, lang):
            # "Auto" mode: the model picks a voice. Fixing the seed keeps it the same
            # voice across sentences.
            torch.manual_seed(0)
            return model.generate(text=text, language=lang)[0], 24000
        return say

    if name == "voxcpm":
        from voxcpm import VoxCPM
        model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)

        def say(text, lang):
            torch.manual_seed(0)
            wav = model.generate(text=text, cfg_value=2.0, inference_timesteps=10)
            return wav, model.tts_model.sample_rate
        return say

    if name == "chatterbox":
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        model = ChatterboxMultilingualTTS.from_pretrained(device="cuda", t3_model="v3")

        def say(text, lang):
            torch.manual_seed(0)
            wav = model.generate(text, language_id=lang)
            return wav.squeeze(0).cpu().numpy(), model.sr
        return say

    if name.startswith("parler_") or name.startswith("parvaaz_"):
        # parler_<speaker>: the untuned base, to hear its Urdu before any fine-tuning; the
        # voice is chosen by naming a speaker in the description, as its model card says.
        # parvaaz_<tag>: our fine-tune, described in the wording it was trained on
        # (rasa_voice/describe_rasa.py) -- conversational, moderate speed, clean.
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        repo = sys.argv[2] if name.startswith("parvaaz_") else "ai4bharat/indic-parler-tts"
        model = ParlerTTSForConditionalGeneration.from_pretrained(repo).to("cuda:0")
        tok = AutoTokenizer.from_pretrained(repo)
        desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
        if name.startswith("parvaaz_"):
            text = ("Parvaaz speaks in a natural, conversational tone, at a moderate speed, with a moderate-pitched "
                    "voice that is slightly expressive and animated. The recording is clean and very close-sounding.")
        else:
            speaker = name.split("_", 1)[1].capitalize()
            text = (f"{speaker} speaks at a moderate pace with a natural, slightly expressive tone. "
                    "The recording is of very high quality, with the speaker's voice sounding clear "
                    "and very close up.")
        desc = desc_tok(text, return_tensors="pt").to("cuda:0")

        def say(text, lang):
            # TTS_SEED re-draws the sampling, to check a ranking isn't one lucky draw.
            torch.manual_seed(int(os.environ.get("TTS_SEED", 0)))
            prompt = tok(text, return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                wav = model.generate(input_ids=desc.input_ids, attention_mask=desc.attention_mask,
                                     prompt_input_ids=prompt.input_ids, prompt_attention_mask=prompt.attention_mask)
            return wav.cpu().numpy().squeeze(), model.config.sampling_rate
        return say

    raise SystemExit(f"unknown model {name}")


def main():
    name = sys.argv[1]
    say = load(name)
    out = HERE / "out" / name
    out.mkdir(parents=True, exist_ok=True)
    for lang in LANGS.get(name, ["ur"]):
        for i, text in enumerate(SENTENCES[lang]):
            t0 = time.time()
            wav, sr = say(text, lang)
            wav = np.asarray(wav, dtype=np.float32).squeeze()
            sf.write(out / f"{lang}_{i}.wav", wav, sr)
            print(f"[{name}] {lang}_{i}: {len(wav) / sr:.1f}s audio in {time.time() - t0:.1f}s "
                  f"(GPU {gpu_temperature()}C)", flush=True)
            cool_down()
    print(f"[{name}] done", flush=True)


if __name__ == "__main__":
    main()

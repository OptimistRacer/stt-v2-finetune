"""Find the best playback settings for a trained VITS voice, without retraining.

VITS deliberately adds randomness when it speaks: `noise_scale` (how much the acoustic
detail varies), `noise_scale_duration` (how much the timing varies) and `speaking_rate`.
The defaults favour expressiveness; turning them down usually trades a little liveliness
for steadier, clearer speech. They're plain attributes on VitsModel, so this is purely
an inference-time knob.

Scores each combination the same way as sweep_checkpoints.py: synthesize the shared
test sentences, run them back through the STT model, and score word error rate, plus
optional voice-match against real recordings of the target speaker.
"""
import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, VitsModel

from sweep_checkpoints import (BASE_MODEL, SENTENCES, VoiceMatcher, merge_weight_norm,
                               transcribe, word_error_rate)

NOISE_SCALES = [0.667, 0.5, 0.33]
NOISE_SCALE_DURATIONS = [0.8, 0.5]
SPEAKING_RATES = [1.0, 0.9]


def load_model(model_path, checkpoint):
    if checkpoint:
        model = VitsModel.from_pretrained(BASE_MODEL)
        missing, _ = model.load_state_dict(
            merge_weight_norm(load_file(os.path.join(checkpoint, "model.safetensors"))), strict=False)
        if missing:
            raise SystemExit(f"checkpoint missing {len(missing)} weights")
        return model
    return VitsModel.from_pretrained(model_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="exported model folder")
    ap.add_argument("--checkpoint", help="checkpoint-N dir (alternative to --model)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_samples", type=int, default=2)
    ap.add_argument("--ref_speaker_dir", default=None)
    args = ap.parse_args()

    if not (args.model or args.checkpoint):
        raise SystemExit("pass --model or --checkpoint")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    matcher = VoiceMatcher(args.ref_speaker_dir) if args.ref_speaker_dir else None
    model = load_model(args.model, args.checkpoint).to(device).eval()

    results = []
    for ns, nsd, rate in itertools.product(NOISE_SCALES, NOISE_SCALE_DURATIONS, SPEAKING_RATES):
        model.noise_scale, model.noise_scale_duration, model.speaking_rate = ns, nsd, rate
        tag = f"ns{ns}_nsd{nsd}_rate{rate}".replace(".", "")
        wers, sims, first = [], [], None
        for si, text in enumerate(SENTENCES):
            inputs = tokenizer(text, return_tensors="pt").to(device)
            ref = text.replace("۔", "").strip()
            for i in range(args.n_samples):
                torch.manual_seed(1234 + i)
                with torch.no_grad():
                    audio = model(**inputs).waveform[0].cpu().numpy()
                p = str(out_dir / f"{tag}_t{si}_s{i}.wav")
                scipy.io.wavfile.write(p, rate=model.config.sampling_rate, data=audio)
                r = transcribe(p, language="ur", device="cuda", compute_type="float16",
                               beam_size=5, vad=False, initial_prompt=None, prompt_chunk_s=0)
                wers.append(word_error_rate(ref, r["text"].replace("۔", "").strip()))
                if matcher:
                    sims.append(matcher.score(p))
                first = first or p
        row = {"noise_scale": ns, "noise_scale_duration": nsd, "speaking_rate": rate,
               "wer": round(float(np.mean(wers)), 3), "example": first}
        if sims:
            row["voice_sim"] = round(float(np.mean(sims)), 3)
        results.append(row)
        print(f"[tune] noise {ns:<5} dur {nsd:<4} rate {rate:<4} -> WER {row['wer']:.3f}"
              + (f"  voice {row['voice_sim']:.3f}" if sims else ""))

    results.sort(key=lambda r: r["wer"])
    with open(out_dir / "playback_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    b = results[0]
    print(f"\n[done] best: noise_scale {b['noise_scale']}, duration {b['noise_scale_duration']}, "
          f"rate {b['speaking_rate']} (WER {b['wer']})")


if __name__ == "__main__":
    main()

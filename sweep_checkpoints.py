"""Sweep every training checkpoint and score it on two axes:

- clarity: synthesize held-out sentences, run them back through our Urdu STT model,
  and score word error rate against the intended text
- voice match (optional, --ref_speaker_dir): speaker-embedding cosine similarity
  between the synthesized audio and real recordings of the target voice

The two can pull against each other (more training = closer voice, but possibly less
clear), so the point is to see both and pick the checkpoint where they balance.

Checkpoints store weight-norm parametrized conv weights (weight_g/weight_v) while
VitsModel expects merged `weight`, so they get merged on load.
"""
import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import soundfile as sf
import torch
from safetensors.torch import load_file
from transformers import VitsModel, AutoTokenizer

sys.path.insert(0, r"C:\Users\Sistems-AI\stt-v2-finetune")
from scripts.transcribe import transcribe

BASE_MODEL = "facebook/mms-tts-urd-script_arabic"

# Everyday conversational Urdu, none of it from any training transcript -- the demo
# speaks this kind of text, not poetry.
SENTENCES = [
    "آج ہم ایک نئے موضوع پر بات کریں گے جو ہمارے سامعین کے لیے بہت اہم ہے۔ "
    "امید ہے آپ کو یہ اپیسوڈ پسند آئے گا۔",
    "ہماری کمپنی نوجوانوں کو نئی مہارتیں سکھانے کے لیے مختلف پروگرام چلاتی ہے۔",
    "کل صبح موسم بہت اچھا تھا اس لیے ہم سب باہر چائے پینے گئے۔",
]


def merge_weight_norm(state):
    """w = g * v / ||v||, matching torch.nn.utils.weight_norm."""
    merged = {}
    skip = set()
    for key in state:
        if key.endswith(".weight_g"):
            base = key[: -len(".weight_g")]
            v_key = base + ".weight_v"
            if v_key in state:
                merged[base + ".weight"] = torch._weight_norm(state[v_key], state[key], 0)
                skip.add(key)
                skip.add(v_key)
    out = {k: v for k, v in state.items() if k not in skip}
    out.update(merged)
    return out


def word_error_rate(ref, hyp):
    r, h = ref.split(), hyp.split()
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(r)][len(h)] / max(len(r), 1)


class VoiceMatcher:
    """Cosine similarity of a clip's speaker embedding to the target voice's centroid."""

    def __init__(self, ref_dir, n_ref=30):
        from huggingface_hub import get_token
        from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

        self.model = PretrainedSpeakerEmbedding(
            "pyannote/wespeaker-voxceleb-resnet34-LM", device=torch.device("cuda"), token=get_token()
        )
        refs = sorted(glob.glob(os.path.join(ref_dir, "*.wav")))
        random.Random(0).shuffle(refs)
        embs = np.stack([self.embed(p) for p in refs[:n_ref]])
        c = embs.mean(axis=0)
        self.centroid = c / np.linalg.norm(c)

    def embed(self, path):
        data, _ = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        e = np.asarray(self.model(torch.from_numpy(data)[None, None, :])).reshape(-1)
        return e / np.linalg.norm(e)

    def score(self, path):
        return float(self.embed(path) @ self.centroid)


def synth_and_score(model, tokenizer, device, out_prefix, n_samples, matcher):
    """VITS has a stochastic duration predictor, so one sample is noisy -- average a few."""
    wers, sims, examples = [], [], []
    for si, text in enumerate(SENTENCES):
        inputs = tokenizer(text, return_tensors="pt").to(device)
        ref = text.replace("۔", "").strip()
        for i in range(n_samples):
            torch.manual_seed(1234 + i)
            with torch.no_grad():
                audio = model(**inputs).waveform[0].cpu().numpy()
            p = f"{out_prefix}_t{si}_s{i}.wav"
            scipy.io.wavfile.write(p, rate=model.config.sampling_rate, data=audio)
            r = transcribe(p, language="ur", device="cuda", compute_type="float16",
                           beam_size=5, vad=False, initial_prompt=None, prompt_chunk_s=0)
            wers.append(word_error_rate(ref, r["text"].replace("۔", "").strip()))
            if matcher:
                sims.append(matcher.score(p))
            if i == 0:
                examples.append({"sentence": si, "wav": p, "stt": r["text"]})
    return wers, sims, examples


def summarize(step, wers, sims, examples, extra=None):
    row = {"step": step, "wer": round(float(np.mean(wers)), 3),
           "wer_min": round(min(wers), 3), "wer_max": round(max(wers), 3),
           "examples": examples}
    if sims:
        row["voice_sim"] = round(float(np.mean(sims)), 3)
    if extra:
        row.update(extra)
    return row


def fmt(row):
    s = f"[sweep] step {row['step']:6d}  WER {row['wer']:.3f}"
    if "voice_sim" in row:
        s += f"  voice {row['voice_sim']:.3f}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="training output_dir containing checkpoint-*")
    ap.add_argument("--out_dir", required=True, help="where to write per-checkpoint wavs + results")
    ap.add_argument("--n_samples", type=int, default=2)
    ap.add_argument("--ref_speaker_dir", default=None,
                    help="dir of real recordings of the target voice, enables the voice-match score")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    matcher = VoiceMatcher(args.ref_speaker_dir) if args.ref_speaker_dir else None

    ckpts = sorted(
        glob.glob(os.path.join(args.run_dir, "checkpoint-*")),
        key=lambda p: int(p.rsplit("-", 1)[1]),
    )
    print(f"[sweep] {len(ckpts)} checkpoints in {args.run_dir}")

    results = []

    # step 0 = the untouched base model, the clarity reference we must not fall far below
    base = VitsModel.from_pretrained(BASE_MODEL).to(device).eval()
    results.append(summarize(0, *synth_and_score(base, tokenizer, device,
                                                 str(out_dir / "step_00000"), args.n_samples, matcher)))
    print(fmt(results[0]) + "  (BASE reference)")
    del base
    torch.cuda.empty_cache()

    for ckpt in ckpts:
        step = int(ckpt.rsplit("-", 1)[1])
        weights = os.path.join(ckpt, "model.safetensors")
        if not os.path.exists(weights):
            continue

        model = VitsModel.from_pretrained(BASE_MODEL)
        missing, _ = model.load_state_dict(merge_weight_norm(load_file(weights)), strict=False)
        model = model.to(device).eval()

        row = summarize(step, *synth_and_score(model, tokenizer, device, str(out_dir / f"step_{step:05d}"),
                                               args.n_samples, matcher),
                        extra={"n_missing_keys": len(missing)})
        results.append(row)
        print(fmt(row))

        del model
        torch.cuda.empty_cache()

    with open(out_dir / "sweep_results.json", "w", encoding="utf-8") as f:
        json.dump({"sentences": SENTENCES, "results": results}, f, ensure_ascii=False, indent=2)
    best = min(results, key=lambda x: x["wer"])
    print(f"\n[done] clearest: step {best['step']} (WER {best['wer']}) -> {out_dir}")


if __name__ == "__main__":
    main()

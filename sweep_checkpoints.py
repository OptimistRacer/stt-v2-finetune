"""Sweep every training checkpoint: synthesize the same held-out sentence with each,
run it back through our Urdu STT model, and score word error rate against the intended
text. This maps the trade-off between "sounds like the host" (more training) and
"is understandable" (less training), so we can pick the checkpoint at the peak
instead of guessing.

Checkpoints store weight-norm parametrized conv weights (weight_g/weight_v) while
VitsModel expects merged `weight`, so they get merged on load.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import scipy.io.wavfile
import torch
from safetensors.torch import load_file
from transformers import VitsModel, AutoTokenizer

sys.path.insert(0, r"C:\Users\Sistems-AI\stt-v2-finetune")
from scripts.transcribe import transcribe

BASE_MODEL = "facebook/mms-tts-urd-script_arabic"
GEN_TEXT = (
    "آج ہم ایک نئے موضوع پر بات کریں گے جو ہمارے سامعین کے لیے بہت اہم ہے۔ "
    "امید ہے آپ کو یہ اپیسوڈ پسند آئے گا۔"
)


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


def synth_and_score(model, tokenizer, device, out_prefix, n_samples):
    """VITS has a stochastic duration predictor, so one sample is noisy -- average a few."""
    inputs = tokenizer(GEN_TEXT, return_tensors="pt").to(device)
    ref = GEN_TEXT.replace("۔", "").strip()
    wers, paths = [], []
    for i in range(n_samples):
        torch.manual_seed(1234 + i)
        with torch.no_grad():
            audio = model(**inputs).waveform[0].cpu().numpy()
        p = f"{out_prefix}_s{i}.wav"
        scipy.io.wavfile.write(p, rate=model.config.sampling_rate, data=audio)
        r = transcribe(p, language="ur", device="cuda", compute_type="float16",
                       beam_size=5, vad=False, initial_prompt=None, prompt_chunk_s=0)
        wers.append(word_error_rate(ref, r["text"].replace("۔", "").strip()))
        paths.append((p, r["text"]))
    return wers, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="training output_dir containing checkpoint-*")
    ap.add_argument("--out_dir", required=True, help="where to write per-checkpoint wavs + results")
    ap.add_argument("--n_samples", type=int, default=3)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    ckpts = sorted(
        glob.glob(os.path.join(args.run_dir, "checkpoint-*")),
        key=lambda p: int(p.rsplit("-", 1)[1]),
    )
    print(f"[sweep] {len(ckpts)} checkpoints in {args.run_dir}")

    results = []

    # step 0 = the untouched base model, the reference we must not fall below
    base = VitsModel.from_pretrained(BASE_MODEL).to(device).eval()
    wers, paths = synth_and_score(base, tokenizer, device, str(out_dir / "step_00000"), args.n_samples)
    results.append({"step": 0, "wer": round(sum(wers) / len(wers), 3),
                    "wer_min": round(min(wers), 3), "wer_max": round(max(wers), 3),
                    "stt": paths[0][1], "wav": paths[0][0]})
    print(f"[sweep] step      0  WER {results[0]['wer']:.3f}  (BASE reference)")
    del base
    torch.cuda.empty_cache()

    for ckpt in ckpts:
        step = int(ckpt.rsplit("-", 1)[1])
        weights = os.path.join(ckpt, "model.safetensors")
        if not os.path.exists(weights):
            continue

        model = VitsModel.from_pretrained(BASE_MODEL)
        state = merge_weight_norm(load_file(weights))
        missing, unexpected = model.load_state_dict(state, strict=False)
        model = model.to(device).eval()

        wers, paths = synth_and_score(model, tokenizer, device,
                                      str(out_dir / f"step_{step:05d}"), args.n_samples)
        mean_wer = sum(wers) / len(wers)
        results.append({"step": step, "wer": round(mean_wer, 3),
                        "wer_min": round(min(wers), 3), "wer_max": round(max(wers), 3),
                        "stt": paths[0][1], "wav": paths[0][0],
                        "n_missing_keys": len(missing)})
        print(f"[sweep] step {step:6d}  WER {mean_wer:.3f}  (min {min(wers):.3f} max {max(wers):.3f})")

        del model
        torch.cuda.empty_cache()

    results.sort(key=lambda x: x["wer"])
    with open(out_dir / "sweep_results.json", "w", encoding="utf-8") as f:
        json.dump({"intended": GEN_TEXT, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n[done] best: step {results[0]['step']} (WER {results[0]['wer']}) -> {out_dir}")


if __name__ == "__main__":
    main()

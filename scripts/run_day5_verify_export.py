"""Day 5: verify the exported CTranslate2 model actually matches the model it came from.

A merge + quantization round-trip can silently change behaviour -- the LoRA deltas fold
into the base weights, then everything is requantized to fp16/int8 -- so the export has
to be scored, not just loaded. This transcribes a test split through faster-whisper
(the serving path) and reports WER/CER against the same references the HF evaluation
used, so the numbers are directly comparable to run_day4_eval.py's.

Usage:
    python scripts/run_day5_verify_export.py --config configs/day3_lora_augmented.yaml
    python scripts/run_day5_verify_export.py --config configs/day3_lora_augmented.yaml --limit 40
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from faster_whisper import WhisperModel

from src.evaluation.transcribe_eval import score
from src.training.dataset import load_manifest


def run(config_path: str, ct2_dir: str | None, split: str, manifest: str | None,
        limit: int | None, compute_type: str, device: str) -> None:
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    train_out = Path(cfg["training"]["output_dir"])
    model_dir = Path(ct2_dir) if ct2_dir else train_out / "export" / "ct2"
    if not model_dir.exists():
        raise FileNotFoundError(f"No CTranslate2 model at {model_dir} -- run run_day5_export.py first")

    records = load_manifest(manifest or cfg["manifest_path"], split)
    if limit:
        records = records[:limit]
    print(f"[verify] {len(records)} clips from split={split}")
    print(f"[verify] model: {model_dir} ({compute_type} on {device})")

    model = WhisperModel(str(model_dir), device=device, compute_type=compute_type)

    results, t0 = [], time.time()
    audio_seconds = 0.0
    for i, rec in enumerate(records, 1):
        segments, _ = model.transcribe(rec["path"], language=cfg["language"][:2])
        hyp = " ".join(s.text.strip() for s in segments)
        results.append({**rec, "hypothesis": hyp.strip()})
        audio_seconds += rec.get("duration_s", 0.0)
        if i % 25 == 0 or i == len(records):
            print(f"  {i}/{len(records)}")
    elapsed = time.time() - t0

    s = score(results)
    rtf = elapsed / audio_seconds if audio_seconds else float("nan")
    print(f"\n[verify] WER {s['wer']}%  CER {s['cer']}%  (n={s['n']})")
    print(f"[verify] {elapsed:.1f}s for {audio_seconds/60:.1f} min of audio "
          f"-- RTF {rtf:.3f} ({1/rtf:.1f}x realtime)" if rtf == rtf else "")

    out = train_out / f"day5_export_verify_{split}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"split": split, "model_dir": str(model_dir), "compute_type": compute_type,
                   "score": s, "elapsed_s": round(elapsed, 1), "audio_s": round(audio_seconds, 1),
                   "rtf": round(rtf, 4), "examples": results[:10]}, f, ensure_ascii=False, indent=2)
    print(f"[done] report -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/day3_lora_augmented.yaml")
    p.add_argument("--ct2_dir", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--manifest", default=None, help="Override the config's manifest_path")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--compute_type", default="float16")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    run(a.config, a.ct2_dir, a.split, a.manifest, a.limit, a.compute_type, a.device)

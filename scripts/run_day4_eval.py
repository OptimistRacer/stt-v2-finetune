"""Day 4: WER/CER evaluation, base whisper-large-v3 vs the Day 3 LoRA adapter.

Fixes the eval bug hit during Day 3 training: `force_language()` fixes the model's
generation_config up front instead of letting Whisper auto-detect language per call,
which is what produced the meaningless 250%+ WER numbers printed during training.

Note: an accent-wise breakdown (also a Day 4 task in the 7-Day Plan) isn't possible on
this dataset -- FLEURS has no accent labels, and the podcast data that was going to
carry accent tags is out of scope for this pass (see README).

Crash-resumable (2026-09-09): a full validation-set run previously crashed this
machine outright partway through (GPU driver fault + unclean reboot -- see README).
Per-clip results now stream to `<out_dir>/day4_raw_<split>_<base|lora>.jsonl` as they
complete; rerunning the same command resumes from whatever's already there instead of
starting over.

Usage:
    python scripts/run_day4_eval.py --config configs/day3_lora.yaml --split validation
    python scripts/run_day4_eval.py --config configs/day3_lora.yaml --split test --limit 20  # smoke test
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from src.evaluation.transcribe_eval import load_base_model, load_lora_model, score, transcribe_records
from src.training.dataset import load_manifest
from src.training.lora_setup import load_processor


def run(config_path: str, split: str, limit: int | None, batch_size: int) -> None:
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    sample_rate = cfg.get("sample_rate", 16000)
    adapter_dir = str(Path(cfg["training"]["output_dir"]) / "final_adapter")
    out_dir = Path(cfg["training"]["output_dir"])

    records = load_manifest(cfg["manifest_path"], split)
    if limit:
        records = records[:limit]
    print(f"[day4] evaluating {len(records)} clips from split={split}, batch_size={batch_size}")

    processor = load_processor(cfg["base_model"], cfg["language"], cfg["task"])

    report = {"split": split, "n": len(records)}

    print("[day4] base whisper-large-v3 (zero-shot)")
    base_model = load_base_model(cfg["base_model"], cfg["language"], cfg["task"])
    base_out = out_dir / f"day4_raw_{split}_base.jsonl"
    base_results = transcribe_records(base_model, processor, records, base_out, sample_rate, batch_size)
    report["base"] = score(base_results)
    print(f"  base: {report['base']}")
    del base_model
    torch.cuda.empty_cache()

    print("[day4] Day 3 LoRA adapter")
    lora_model = load_lora_model(cfg["base_model"], adapter_dir, cfg["language"], cfg["task"])
    lora_out = out_dir / f"day4_raw_{split}_lora.jsonl"
    lora_results = transcribe_records(lora_model, processor, records, lora_out, sample_rate, batch_size)
    report["lora"] = score(lora_results)
    print(f"  lora: {report['lora']}")
    del lora_model
    torch.cuda.empty_cache()

    report_path = out_dir / f"day4_eval_{split}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                **report,
                "base_examples": base_results[:10],
                "lora_examples": lora_results[:10],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[done] report -> {report_path}")
    print(f"[summary] base WER={report['base']['wer']}% CER={report['base']['cer']}%"
          f"  |  lora WER={report['lora']['wer']}% CER={report['lora']['cer']}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/day3_lora.yaml")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--limit", type=int, default=None, help="Smoke test: cap number of clips")
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()
    run(args.config, args.split, args.limit, args.batch_size)

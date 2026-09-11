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


def run(config_path: str, split: str, limit: int | None, batch_size: int, domain: str,
        manifest: str | None = None, tag_suffix: str = "") -> None:
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    if manifest:
        # Evaluate an adapter against a manifest other than the one it trained on --
        # e.g. scoring the FLEURS-only adapter on podcast clips, which only exist in
        # the combined manifest.
        cfg["manifest_path"] = manifest
    sample_rate = cfg.get("sample_rate", 16000)
    adapter_dir = str(Path(cfg["training"]["output_dir"]) / "final_adapter")
    out_dir = Path(cfg["training"]["output_dir"])

    records = load_manifest(cfg["manifest_path"], split)
    if domain == "podcast":
        records = [r for r in records if r["source_id"].startswith("podcast")]
    elif domain == "fleurs":
        records = [r for r in records if not r["source_id"].startswith("podcast")]
    if limit:
        records = records[:limit]
    print(f"[day4] evaluating {len(records)} clips from split={split} domain={domain}, batch_size={batch_size}")

    processor = load_processor(cfg["base_model"], cfg["language"], cfg["task"])

    report = {"split": split, "domain": domain, "n": len(records)}

    print("[day4] base whisper-large-v3 (zero-shot)")
    base_model = load_base_model(cfg["base_model"], cfg["language"], cfg["task"])
    tag = (split if domain == "all" else f"{split}_{domain}") + tag_suffix
    base_out = out_dir / f"day4_raw_{tag}_base.jsonl"
    base_results = transcribe_records(base_model, processor, records, base_out, sample_rate, batch_size)
    report["base"] = score(base_results)
    print(f"  base: {report['base']}")
    del base_model
    torch.cuda.empty_cache()

    print("[day4] Day 3 LoRA adapter")
    lora_model = load_lora_model(cfg["base_model"], adapter_dir, cfg["language"], cfg["task"])
    lora_out = out_dir / f"day4_raw_{tag}_lora.jsonl"
    lora_results = transcribe_records(lora_model, processor, records, lora_out, sample_rate, batch_size)
    report["lora"] = score(lora_results)
    print(f"  lora: {report['lora']}")
    del lora_model
    torch.cuda.empty_cache()

    report_path = out_dir / f"day4_eval_{tag}.json"
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
    parser.add_argument("--domain", default="all", choices=["all", "fleurs", "podcast"],
                        help="Restrict to one corpus. The combined manifest mixes FLEURS and podcast "
                             "clips, so comparing against the FLEURS-only baseline needs --domain fleurs.")
    parser.add_argument("--manifest", default=None,
                        help="Override the config's manifest_path (to score an adapter on another corpus).")
    parser.add_argument("--tag_suffix", default="", help="Suffix for output filenames, to avoid clobbering.")
    args = parser.parse_args()
    run(args.config, args.split, args.limit, args.batch_size, args.domain, args.manifest, args.tag_suffix)

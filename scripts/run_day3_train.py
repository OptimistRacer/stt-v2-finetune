"""Day 3: LoRA fine-tune Whisper large-v3 on the FLEURS Urdu manifest. Needs a local
CUDA GPU -- unlike Day 1, this runs on this machine (RTX 5090) rather than Colab.

Crash-hardened (2026-09-09): this machine crashed outright (GPU driver fault -> unclean
Windows reboot) three times, always inside a batched model.generate() call -- including
once inside this script's own predict_with_generate eval at step 200. Two fixes:
1. predict_with_generate is now forced off unconditionally -- training eval only
   computes loss (teacher-forced, never crashed in any incident), never calls
   generate(). WER/CER evaluation moved entirely to the crash-resumable
   run_day4_eval.py, which also loads models with attn_implementation="eager" to avoid
   the fused-kernel instability implicated in all three crashes (see
   src/training/lora_setup.py's docstring).
2. Training now auto-resumes from the latest checkpoint in output_dir if one exists,
   so a crash during plain training (not yet observed, but cheap insurance) loses at
   most one save_steps interval, not the whole run -- just rerun the same command.

Usage:
    python scripts/run_day3_train.py --config configs/day3_lora.yaml
    python scripts/run_day3_train.py --config configs/day3_lora.yaml --max_train_examples 8 --max_steps 2  # smoke test
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import soundfile as sf
import torch
import yaml
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from src.training.augment import Augmenter
from src.training.dataset import ManifestAudioDataset, WhisperCollator, load_manifest
from src.training.lora_setup import load_lora_model, load_processor

# See src/training/lora_setup.py's module docstring for why.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def run(config_path: str, max_train_examples: int | None, max_steps: int | None) -> None:
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    assert torch.cuda.is_available(), "Day 3 training needs a CUDA GPU"

    processor = load_processor(cfg["base_model"], cfg["language"], cfg["task"])
    model = load_lora_model(cfg["base_model"], cfg["lora"])

    train_records = load_manifest(cfg["manifest_path"], cfg["train_split"])
    eval_records = load_manifest(cfg["manifest_path"], cfg["eval_split"])
    if max_train_examples:
        train_records = train_records[:max_train_examples]
        eval_records = eval_records[:max_train_examples]
    print(f"[day3] train={len(train_records)} eval={len(eval_records)}")

    sample_rate = cfg.get("sample_rate", 16000)

    # Augmentation is train-only, on purpose: augmenting eval would make a real
    # robustness gain indistinguishable from an easier test set.
    aug_cfg = cfg.get("augment") or {}
    augmenter = None
    if aug_cfg.get("enabled"):
        augmenter = Augmenter(
            sample_rate=sample_rate,
            p_clean=aug_cfg.get("p_clean", 0.35),
            p_noise=aug_cfg.get("p_noise", 0.6),
            p_babble=aug_cfg.get("p_babble", 0.3),
            p_reverb=aug_cfg.get("p_reverb", 0.3),
            p_telephone=aug_cfg.get("p_telephone", 0.15),
            p_gain=aug_cfg.get("p_gain", 0.4),
            snr_range=tuple(aug_cfg.get("snr_range", (5.0, 25.0))),
            seed=aug_cfg.get("seed", 0),
        )
        pool_n = aug_cfg.get("babble_pool", 24)
        pool = []
        for rec in train_records[:: max(1, len(train_records) // pool_n)][:pool_n]:
            clip, _ = sf.read(rec["path"], dtype="float32")
            pool.append(clip)
        augmenter.set_babble_pool(pool)
        print(f"[day3] augmentation ON (babble pool: {len(pool)} clips)")

    train_ds = ManifestAudioDataset(train_records, processor, sample_rate, augmenter=augmenter)
    eval_ds = ManifestAudioDataset(eval_records, processor, sample_rate)
    collator = WhisperCollator(processor)

    t = cfg["training"]
    args = Seq2SeqTrainingArguments(
        output_dir=t["output_dir"],
        per_device_train_batch_size=t.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=t.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 2),
        learning_rate=float(t.get("learning_rate", 1e-3)),
        weight_decay=float(t.get("weight_decay", 0.0)),
        warmup_steps=t.get("warmup_steps", 50),
        num_train_epochs=t.get("num_train_epochs", 3),
        max_steps=max_steps if max_steps else -1,
        bf16=t.get("bf16", True),
        eval_strategy=t.get("eval_strategy", "steps"),
        eval_steps=t.get("eval_steps", 200),
        save_steps=t.get("save_steps", 200),
        save_total_limit=t.get("save_total_limit", 3),
        logging_steps=t.get("logging_steps", 25),
        predict_with_generate=False,  # see module docstring -- forced off, not config-controlled
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=processor.feature_extractor,
    )

    resume_from = get_last_checkpoint(t["output_dir"]) if Path(t["output_dir"]).is_dir() else None
    if resume_from:
        print(f"[day3] resuming from checkpoint: {resume_from}")
    trainer.train(resume_from_checkpoint=resume_from)

    final_dir = Path(t["output_dir"]) / "final_adapter"
    model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"[done] adapter saved -> {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/day3_lora.yaml")
    parser.add_argument("--max_train_examples", type=int, default=None, help="Smoke test: cap dataset size")
    parser.add_argument("--max_steps", type=int, default=None, help="Smoke test: cap training steps")
    args = parser.parse_args()
    run(args.config, args.max_train_examples, args.max_steps)

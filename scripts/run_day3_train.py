"""Day 3: LoRA fine-tune Whisper large-v3 on the FLEURS Urdu manifest. Needs a local
CUDA GPU -- unlike Day 1, this runs on this machine (RTX 5090) rather than Colab.

Usage:
    python scripts/run_day3_train.py --config configs/day3_lora.yaml
    python scripts/run_day3_train.py --config configs/day3_lora.yaml --max_train_examples 8 --max_steps 2  # smoke test
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate
import torch
import yaml
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from src.training.dataset import ManifestAudioDataset, WhisperCollator, load_manifest
from src.training.lora_setup import load_lora_model, load_processor

wer_metric = evaluate.load("wer")


def make_compute_metrics(processor):
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100 * wer_metric.compute(predictions=pred_str, references=label_str)}

    return compute_metrics


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
    train_ds = ManifestAudioDataset(train_records, processor, sample_rate)
    eval_ds = ManifestAudioDataset(eval_records, processor, sample_rate)
    collator = WhisperCollator(processor)

    t = cfg["training"]
    args = Seq2SeqTrainingArguments(
        output_dir=t["output_dir"],
        per_device_train_batch_size=t.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=t.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 2),
        learning_rate=float(t.get("learning_rate", 1e-3)),
        warmup_steps=t.get("warmup_steps", 50),
        num_train_epochs=t.get("num_train_epochs", 3),
        max_steps=max_steps if max_steps else -1,
        bf16=t.get("bf16", True),
        eval_strategy=t.get("eval_strategy", "steps"),
        eval_steps=t.get("eval_steps", 200),
        save_steps=t.get("save_steps", 200),
        save_total_limit=t.get("save_total_limit", 3),
        logging_steps=t.get("logging_steps", 25),
        predict_with_generate=t.get("predict_with_generate", True),
        generation_max_length=t.get("generation_max_length", 225),
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
        compute_metrics=make_compute_metrics(processor),
        processing_class=processor.feature_extractor,
    )

    trainer.train()

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

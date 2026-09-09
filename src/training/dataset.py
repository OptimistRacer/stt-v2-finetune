"""Manifest-driven audio/text dataset + collator for Whisper LoRA fine-tuning.

Reads clips straight from manifest_opendata.jsonl (see scripts/fetch_open_datasets.py)
-- no separate train/val/test split step, since the manifest's `source_file` already
carries the split FLEURS shipped it with (e.g. "google/fleurs/ur_pk:train").
"""
import json
from dataclasses import dataclass
from typing import Any

import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import WhisperProcessor


def load_manifest(path: str, split: str) -> list[dict]:
    """Records for one split. Prefers an explicit `split` field (written by
    src/manifest/split_and_merge.py for the combined FLEURS+podcast manifest) and falls
    back to parsing it out of FLEURS's `source_file` (e.g. '...ur_pk:train'), so the
    original FLEURS-only manifest and its baseline numbers stay reproducible."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec_split = rec.get("split") or rec["source_file"].rsplit(":", 1)[-1]
            if rec_split == split:
                records.append(rec)
    return records


class ManifestAudioDataset(Dataset):
    def __init__(self, records: list[dict], processor: WhisperProcessor, sample_rate: int = 16000):
        self.records = records
        self.processor = processor
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        audio, sr = sf.read(rec["path"], dtype="float32")
        if sr != self.sample_rate:
            raise ValueError(f"{rec['path']} is {sr}Hz, expected {self.sample_rate}Hz")
        features = self.processor.feature_extractor(audio, sampling_rate=sr).input_features[0]
        labels = self.processor.tokenizer(rec["draft_transcript"]).input_ids
        return {"input_features": features, "labels": labels}


@dataclass
class WhisperCollator:
    """Pads variable-length feature/label batches; standard shape for HF's Whisper
    fine-tuning recipe -- labels padded with -100 so the loss ignores them, and the
    leading `<|startoftranscript|>` token (which the tokenizer adds because the
    processor was configured with language/task) is stripped, since the model's own
    `shift_tokens_right` prepends `decoder_start_token_id` (also
    `<|startoftranscript|>`) when building decoder_input_ids from labels -- otherwise
    every decoder position is off by one for the model's entire target sequence.

    Bug fixed 2026-09-09: this used to compare against
    `processor.tokenizer.bos_token_id`, which for Whisper is 50257
    (`<|endoftext|>` -- Whisper's tokenizer sets bos_token_id == eos_token_id, it is
    NOT `<|startoftranscript|>`, which is a distinct token, 50258).  That comparison
    was never true, so the strip never fired, and every single training example
    carried a duplicate `<|startoftranscript|>` at the start of decoder_input_ids for
    the entire life of this repo's training runs. Teacher-forced loss still dropped
    normally (the model just learned a consistently *shifted* mapping), but real
    (free-running) generation -- which starts from a single correctly-placed prompt,
    not two -- hit a decoder-input distribution the model never actually trained on,
    and degraded into repeating a token after a few words. This is what caused every
    "successful" training run (by loss) to still produce a broken adapter (by WER),
    regardless of learning rate, LoRA rank, or regularization -- all things that were
    tried and ruled out before this was found. Confirmed by generating from an
    untrained (zero LoRA delta -> mathematically identical to base model) PEFT-wrapped
    model: it matched the base model's output quality exactly, proving the corruption
    came from training on mislabeled targets, not from the PEFT/generate() integration
    itself."""

    processor: WhisperProcessor

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": ex["input_features"]} for ex in batch]
        batch_inputs = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": ex["labels"]} for ex in batch]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        sot_id = self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        if not (labels[:, 0] == sot_id).all().item():
            raise ValueError(
                f"Expected every label to start with <|startoftranscript|> ({sot_id}), "
                f"got {labels[:, 0].tolist()} -- processor.tokenizer may not be configured "
                f"with language/task, or the tokenizer's special-token layout changed."
            )
        labels = labels[:, 1:]

        batch_inputs["labels"] = labels
        return batch_inputs

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
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["source_file"].rsplit(":", 1)[-1] == split:
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
    fine-tuning recipe -- labels padded with -100 so the loss ignores them, and a
    duplicate leading BOS token (tokenizer adds one, the model adds another at
    generation time) is stripped if present."""

    processor: WhisperProcessor

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": ex["input_features"]} for ex in batch]
        batch_inputs = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": ex["labels"]} for ex in batch]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]

        batch_inputs["labels"] = labels
        return batch_inputs

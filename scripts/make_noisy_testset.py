"""Build a deterministically-degraded copy of a test split, to measure robustness.

Without this there's no way to tell whether augmented training actually helped: the
existing test sets are clean, and the whole point of augmentation is performance on
audio that isn't. Degradations are applied with a fixed seed and written to disk, so
every model is scored on byte-identical audio -- comparing two models against two
different random degradations would be meaningless.

Uses the same Augmenter as training but with p_clean=0 (every clip degraded) and a
different seed, so this is not a memorization test of the exact noise seen in training.

Usage:
    python scripts/make_noisy_testset.py \
        --manifest D:/Audion-Data/Urdu/open_datasets/manifest_combined.jsonl \
        --split test \
        --out_dir D:/Audion-Data/Urdu/open_datasets/noisy_test \
        --out_manifest D:/Audion-Data/Urdu/open_datasets/manifest_noisy_test.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf

from src.training.augment import Augmenter
from src.training.dataset import load_manifest


def run(manifest: str, split: str, out_dir: str, out_manifest: str, seed: int, sample_rate: int) -> None:
    records = load_manifest(manifest, split)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    aug = Augmenter(sample_rate=sample_rate, p_clean=0.0, seed=seed)
    # Babble drawn from the same split, which is fine here: it is background murmur at
    # 10-25 dB below the speech, not a transcription target.
    pool = []
    for rec in records[:: max(1, len(records) // 24)][:24]:
        clip, _ = sf.read(rec["path"], dtype="float32")
        pool.append(clip)
    aug.set_babble_pool(pool)

    out_records = []
    for i, rec in enumerate(records, 1):
        audio, sr = sf.read(rec["path"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        noisy = aug(audio)
        out_path = out_dir_p / f"{rec['clip_id']}.wav"
        sf.write(out_path, noisy, sr)
        out_records.append({**rec, "path": str(out_path.resolve()), "clean_path": rec["path"]})
        if i % 50 == 0 or i == len(records):
            print(f"  {i}/{len(records)}")

    out_manifest_p = Path(out_manifest)
    out_manifest_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_manifest_p, "w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[done] {len(out_records)} degraded clips -> {out_dir}")
    print(f"[done] manifest -> {out_manifest}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="D:/Audion-Data/Urdu/open_datasets/manifest_combined.jsonl")
    p.add_argument("--split", default="test")
    p.add_argument("--out_dir", default="D:/Audion-Data/Urdu/open_datasets/noisy_test")
    p.add_argument("--out_manifest", default="D:/Audion-Data/Urdu/open_datasets/manifest_noisy_test.jsonl")
    p.add_argument("--seed", type=int, default=1234, help="Fixed so every model sees identical audio.")
    p.add_argument("--sample_rate", type=int, default=16000)
    a = p.parse_args()
    run(a.manifest, a.split, a.out_dir, a.out_manifest, a.seed, a.sample_rate)

"""Merge Common Voice Urdu into the FLEURS+podcast training manifest.

Adding CV's ~20 h of read-aloud speech makes it ~87% of training data, and the
podcast audio -- conversational, code-switched, and the actual deployment domain --
shrinks to a sliver. The README records what happened last time read speech dominated:
FLEURS-only training reached 56.2% WER on English-heavy podcast clips vs 26.2% for the
untouched base model. So podcast *train* clips are repeated PODCAST_REPEAT times to
keep conversational speech meaningfully weighted. Whether that's enough is decided by
the per-domain numbers from scripts/eval_served.py, not assumed.

Writes a new manifest; the existing ones are left untouched so earlier results stay
reproducible.

    python scripts/merge_cv_manifest.py
    python scripts/merge_cv_manifest.py --podcast_repeat 1     # no oversampling
"""
import argparse
import json
from collections import Counter
from pathlib import Path

BASE = Path(r"D:\Audion-Data\Urdu\open_datasets")
COMBINED = BASE / "manifest_combined.jsonl"
CV = BASE / "manifest_common_voice_ur.jsonl"
OUT = BASE / "manifest_combined_cv.jsonl"


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--podcast_repeat", type=int, default=3)
    args = ap.parse_args()

    combined, cv = read(COMBINED), read(CV)
    out = []
    for rec in combined:
        is_podcast_train = rec["source_id"].startswith("podcast") and rec.get("split") == "train"
        repeats = args.podcast_repeat if is_podcast_train else 1
        for k in range(repeats):
            out.append(rec if k == 0 else {**rec, "clip_id": f"{rec['clip_id']}__rep{k}"})
    out += cv

    with open(OUT, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    hours = Counter()
    for rec in out:
        if rec.get("split") == "train":
            domain = ("podcast" if rec["source_id"].startswith("podcast")
                      else "common_voice" if rec["source_id"] == "common_voice_ur" else "fleurs")
            hours[domain] += rec["duration_s"] / 3600
    total = sum(hours.values())
    print(f"[merge] {len(out)} records -> {OUT}")
    for domain, h in sorted(hours.items()):
        print(f"  train {domain:13s} {h:5.1f} h  ({100 * h / total:4.1f}%)")


if __name__ == "__main__":
    main()

"""Day 2, final step: filter podcast clips to neutral, split them, and merge with the
FLEURS manifest into one combined training manifest. See src/manifest/split_and_merge.py.

Usage:
    python scripts/run_day2_split_merge.py \
        --podcast_manifest outputs/urdu/manifest_verified.jsonl \
        --fleurs_manifest D:/Audion-Data/Urdu/open_datasets/manifest_opendata.jsonl \
        --out D:/Audion-Data/Urdu/open_datasets/manifest_combined.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.manifest.split_and_merge import assign_splits, filter_neutral, load_jsonl, merge


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--podcast_manifest", default="outputs/urdu/manifest_verified.jsonl")
    parser.add_argument("--fleurs_manifest", default="D:/Audion-Data/Urdu/open_datasets/manifest_opendata.jsonl")
    parser.add_argument("--out", default="D:/Audion-Data/Urdu/open_datasets/manifest_combined.jsonl")
    args = parser.parse_args()

    podcast = load_jsonl(args.podcast_manifest)
    kept, dropped = filter_neutral(podcast)
    print(f"[split-merge] podcast: {len(podcast)} -> {len(kept)} neutral (dropped {dropped})")

    kept = assign_splits(kept)
    counts = merge(args.fleurs_manifest, kept, args.out)
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    print(f"\n[done] combined manifest -> {counts['out_path']}")

"""Day 2, export step: turn the Day-1 manifest into per-podcast review files for
AI-assisted correction. See src/correction/export_for_review.py.

Usage:
    python scripts/run_day2_export.py \
        --manifest outputs/urdu/manifest_day1_complete.jsonl \
        --out_dir outputs/urdu/review
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.correction.export_for_review import export


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/urdu/manifest_day1_complete.jsonl")
    parser.add_argument("--out_dir", default="outputs/urdu/review")
    args = parser.parse_args()
    result = export(args.manifest, args.out_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n[day2-export] review files + CORRECTION_GUIDE.md -> {result['review_dir']}")
    print("Correct each *_for_review.jsonl, then re-import with run_day2_import.py")

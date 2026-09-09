"""Day 2, import step: merge corrected review files back into a verified manifest.
See src/correction/import_corrections.py.

Usage:
    python scripts/run_day2_import.py \
        --manifest outputs/urdu/manifest_day1_complete.jsonl \
        --corrected outputs/urdu/review/podcast1_for_review.jsonl \
                    outputs/urdu/review/podcast2_for_review.jsonl \
                    outputs/urdu/review/podcast3_for_review.jsonl \
        --out outputs/urdu/manifest_verified.jsonl \
        --glossary configs/codeswitch_glossary.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.correction.import_corrections import import_corrections


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/urdu/manifest_day1_complete.jsonl")
    parser.add_argument("--corrected", nargs="+", required=True, help="Corrected *_for_review.jsonl files")
    parser.add_argument("--out", default="outputs/urdu/manifest_verified.jsonl")
    parser.add_argument("--glossary", default="configs/codeswitch_glossary.json")
    args = parser.parse_args()

    result = import_corrections(args.manifest, args.corrected, args.out, args.glossary)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] == "error":
        print("\n[day2-import] REFUSED to write -- fix the problems above and rerun.")
        sys.exit(1)
    print(f"\n[day2-import] verified manifest -> {result['out_path']}")

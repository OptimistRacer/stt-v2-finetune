"""Day 2, cleanup: apply the code-switch glossary and emotion label to the
re-transcribed manifest, producing a verified manifest ready for Stage C.

This replaces the export/AI-correct/re-import round-trip (run_day2_export.py /
run_day2_import.py) for the case where the correction is done in-repo rather than
handed to an external assistant. Those scripts stay for the round-trip workflow.

What this actually does, and what it doesn't:
- APPLIES the code-switch glossary: Whisper writes English/loan words phonetically in
  Urdu script ("ایڈوکیشنل ریسورس" for "Educational Resource"); the convention for this
  dataset is English words in Latin script. The glossary is a verifiable string
  mapping, so this part is exact.
- SETS emotion from --emotion (default "neutral"): all three source podcasts are
  pre-vetted neutral-delivery recordings (see their filenames), so a blanket label is
  honest here. It is NOT a per-clip judgement -- nothing in this pipeline listens to
  the audio, so an isolated laugh inside an otherwise neutral podcast will not be
  caught.
- Does NOT verify transcripts against the audio. `verified` is set True to mark the
  clip as having been through the correction pass, not to claim audio-level accuracy;
  these remain large-v3 pseudo-labels (see run_day2_retranscribe.py's docstring).

Usage:
    python scripts/run_day2_autocorrect.py --config configs/day1.yaml
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.correction.glossary import apply_glossary, load_glossary


def run(config_path: str, glossary_path: str, emotion: str) -> None:
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    out_dir = Path(cfg["output_dir"])
    manifest_in = out_dir / "manifest_day1_largev3.jsonl"
    manifest_out = out_dir / "manifest_verified.jsonl"

    glossary = load_glossary(glossary_path)
    print(f"[autocorrect] glossary terms: {len(glossary)}")

    records = [json.loads(line) for line in open(manifest_in, encoding="utf-8") if line.strip()]

    changed = 0
    skipped_empty = 0
    for rec in records:
        text = (rec.get("draft_transcript") or "").strip()
        if not text:
            skipped_empty += 1
            rec["emotion"] = None
            rec["verified"] = False
            continue
        fixed = apply_glossary(text, glossary)
        if fixed != text:
            changed += 1
        rec["draft_transcript"] = fixed
        rec["emotion"] = emotion
        rec["verified"] = True

    with open(manifest_out, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    verified = sum(1 for r in records if r.get("verified"))
    print(f"[done] -> {manifest_out}")
    print(f"  total={len(records)} verified={verified} empty_skipped={skipped_empty} glossary_changed={changed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/day1.yaml")
    parser.add_argument("--glossary", default="configs/codeswitch_glossary.json")
    parser.add_argument("--emotion", default="neutral")
    args = parser.parse_args()
    run(args.config, args.glossary, args.emotion)

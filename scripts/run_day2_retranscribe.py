"""Day 2, re-transcription: replace the Whisper-base draft transcripts with
Whisper-large-v3 ones.

Why: Day 1's drafts come from Whisper-base (small, fast, and used there only to get
*something* on the page). Those drafts are garbled enough that correcting them from
text alone is closer to guessing than correcting -- you can't recover what was actually
said without listening. large-v3 is the same model this repo fine-tunes and is far more
accurate, so it gives a much better starting point for the text cleanup pass at zero
manual cost.

Known limitation, deliberately accepted: these are pseudo-labels from the very model
being fine-tuned, so they carry little *new* text signal for it. The podcast data's
value here is mostly acoustic -- conversational delivery, mic, and background that
FLEURS's clean read speech doesn't cover. Genuinely better labels would need a human
listening to each clip, which this workflow trades away for speed.

Runs through faster-whisper (CTranslate2), not torch's generate(), so it avoids the
fused-attention path implicated in this machine's GPU crashes (see README).

Usage:
    python scripts/run_day2_retranscribe.py --config configs/day1.yaml
    python scripts/run_day2_retranscribe.py --config configs/day1.yaml --limit 5  # smoke test
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.transcription.draft_transcribe import load_model, transcribe_clip


def run(config_path: str, model_size: str, limit: int | None) -> None:
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    out_dir = Path(cfg["output_dir"])
    manifest_in = out_dir / "manifest_day1_complete.jsonl"
    manifest_out = out_dir / "manifest_day1_largev3.jsonl"

    records = [json.loads(line) for line in open(manifest_in, encoding="utf-8") if line.strip()]
    if limit:
        records = records[:limit]
    print(f"[retranscribe] {len(records)} clips with {model_size}")

    model = load_model(model_size, device=cfg.get("device", "cuda"))

    for i, rec in enumerate(records, 1):
        rec["draft_transcript_base"] = rec.get("draft_transcript", "")
        rec["draft_transcript"] = transcribe_clip(model, rec["path"], language=cfg.get("language", "ur"))
        rec["transcript_model"] = model_size
        if i % 10 == 0 or i == len(records):
            print(f"  {i}/{len(records)}")

    with open(manifest_out, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    empty = sum(1 for r in records if not (r.get("draft_transcript") or "").strip())
    print(f"[done] -> {manifest_out} ({len(records)} clips, {empty} still empty)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/day1.yaml")
    parser.add_argument("--model_size", default="large-v3")
    parser.add_argument("--limit", type=int, default=None, help="Smoke test: cap clips")
    args = parser.parse_args()
    run(args.config, args.model_size, args.limit)

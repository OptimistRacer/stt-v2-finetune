"""Stage B, export half: turn a Day-1 manifest into per-podcast review files that a
human (with AI assistance) corrects externally, plus a guide explaining the conventions.

Design choice: one JSONL file per source_id, so each can be handed to an AI assistant a
podcast at a time rather than as one giant blob. Two kinds of clip are excluded (and
counted, never silently dropped), because neither fits a text-only correction pass:
- needs_manual_split=True: an over-length clip with no clean silence boundary, would
  need a manual re-cut.
- empty draft_transcript: Whisper-base returned nothing (in this dataset these cluster
  at the ~30s clip ceiling -- real speech the small model failed on, not silence).
  There's no draft text to correct from; recovering these would mean transcribing from
  scratch by ear, which this workflow deliberately doesn't do.
"""
import json
from pathlib import Path

CORRECTION_GUIDE = """\
# Correction guide -- Urdu podcast transcripts

You are correcting draft transcripts produced by Whisper-base (a small model, so
expect real errors). Each line in the `*_for_review.jsonl` file is one audio clip:

    {"clip_id": "...", "duration_s": 12.3, "draft_transcript": "...", "emotion": ""}

## What to edit

1. **draft_transcript** -- fix it to what is actually said. Two conventions:
   - **Urdu words stay in Urdu (Nastaliq) script.**
   - **English / loan words are written in Latin script**, NOT phonetically
     transliterated into Urdu script. e.g. if the speaker says "microphone", write
     `microphone`, not `مائیکروفون`. This matches how the code is switched in speech
     and keeps English terms consistent.
   - If you can't make out a word, leave your best guess -- don't delete the line.

2. **emotion** -- fill in exactly one of:
   - `neutral`  -- ordinary speech, flat/informational tone (what we want to keep)
   - `non-neutral` -- laughter, shouting, crying, heavy emotion, singing, etc.
   Only `neutral` clips will be used for training in this pass. When unsure, prefer
   `neutral` only if the delivery is genuinely flat.

## What NOT to change

- **Do not change `clip_id`.** It's how corrections are matched back to the audio.
  Changing or reordering it will drop that clip on re-import.
- Don't add or remove lines. Every clip_id that goes out must come back exactly once.

Hand the corrected file back and it will be re-imported with `run_day2_import.py`.
"""


def export(manifest_path: str | Path, out_dir: str | Path) -> dict:
    manifest_path = Path(manifest_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_source: dict[str, list[dict]] = {}
    excluded_split = 0
    excluded_empty = 0
    total = 0
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            if rec.get("needs_manual_split"):
                excluded_split += 1
                continue
            if not (rec.get("draft_transcript") or "").strip():
                excluded_empty += 1
                continue
            by_source.setdefault(rec["source_id"], []).append(rec)

    written = {}
    for source_id, recs in sorted(by_source.items()):
        out_path = out_dir / f"{source_id}_for_review.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in recs:
                review_rec = {
                    "clip_id": rec["clip_id"],
                    "duration_s": rec.get("duration_s"),
                    "draft_transcript": rec.get("draft_transcript", ""),
                    "emotion": "",
                }
                f.write(json.dumps(review_rec, ensure_ascii=False) + "\n")
        written[source_id] = len(recs)

    guide_path = out_dir / "CORRECTION_GUIDE.md"
    guide_path.write_text(CORRECTION_GUIDE, encoding="utf-8")

    return {
        "total_clips": total,
        "excluded_needs_manual_split": excluded_split,
        "excluded_empty_transcript": excluded_empty,
        "exported_per_source": written,
        "exported_total": sum(written.values()),
        "review_dir": str(out_dir),
    }

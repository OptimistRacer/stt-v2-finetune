"""Turn a transcript into a plain-text sheet someone can correct in Notepad.

Correcting is far less work than transcribing from scratch, and a handful of spread-out
2-minute chunks is enough to estimate WER for a whole recording -- transcribing all of a
35-minute file by hand would take hours and buy very little extra confidence.

Plain text rather than JSON on purpose: the person doing the correction shouldn't have
to care about escaping or syntax, and a broken quote in a JSON file loses the work.

Usage:
    python scripts/make_correction_sheet.py --json D:/.../realtest_finetuned.json \
        --audio "C:/Users/Sistems-AI/Downloads/Test for SST" \
        --out D:/.../CORRECT_ME.txt
"""
import argparse
import json
from pathlib import Path

SEP = "=" * 70


def build(transcript_json: str, audio_path: str, out_path: str, starts: list[int], window: int) -> None:
    data = json.loads(Path(transcript_json).read_text(encoding="utf-8"))
    segs = data["segments"]

    L = [
        "HOW TO USE THIS FILE",
        "====================",
        f"1. Open your audio file:  {audio_path}",
        f"2. For each PART below, jump to the time shown and listen for {window // 60} minutes.",
        "3. Read the text under 'WHAT THE MODEL HEARD'.",
        "4. Under 'YOUR CORRECTION', fix any words that are wrong.",
        "5. Save this file and send it back.",
        "",
        "Only fix actual wrong words. Ignore punctuation and spacing -- they are not scored.",
        "If a part is too unclear to follow, write SKIP under that part's correction.",
        "",
    ]

    for i, t0 in enumerate(starts, 1):
        text = " ".join(s["text"] for s in segs if t0 <= s["start"] < t0 + window).strip()
        if not text:
            continue
        L += [
            SEP,
            f"PART {i}   --   LISTEN FROM {t0 // 60:02d}:{t0 % 60:02d}   (for {window // 60} minutes)",
            SEP,
            "",
            "--- WHAT THE MODEL HEARD ---",
            text,
            "",
            "--- YOUR CORRECTION (edit the copy below) ---",
            text,
            "",
            "",
        ]

    Path(out_path).write_text("\n".join(L), encoding="utf-8")
    words = sum(len(" ".join(s["text"] for s in segs if t <= s["start"] < t + window).split()) for t in starts)
    print(f"[sheet] {len(starts)} parts, ~{words} words to check -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True, help="Transcript JSON from scripts/transcribe.py")
    p.add_argument("--audio", required=True, help="Path shown in the instructions")
    p.add_argument("--out", required=True)
    p.add_argument("--starts", type=int, nargs="+", default=[60, 480, 900, 1400, 1850],
                   help="Chunk start times in seconds, spread across the recording")
    p.add_argument("--window", type=int, default=120)
    a = p.parse_args()
    build(a.json, a.audio, a.out, a.starts, a.window)

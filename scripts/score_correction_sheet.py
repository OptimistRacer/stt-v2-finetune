"""Score a returned correction sheet: real WER/CER on real audio.

Picks, per part, whichever of the two boxes the human actually edited. The sheet ships
with both boxes containing identical model output, so the edited one is simply the one
that no longer matches the original transcript -- which means a part filled in the wrong
box (easy to do) is still usable rather than lost.

Caveat worth knowing when reading the numbers: the reference covers a fixed time window,
and each model segments audio differently, so the text extracted at the window edges may
clip or include a partial sentence. With 2-minute windows that is a small effect, but it
is not zero, and it applies equally to both models being compared.

Usage:
    python scripts/score_correction_sheet.py \
        --sheet D:/.../CORRECTED_ME.txt \
        --model_json D:/.../realtest_finetuned.json \
        --compare_json D:/.../realtest_base.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jiwer


def parse_sheet(path: str) -> list[dict]:
    txt = Path(path).read_text(encoding="utf-8")
    chunks = re.split(r"PART\s+(\d+)\s+--\s+LISTEN FROM\s+(\d+):(\d+)", txt)
    out = []
    for i in range(1, len(chunks), 4):
        body = chunks[i + 3]
        heard = re.search(r"--- WHAT THE MODEL HEARD ---\s*(.*?)\s*--- YOUR CORRECTION", body, re.S)
        corr = re.search(r"--- YOUR CORRECTION[^\n]*---\s*(.*?)\s*(?:={10,}|$)", body, re.S)
        out.append({
            "part": int(chunks[i]),
            "start": int(chunks[i + 1]) * 60 + int(chunks[i + 2]),
            "heard": heard.group(1).strip() if heard else "",
            "corr": corr.group(1).strip() if corr else "",
        })
    return out


def window_text(segments: list[dict], t0: int, window: int) -> str:
    return " ".join(s["text"] for s in segments if t0 <= s["start"] < t0 + window).strip()


def wer(ref: str, hyp: str) -> float:
    return 100 * jiwer.wer(ref, hyp)


def cer(ref: str, hyp: str) -> float:
    return 100 * jiwer.cer(ref, hyp)


def run(sheet: str, model_json: str, compare_json: str | None, window: int) -> None:
    parts = parse_sheet(sheet)
    model = json.loads(Path(model_json).read_text(encoding="utf-8"))["segments"]
    other = json.loads(Path(compare_json).read_text(encoding="utf-8"))["segments"] if compare_json else None

    refs, hyps, others, rows = [], [], [], []
    for p in parts:
        original = window_text(model, p["start"], window)
        # Whichever box diverges from the original is the edited one. If both were
        # touched, prefer the correction box; if neither, the part was left blank.
        d_heard = wer(original, p["heard"]) if p["heard"] else 0.0
        d_corr = wer(original, p["corr"]) if p["corr"] else 0.0
        if d_corr >= 0.5:
            ref, box = p["corr"], "correction"
        elif d_heard >= 0.5:
            ref, box = p["heard"], "model-heard (filled in wrong box)"
        else:
            rows.append((p["part"], box := "UNEDITED -- skipped", None, None))
            continue
        if ref.strip().upper() == "SKIP":
            rows.append((p["part"], "SKIP", None, None))
            continue

        refs.append(ref)
        hyps.append(original)
        w = wer(ref, original)
        o = None
        if other is not None:
            ot = window_text(other, p["start"], window)
            others.append(ot)
            o = wer(ref, ot)
        rows.append((p["part"], box, w, o))

    print(f"{'part':>4}  {'box used':<34} {'fine-tuned':>11} {'base':>8}")
    for part, box, w, o in rows:
        ws = f"{w:10.1f}%" if w is not None else "         -"
        os_ = f"{o:7.1f}%" if o is not None else "       -"
        print(f"{part:4}  {box:<34} {ws} {os_}")

    if refs:
        R, H = " ".join(refs), " ".join(hyps)
        print()
        print("=" * 58)
        print(f"OVERALL on real audio ({len(refs)} parts, {len(R.split())} reference words)")
        print("=" * 58)
        print(f"  fine-tuned model : WER {wer(R, H):5.2f}%   CER {cer(R, H):5.2f}%")
        if others:
            O = " ".join(others)
            print(f"  base whisper-v3  : WER {wer(R, O):5.2f}%   CER {cer(R, O):5.2f}%")
            delta = wer(R, O) - wer(R, H)
            verdict = "fine-tuned is better" if delta > 0 else "base is better"
            print(f"  difference       : {abs(delta):.2f}pp WER -- {verdict}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sheet", required=True)
    p.add_argument("--model_json", required=True)
    p.add_argument("--compare_json", default=None)
    p.add_argument("--window", type=int, default=120)
    a = p.parse_args()
    run(a.sheet, a.model_json, a.compare_json, a.window)

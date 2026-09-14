"""Discover code-switch dictionary entries by diffing two transcriptions of the same audio.

Transcribe once normally and once with the code-switch prompt, then align the two word
sequences. Where one side has Urdu script and the other has Latin for the same position,
that pair *is* a dictionary entry -- "ٹرگر" -> "trigger" -- found by the model rather
than guessed.

This replaced a heuristic version that tried to spot transliterations by frequency and
absence from a reference Urdu corpus. It did not work: on a religious lecture it
proposed رمضان, دعا, قرآن and صاحب, which are ordinary Urdu missing from FLEURS only
because FLEURS is general-domain. Listing high-frequency Latin words was no better --
the top hits were "you", "the", "and", which Whisper already writes in Latin and which
need no entry at all. Both produced mostly noise, and noise a human must wade through
is worse than nothing, because it makes the review step feel pointless.

Output still goes to a candidates file for a person to approve. The diff is high
precision, not perfect: Whisper can change other wording between the two runs, so a pair
occasionally captures an unrelated substitution.

Usage:
    python scripts/harvest_codeswitch.py --audio recording.mp3
    python scripts/harvest_codeswitch.py --audio a.mp3 --out cands.txt --min-count 2
"""
import argparse
import collections
import difflib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.postprocess.codeswitch import DEFAULT_PATH, CodeSwitcher

LATIN_ONLY = re.compile(r"^[A-Za-z][A-Za-z'-]*$")
URDU_ONLY = re.compile(r"^[؀-ۿݐ-ݿ]+$")


def _words(text: str) -> list[str]:
    return [w.strip("،۔.!?\"'()") for w in text.split() if w.strip("،۔.!?\"'()")]


def diff_pairs(plain: str, prompted: str) -> collections.Counter:
    """Aligned (urdu, latin) pairs where the prompted run switched script."""
    a, b = _words(plain), _words(prompted)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    pairs = collections.Counter()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace":
            continue
        src, dst = a[i1:i2], b[j1:j2]
        # Only trust short, tight substitutions; a long replaced span usually means
        # Whisper rewrote the sentence rather than switching one word's script.
        if len(src) > 3 or len(dst) > 3:
            continue
        for u in src:
            if not URDU_ONLY.match(u):
                continue
            for e in dst:
                if LATIN_ONLY.match(e) and len(e) > 2:
                    pairs[(u, e.lower())] += 1
    return pairs


def run(audio: str, out: str, dict_path: str, min_count: int, model_dir: str | None) -> None:
    from scripts.transcribe import CODESWITCH_PROMPT, DEFAULT_MODEL, transcribe

    model = model_dir or DEFAULT_MODEL
    print("[harvest] pass 1/2: plain transcription")
    plain = transcribe(audio, model_dir=model)
    print("[harvest] pass 2/2: with code-switch prompt")
    prompted = transcribe(audio, model_dir=model, initial_prompt=CODESWITCH_PROMPT)

    pairs = diff_pairs(plain["text"], prompted["text"])
    cs = CodeSwitcher(dict_path)
    known = set(cs.terms) | set(cs.protected)

    rows = [(u, e, n) for (u, e), n in pairs.most_common() if n >= min_count and u not in known]

    lines = [
        "CODE-SWITCH DICTIONARY CANDIDATES",
        "=================================",
        f"From: {audio}",
        "",
        "Found by transcribing twice (with and without the code-switch prompt) and",
        "diffing. Each line is a word the model itself rendered both ways.",
        "",
        "KEEP the lines you want, DELETE the rest, then run:",
        "    python scripts/apply_codeswitch_candidates.py --file <this file>",
        "",
        "Move a word to the PROTECTED section instead if it should NEVER be converted",
        "(naturalized loanwords like منٹ, پروگرام, فون that are normal written Urdu).",
        "",
        "--- PROPOSED ENTRIES ---",
        "",
    ]
    for u, e, n in rows:
        lines.append(f"{u} = {e}    # seen {n}x")
    if not rows:
        lines.append("# (none found above the frequency threshold)")

    lines += ["", "--- PROTECTED (never convert these; one word per line) ---", ""]

    Path(out).write_text("\n".join(lines), encoding="utf-8")
    print(f"[harvest] {len(rows)} candidate(s) -> {out}")
    print(f"[harvest] dictionary has {len(cs.terms)} terms, {len(cs.protected)} protected")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True)
    p.add_argument("--out", default="codeswitch_candidates.txt")
    p.add_argument("--dict", default=str(DEFAULT_PATH))
    p.add_argument("--min-count", type=int, default=1)
    p.add_argument("--model_dir", default=None)
    a = p.parse_args()
    run(a.audio, a.out, a.dict, a.min_count, a.model_dir)

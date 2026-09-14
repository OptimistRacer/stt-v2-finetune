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
from src.postprocess.translit import looks_like_transliteration, similarity

LATIN_ONLY = re.compile(r"^[A-Za-z][A-Za-z'-]*$")
URDU_ONLY = re.compile(r"^[؀-ۿݐ-ݿ]+$")


def _words(text: str) -> list[str]:
    return [w.strip("،۔.!?\"'()") for w in text.split() if w.strip("،۔.!?\"'()")]


def diff_pairs(plain: str, prompted: str, check_phonetics: bool = True) -> collections.Counter:
    """Aligned (urdu, latin) pairs where the prompted run switched script.

    Two filters, both added after seeing what the unfiltered version proposed:

    - Words are paired POSITIONALLY when the replaced spans are the same length. Taking
      the cross-product instead turned "سوشل میڈیا ایپس" -> "social media apps" into
      nine pairs (every Urdu word against every English one) rather than three.
    - Pairs must be phonetically plausible. The model sometimes *translates* rather than
      transliterates, producing تقریباً/"almost", تحجد/"time", ہے/"possible". Those are
      the dangerous ones: تقریباً is ordinary Urdu for "almost", so accepting it would
      rewrite a real Urdu word as English everywhere it occurs.
    """
    a, b = _words(plain), _words(prompted)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    pairs = collections.Counter()

    def keep(u: str, e: str) -> bool:
        if not URDU_ONLY.match(u) or not LATIN_ONLY.match(e) or len(e) <= 2:
            return False
        return looks_like_transliteration(u, e) if check_phonetics else True

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace":
            continue
        src, dst = a[i1:i2], b[j1:j2]
        # A long replaced span usually means Whisper rewrote the sentence rather than
        # switching one word's script.
        if len(src) > 3 or len(dst) > 3:
            continue
        if len(src) == len(dst):
            candidates = list(zip(src, dst))
        else:
            # Lengths differ, so position carries no information; fall back to the
            # cross-product and let the phonetic filter discard the mismatches.
            candidates = [(u, e) for u in src for e in dst]
        for u, e in candidates:
            if keep(u, e):
                pairs[(u, e.lower())] += 1
    return pairs


def run(audio: str, out: str, dict_path: str, min_count: int, model_dir: str | None,
        chunk_s: int = 60) -> None:
    from scripts.transcribe import CODESWITCH_PROMPT, DEFAULT_MODEL, transcribe

    model = model_dir or DEFAULT_MODEL
    print("[harvest] pass 1/2: plain transcription")
    plain = transcribe(audio, model_dir=model)
    print("[harvest] pass 2/2: with code-switch prompt (chunked)")
    # chunked, else the prompt only affects the first 30s window and the diff is empty
    prompted = transcribe(audio, model_dir=model, initial_prompt=CODESWITCH_PROMPT,
                          prompt_chunk_s=chunk_s)

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
        lines.append(f"{u} = {e}    # seen {n}x, phonetic match {similarity(u, e):.2f}")
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
    p.add_argument("--prompt_chunk_s", type=int, default=60,
                   help="Re-apply the prompt every N seconds in the prompted pass")
    a = p.parse_args()
    run(a.audio, a.out, a.dict, a.min_count, a.model_dir, a.prompt_chunk_s)

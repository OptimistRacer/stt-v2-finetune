"""Is a model really worse (or better) than base on the same clips, or is it noise?

Paired bootstrap over clips. Reads the per-clip result files written by
`eval_arabic.py --out` (the TTS repo's scorer), named `<prefix>_<set>.json`, and for each
candidate reports the corpus-WER difference against base with a 95% interval. Positive
means worse than base. Pairing by clip keeps the clip difficulty out of the comparison,
which is what makes gaps of a few tenths of a point testable on a few hundred clips.

WER is recomputed per clip with the scorer's own text normalisation (copied below -- keep
it in sync with eval_arabic.py) and checked against each file's stored summary, so a
drift between the two shows up as an error rather than as quietly different numbers.

    python scripts/bootstrap_vs_base.py --eval_dir D:/Audion-Data/Arabic/eval \
        --base base --candidates clean ckpt500 ckpt1000 --sets sa bh ye

Prints each set on its own, then all sets pooled. The pool is a plain concatenation of
clips, so larger sets weigh more.
"""
import argparse
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

import jiwer

DIACRITICS = re.compile(r"[ً-ْٰ]")
PUNCT = re.compile(r"[،.؟?!:;\"«»()\-_]")


def normalize(text: str) -> str:
    """Same as eval_arabic.normalize(text, "ar")."""
    text = unicodedata.normalize("NFC", text or "")
    text = DIACRITICS.sub("", text)
    text = PUNCT.sub(" ", text)
    text = re.sub("[أإآٱ]", "ا", text).replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    return re.sub(r"\s+", " ", text).strip()


def clip_errors(path: Path) -> dict[str, tuple[int, int]]:
    """clip -> (word errors, reference words). Clips with an empty reference are skipped,
    as the scorer does."""
    data = json.load(open(path, encoding="utf-8"))
    out = {}
    for c in data["clips"]:
        ref, hyp = normalize(c["reference"]), normalize(c["heard"])
        n_ref = len(ref.split())
        if n_ref == 0:
            continue
        m = jiwer.process_words(ref, hyp)
        out[c["clip"]] = (m.substitutions + m.deletions + m.insertions, n_ref)

    errs, words = sum(e for e, _ in out.values()), sum(n for _, n in out.values())
    stored = data["summary"]["wer"]
    if abs(100 * errs / words - stored) > 0.06:
        sys.exit(f"{path.name}: recomputed WER {100 * errs / words:.2f} != stored {stored} -- "
                 "normalisation here has drifted from eval_arabic.py")
    return out


def bootstrap(pairs, boots: int):
    """pairs: [(base (err, n_ref), candidate (err, n_ref)), ...] -> (diff, lo, hi, P(worse))"""
    n = len(pairs)
    diffs = []
    for _ in range(boots):
        eb = ec = nr = 0
        for _ in range(n):
            (b, n_ref), (c, _) = pairs[random.randrange(n)]
            eb += b
            ec += c
            nr += n_ref
        diffs.append(100 * (ec - eb) / nr)
    diffs.sort()
    point = 100 * (sum(c[0] for _, c in pairs) - sum(b[0] for b, _ in pairs)) / sum(b[1] for b, _ in pairs)
    return point, diffs[int(0.025 * boots)], diffs[int(0.975 * boots)], sum(d > 0 for d in diffs) / boots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True, help="folder holding <prefix>_<set>.json files")
    ap.add_argument("--base", default="base", help="file prefix of the baseline model")
    ap.add_argument("--candidates", nargs="+", required=True, help="file prefixes to compare to base")
    ap.add_argument("--sets", nargs="+", required=True, help="set names, e.g. sa bh ye")
    ap.add_argument("--boots", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    d = Path(a.eval_dir)
    errs = {p: {s: clip_errors(d / f"{p}_{s}.json") for s in a.sets} for p in [a.base, *a.candidates]}
    print(f"recomputed WER matches the stored summary in all {len(errs) * len(a.sets)} files\n")

    random.seed(a.seed)
    print(f"WER difference vs {a.base} (positive = worse), paired bootstrap over clips, 95% CI")
    print(f"{'subset':16} {'model':10} {'diff':>6}  {'95% CI':>15}  P(worse)")
    subsets = [(s, [s]) for s in a.sets]
    if len(a.sets) > 1:
        subsets.append(("pooled", list(a.sets)))
    for label, keys in subsets:
        for cand in a.candidates:
            pairs = [(errs[a.base][k][c], errs[cand][k][c])
                     for k in keys for c in errs[a.base][k] if c in errs[cand][k]]
            point, lo, hi, p_worse = bootstrap(pairs, a.boots)
            print(f"{label + ' (' + str(len(pairs)) + ')':16} {cand:10} {point:+6.2f}  "
                  f"[{lo:+5.2f}, {hi:+5.2f}]  {p_worse:5.1%}")
        print()


if __name__ == "__main__":
    main()

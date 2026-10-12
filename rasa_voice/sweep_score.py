"""Score a settings sweep: word / character error as heard back by a speech recogniser,
per config, plus the "best of the takes" error (what the app's self-check would give).

    python sweep_score.py                                   # Parvaaz sweep: eval_out/sweep/, Urdu judge
    python sweep_score.py --dir ..\\candidates\\out\\arabic_sweep --lang ar   # Arabic sweep, base Whisper judge

Expects <dir>/<config>_s<seed>/<clip>.wav and, for the reference texts, the shared test
sentences (std<i>) plus eval_out/heldout_texts.json (Urdu) or candidates/sentences.json (Arabic).
Writes <dir>/scores.json and prints a table.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import jiwer
from faster_whisper import WhisperModel

HERE = Path(__file__).resolve().parent
CAND = HERE.parent / "candidates"
sys.path.insert(0, str(CAND))
from score_voices import JUDGES, SENTENCES, normalize  # noqa: E402


def references(lang):
    refs = {f"std{i}": t for i, t in enumerate(SENTENCES[lang])}
    if lang == "ur":
        refs.update({h["id"]: h["text"] for h in json.loads((HERE / "eval_out" / "heldout_texts.json").read_text(encoding="utf-8"))})
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE / "eval_out" / "sweep"))
    ap.add_argument("--lang", default="ur")
    a = ap.parse_args()
    root, lang = Path(a.dir), a.lang
    refs = references(lang)
    judge = WhisperModel(JUDGES[lang], device="cuda", compute_type="float16")
    per = defaultdict(dict)  # config -> seed -> {clip: (errors, words, char_errs, chars)}
    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        m = re.match(r"(.+)_s(\d+)$", run.name)
        if not m:
            continue
        cfg, seed = m.group(1), int(m.group(2))
        clips = {}
        for wav in sorted(run.glob("*.wav")):
            if wav.stem not in refs:
                continue
            segs, _ = judge.transcribe(str(wav), language=lang, beam_size=5, vad_filter=True)
            r, h = normalize(refs[wav.stem], lang), normalize(" ".join(s.text.strip() for s in segs), lang)
            w, c = jiwer.process_words(r, h), jiwer.process_characters(r, h)
            clips[wav.stem] = (w.substitutions + w.deletions + w.insertions, len(r.split()),
                               c.substitutions + c.deletions + c.insertions, len(r))
        per[cfg][seed] = clips

    results = {}
    print(f"{'config':18s} {'one take':>16s} {'best of takes':>16s}   (word / char error over all clips)")
    for cfg, seeds in per.items():
        ids = sorted(set.intersection(*(set(c) for c in seeds.values())))
        tot = lambda pick: [sum(pick(i)[k] for i in ids) for k in range(4)]
        one = [sum(seeds[s][i][k] for s in seeds for i in ids) / len(seeds) for k in range(4)]
        best = tot(lambda i: min((seeds[s][i] for s in seeds), key=lambda x: (x[0] / max(1, x[1]), x[2])))
        results[cfg] = {"clips": len(ids), "takes": len(seeds),
                        "one_take": {"wer": round(100 * one[0] / one[1], 1), "cer": round(100 * one[2] / one[3], 1)},
                        "best_of_takes": {"wer": round(100 * best[0] / best[1], 1), "cer": round(100 * best[2] / best[3], 1)}}
        o, b = results[cfg]["one_take"], results[cfg]["best_of_takes"]
        print(f"{cfg:18s} {o['wer']:6.1f}% / {o['cer']:4.1f}% {b['wer']:7.1f}% / {b['cer']:4.1f}%")
    (root / "scores.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Score Parvaaz checkpoints by what our Urdu STT model hears back.

Two sets:
- the 6 shared test sentences (candidates/out/parvaaz_<tag>/), comparable with the
  OmniVoice / Meta / untrained-Parler numbers in candidates/parler_base_scores.json
- the 20 held-out Rasa sentences (eval_out/<tag>/, made by gen_heldout.py), plus the
  real recordings of the same sentences (eval_out/real/) -- the judge's own error on the
  real speaker is the floor a synthetic voice can't go below.

    python score_parvaaz.py step150 step300 step450 step600 final     (main environment)

Writes eval_out/scores.json and prints a table.
"""
import json
import sys
from pathlib import Path

import jiwer
import pyarrow.parquet as pq
from faster_whisper import WhisperModel

HERE = Path(__file__).resolve().parent
CAND = HERE.parent / "candidates"
sys.path.insert(0, str(CAND))
from score_voices import JUDGES, SENTENCES, normalize  # noqa: E402


def extract_real(items):
    real = HERE / "eval_out" / "real"
    real.mkdir(parents=True, exist_ok=True)
    want = {i["id"] for i in items} - {p.stem for p in real.glob("*.wav")}
    if want:
        for f in (HERE / "rasa_male_hf" / "data").glob("test-*.parquet"):
            for r in pq.read_table(f, columns=["id", "audio"]).to_pylist():
                if r["id"] in want:
                    (real / f"{r['id']}.wav").write_bytes(r["audio"]["bytes"])
    return real


def score(judge, pairs):
    refs, hyps, clips = [], [], []
    for wav, text in pairs:
        if not wav.exists():
            continue
        segs, _ = judge.transcribe(str(wav), language="ur", beam_size=5, vad_filter=True)
        heard = " ".join(s.text.strip() for s in segs)
        r, h = normalize(text, "ur"), normalize(heard, "ur")
        refs.append(r)
        hyps.append(h)
        clips.append({"clip": wav.stem, "wer": round(100 * jiwer.wer(r, h), 1), "heard": heard})
    if not refs:
        return None
    return {"wer": round(100 * jiwer.wer(refs, hyps), 1), "cer": round(100 * jiwer.cer(refs, hyps), 1), "n": len(refs), "clips": clips}


def main():
    tags = sys.argv[1:]
    items = json.loads((HERE / "eval_out" / "heldout_texts.json").read_text(encoding="utf-8"))
    real = extract_real(items)
    judge = WhisperModel(JUDGES["ur"], device="cuda", compute_type="float16")
    path = HERE / "eval_out" / "scores.json"
    results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    results["real_recordings"] = {"heldout": score(judge, [(real / f"{i['id']}.wav", i["text"]) for i in items])}
    for tag in tags:
        results[tag] = {
            "std6": score(judge, [(CAND / "out" / f"parvaaz_{tag}" / f"ur_{k}.wav", t) for k, t in enumerate(SENTENCES["ur"])]),
            "heldout": score(judge, [(HERE / "eval_out" / tag / f"{i['id']}.wav", i["text"]) for i in items]),
        }
    path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    fmt = lambda s: f"{s['wer']:5.1f}% / {s['cer']:4.1f}%" if s else "      -      "
    print(f"{'model':16s} {'6 test sentences':>18s} {'20 held-out Rasa':>18s}   (word / character error)")
    for k, v in results.items():
        print(f"{k:16s} {fmt(v.get('std6')):>18s} {fmt(v.get('heldout')):>18s}")


if __name__ == "__main__":
    main()

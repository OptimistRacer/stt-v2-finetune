"""Build the training set for our Urdu voice from the male speaker of AI4Bharat's Rasa
dataset (CC-BY-4.0: commercial use allowed with attribution).

Reads the Rasa Urdu parquet shards already downloaded to ../rasa_urdu/Urdu/ (only the
ones holding the male speaker -- see ../rasa_urdu/shard_index.json) and writes a local
Hugging Face-style dataset that Parler-TTS's training script and DataSpeech can both
load with load_dataset(<folder>):

    data/train-*.parquet, data/test-*.parquet
    columns: id, audio (48 kHz, as recorded), text, style, duration

Choices, each from looking at the data:
- PROPER NOUN rows are dropped: short lists of Indian place and company names, mostly
  under 2 s, which teach nothing about speaking Urdu naturally.
- Text is NFC-normalised; invisible direction marks, no-break spaces and newlines become
  plain spaces. A row with letters from another script (one has Bengali) is dropped.
- 2-20 s only, matching the training script's duration limits.
- Rasa's own split is kept: its train shards train, its test shards are held out for
  evaluation, so nothing we score on was trained on.

Run with a Python that has `datasets` < 4 (venv_parler or venv_dataspeech):
    ..\\candidates\\venv_parler\\Scripts\\python.exe prepare_rasa_male.py
"""
import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Audio, Dataset, Features, Value

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "rasa_urdu"
OUT = HERE / "rasa_male_hf"
DROP_STYLES = {"PROPER NOUN"}
MIN_S, MAX_S = 2.0, 20.0
SHARD_ROWS = 2000

ARABIC_SCRIPT = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
OTHER_LETTER = re.compile(r"[^\W\d_]")  # any letter; checked against ARABIC_SCRIPT below


def clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[‎‏‪-‮ \n\r\t]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def foreign_letters(text: str) -> bool:
    return any(not ARABIC_SCRIPT.match(ch) for ch in OTHER_LETTER.findall(text))


def main():
    # Defaults reproduce the male-only set Parvaaz v1 was trained on. Parvaaz v2 adds the
    # female speaker: --genders Male Female --out rasa_mf_hf
    ap = argparse.ArgumentParser()
    ap.add_argument("--genders", nargs="+", default=["Male"])
    ap.add_argument("--out", default="rasa_male_hf")
    a = ap.parse_args()
    global OUT
    OUT = HERE / a.out
    index = json.loads((SRC / "shard_index.json").read_text(encoding="utf-8"))
    shards = [SRC / r["file"] for r in index if set(a.genders) & set(r["genders"])]
    missing = [s for s in shards if not s.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} shard(s) not downloaded yet, e.g. {missing[0].name}")

    kept = {"train": [], "test": []}
    dropped = Counter()
    for shard in shards:
        split = "test" if shard.name.startswith("test-") else "train"
        for r in pq.read_table(shard).to_pylist():
            if r["gender"] not in a.genders:
                continue
            dur = float(r["duration"])
            text = clean(r["text"])
            if r["style"] in DROP_STYLES:
                dropped["proper_noun"] += 1
            elif not (MIN_S <= dur <= MAX_S):
                dropped["duration"] += 1
            elif not text or foreign_letters(text):
                dropped["text"] += 1
            else:
                kept[split].append({"id": r["filename"], "audio": {"bytes": r["audio"]["bytes"], "path": r["filename"] + ".wav"},
                                    "text": text, "style": r["style"], "gender": r["gender"], "duration": dur})

    features = Features({"id": Value("string"), "audio": Audio(sampling_rate=48000), "text": Value("string"),
                         "style": Value("string"), "gender": Value("string"), "duration": Value("float32")})
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    summary = {"dropped": dict(dropped)}
    for split, rows in kept.items():
        rows.sort(key=lambda r: r["id"])
        for old in (OUT / "data").glob(f"{split}-*.parquet"):
            old.unlink()
        n_shards = max(1, -(-len(rows) // SHARD_ROWS))
        for i in range(n_shards):
            part = rows[i * SHARD_ROWS:(i + 1) * SHARD_ROWS]
            Dataset.from_list(part, features=features).to_parquet(
                str(OUT / "data" / f"{split}-{i:05d}-of-{n_shards:05d}.parquet"))
        summary[split] = {"clips": len(rows), "hours": round(sum(r["duration"] for r in rows) / 3600, 2),
                          "genders": {g: {"clips": sum(r["gender"] == g for r in rows),
                                          "hours": round(sum(r["duration"] for r in rows if r["gender"] == g) / 3600, 2)}
                                      for g in sorted({r["gender"] for r in rows})},
                          "styles": dict(Counter(r["style"] for r in rows).most_common())}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

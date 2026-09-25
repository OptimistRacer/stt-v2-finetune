"""Build a Gulf Arabic *test set* from the Lahgtna v2 test split. Evaluation only.

`oddadmix/dialectal-arabic-lahgtna-v2` states no licence and its audio is re-hosted
from other people's YouTube channels and podcasts (see ARABIC_STATUS.md), so it is used
here only to measure, never to train on or redistribute. Output goes to
D:\\Audion-Data\\Arabic\\gulf_eval_lahgtna, outside every git repo.

Its transcripts are AI-generated, so absolute WER against them is approximate; the
useful signal is the *comparison* between models on the same clips.

Writes one folder per dialect (sa = Saudi, bh = Bahraini, ye = Yemeni) of foo.wav +
foo.txt pairs, the layout eval_arabic.py reads. Non-speech tags such as [exhale] are
removed from the .txt (they aren't words and would count as errors); the untouched
transcript is kept in index.jsonl.

    python fetch_gulf_eval.py
    python eval_arabic.py --dir D:/Audion-Data/Arabic/gulf_eval_lahgtna/sa --language ar
"""
import argparse
import io
import json
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

REPO = "oddadmix/dialectal-arabic-lahgtna-v2"
FILES = ["data/test-00000-of-00002.parquet", "data/test-00001-of-00002.parquet"]
OUT = Path(r"D:\Audion-Data\Arabic\gulf_eval_lahgtna")
GULF = ("sa", "bh", "ye")
TAG = re.compile(r"\[[^\]]*\]")


def decode(audio_bytes: bytes, sample_rate: int = 16000) -> np.ndarray:
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != sample_rate:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=sample_rate)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialects", nargs="+", default=list(GULF))
    args = ap.parse_args()

    counts = {d: 0 for d in args.dialects}
    OUT.mkdir(parents=True, exist_ok=True)
    index = open(OUT / "index.jsonl", "w", encoding="utf-8")
    for name in FILES:
        table = pq.read_table(hf_hub_download(REPO, name, repo_type="dataset"))
        langs = table.column("language").to_pylist()
        texts = table.column("transcript_text").to_pylist()
        audio = table.column("audio")
        for i, (lang, text) in enumerate(zip(langs, texts)):
            if lang not in counts:
                continue
            clean = re.sub(r"\s+", " ", TAG.sub(" ", text)).strip()
            if not clean:
                continue
            counts[lang] += 1
            stem = f"{lang}_{counts[lang]:04d}"
            folder = OUT / lang
            folder.mkdir(exist_ok=True)
            wav = decode(audio[i].as_py()["bytes"])
            sf.write(folder / f"{stem}.wav", wav, 16000)
            (folder / f"{stem}.txt").write_text(clean, encoding="utf-8")
            index.write(json.dumps({"clip": f"{lang}/{stem}.wav", "dialect": lang,
                                    "duration_s": round(len(wav) / 16000, 2),
                                    "transcript_raw": text, "source": f"{REPO}:{name}"},
                                   ensure_ascii=False) + "\n")
        print(f"[gulf] {name}: {counts}")
    index.close()
    print(f"[done] {sum(counts.values())} Gulf clips -> {OUT}  {counts}")


if __name__ == "__main__":
    main()

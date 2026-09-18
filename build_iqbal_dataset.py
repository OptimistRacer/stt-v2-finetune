"""Lay out the clean muhammadsaadgondal/urdu-tts (Iqbal recitation) dataset as a local
audiofolder, so it can be trained with the exact same recipe/settings as the host data.

This is a controlled A/B: if the identical recipe keeps the model intelligible on this
clean, ground-truth-text data but destroys it on the host's ASR-transcribed podcast
data, the transcripts are proven to be the cause.
"""
import json
import random
import sys
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset
from huggingface_hub import get_token
from scipy.signal import resample_poly

OUT_DIR = Path("D:/Audion-Data/TTS/iqbal_dataset")
TARGET_SR = 16000
N_EVAL = 12

with open("D:/Audion-Data/TTS/vits_finetuned_host/vocab.json", encoding="utf-8") as f:
    VOCAB_CHARS = set(json.load(f).keys())
# Optional diacritics (zer/izafat, zabar, shadda, sukun, ...) aren't in the MMS vocab and
# are normally omitted in written Urdu anyway. NFC runs first so decomposed hamza/maddah
# fold into the precomposed letters the vocab does have (e.g. ا + maddah -> آ).
STRIP_CHARS = str.maketrans("", "", "،؟۔!ًٌٍَُِّْٓٔ")


def clean(text):
    return unicodedata.normalize("NFC", text or "").translate(STRIP_CHARS).strip()


def main():
    ds = load_dataset("muhammadsaadgondal/urdu-tts", split="train", token=get_token())
    print(f"[iqbal] {len(ds)} rows loaded")

    train_dir = OUT_DIR / "train"
    val_dir = OUT_DIR / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    eval_idx = set(random.sample(range(len(ds)), N_EVAL))

    train_rows, val_rows, dropped, durations = [], [], 0, []
    for i, row in enumerate(ds):
        text = clean(row["text"])
        if not text or (set(text) - VOCAB_CHARS):
            dropped += 1
            continue

        audio = row["audio"]
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != TARGET_SR:
            g = np.gcd(int(sr), TARGET_SR)
            arr = resample_poly(arr, TARGET_SR // g, int(sr) // g).astype(np.float32)
        dur = len(arr) / TARGET_SR
        if dur < 1.0 or dur > 20.0:
            dropped += 1
            continue
        durations.append(dur)

        name = f"{row['filename']}.wav"
        dst = val_dir if i in eval_idx else train_dir
        sf.write(dst / name, arr, TARGET_SR, subtype="PCM_16")
        (val_rows if i in eval_idx else train_rows).append({"file_name": name, "text": text, "duration_s": dur})

    for dir_, rows in [(train_dir, train_rows), (val_dir, val_rows)]:
        with open(dir_ / "metadata.csv", "w", encoding="utf-8", newline="") as f:
            f.write("file_name,text\n")
            for r in rows:
                f.write(f'{r["file_name"]},"{r["text"].replace(chr(34), chr(34)*2)}"\n')

    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(train_rows + val_rows, f, ensure_ascii=False, indent=1)

    print(f"[iqbal] dropped {dropped} (out-of-vocab text or bad duration)")
    print(f"[done] train={len(train_rows)} eval={len(val_rows)} total={sum(durations)/60:.1f} min")


if __name__ == "__main__":
    main()

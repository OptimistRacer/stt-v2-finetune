"""Transcribe all 156 host clips with the fine-tuned Urdu STT model and lay them out
as a local HF `datasets` audiofolder (train/ + validation/ subdirs with metadata.csv),
ready for finetune-hf-vits.

IMPORTANT: deliberately transcribes WITHOUT the codeswitch prompt/dictionary. The
MMS-TTS Urdu vocab is Arabic-script Urdu characters only (58 tokens, no Latin
letters) -- the codeswitch convention used for the STT project (English/loan words
kept in Latin script) makes ~31% of characters in this podcast's transcripts
unrepresentable to the TTS tokenizer, silently dropped during training and badly
corrupting text-audio alignment. Plain transcription phoneticizes English words
into Urdu script instead (e.g. "subscribe" -> "سبسکرائب"), which the vocab can
fully represent.
"""
import json
import random
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\Sistems-AI\stt-v2-finetune")

from scripts.transcribe import transcribe

CLIPS_DIR = Path("D:/Audion-Data/TTS/host_clips")
OUT_DIR = Path("D:/Audion-Data/TTS/vits_dataset_v2")
N_EVAL = 12

# The model's actual character vocabulary (from the converted training checkpoint) --
# anything outside this is unrepresentable to the tokenizer and corrupts text-audio
# alignment if left in. Punctuation/diacritics not in vocab are harmless to strip;
# real Latin-script leftovers (full English clauses Whisper didn't phoneticize) mean
# the clip has to be dropped rather than silently truncated.
with open("D:/Audion-Data/TTS/vits_finetuned_host/vocab.json", encoding="utf-8") as _f:
    VOCAB_CHARS = set(json.load(_f).keys())
STRIP_CHARS = str.maketrans("", "", "،؟۔ًُ")

def main():
    clips = sorted(CLIPS_DIR.glob("*.wav"))
    print(f"[build] {len(clips)} clips to transcribe")

    records = []
    for i, clip in enumerate(clips):
        result = transcribe(
            str(clip), language="ur", device="cuda", compute_type="float16",
            beam_size=5, vad=False, initial_prompt=None, prompt_chunk_s=0,
        )
        text = result["text"]
        records.append({
            "file_name": clip.name,
            "text": text,
            "duration_s": result["duration_s"],
        })
        if (i + 1) % 20 == 0:
            print(f"[build] {i + 1}/{len(clips)} done")

    empty = [r for r in records if not r["text"].strip()]
    print(f"[build] {len(empty)} clips produced empty transcripts, dropping them")
    records = [r for r in records if r["text"].strip()]

    for r in records:
        r["text"] = r["text"].translate(STRIP_CHARS).strip()

    oov = [r for r in records if set(r["text"]) - VOCAB_CHARS]
    print(f"[build] {len(oov)} clips still have out-of-vocab characters "
          f"(untranslated English clauses), dropping them")
    records = [r for r in records if not (set(r["text"]) - VOCAB_CHARS)]

    total_kept_min = sum(r["duration_s"] for r in records) / 60
    print(f"[build] {len(records)} clips kept, {total_kept_min:.1f} min total")

    random.seed(42)
    shuffled = records[:]
    random.shuffle(shuffled)
    eval_names = {r["file_name"] for r in shuffled[:N_EVAL]}

    train_dir = OUT_DIR / "train"
    val_dir = OUT_DIR / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    train_rows, val_rows = [], []
    for r in records:
        dst_dir = val_dir if r["file_name"] in eval_names else train_dir
        shutil.copy(CLIPS_DIR / r["file_name"], dst_dir / r["file_name"])
        (val_rows if r["file_name"] in eval_names else train_rows).append(r)

    for dir_, rows in [(train_dir, train_rows), (val_dir, val_rows)]:
        with open(dir_ / "metadata.csv", "w", encoding="utf-8", newline="") as f:
            f.write("file_name,text\n")
            for r in rows:
                escaped = r["text"].replace('"', '""')
                f.write(f'{r["file_name"]},"{escaped}"\n')

    with open(OUT_DIR / "manifest_full.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)

    total_dur = sum(r["duration_s"] for r in records)
    print(f"[done] train={len(train_rows)} eval={len(val_rows)} total_duration={total_dur/60:.1f}min")
    print(f"[done] -> {OUT_DIR}")

if __name__ == "__main__":
    main()

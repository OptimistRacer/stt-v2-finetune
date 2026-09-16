"""Transcribe all 156 host clips with the fine-tuned Urdu STT model and lay them out
as a local HF `datasets` audiofolder (train/ + validation/ subdirs with metadata.csv),
ready for finetune-hf-vits.
"""
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\Sistems-AI\stt-v2-finetune")

from scripts.transcribe import transcribe, CODESWITCH_PROMPT
from src.postprocess.codeswitch import CodeSwitcher

CLIPS_DIR = Path("D:/Audion-Data/TTS/host_clips")
OUT_DIR = Path("D:/Audion-Data/TTS/vits_dataset")
N_EVAL = 12

def main():
    clips = sorted(CLIPS_DIR.glob("*.wav"))
    print(f"[build] {len(clips)} clips to transcribe")

    cs = CodeSwitcher()
    records = []
    for i, clip in enumerate(clips):
        result = transcribe(
            str(clip), language="ur", device="cuda", compute_type="float16",
            beam_size=5, vad=False, initial_prompt=CODESWITCH_PROMPT, prompt_chunk_s=0,
        )
        text = cs.apply(result["text"])
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

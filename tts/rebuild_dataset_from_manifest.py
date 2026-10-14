"""Reprocess the already-transcribed manifest (vits_dataset_v2/manifest_full.json)
with vocab-based filtering, without re-running Whisper. Strips harmless punctuation
not in the MMS-TTS vocab, then drops any clip that still has real out-of-vocab
characters (untranslated English clauses).
"""
import json
import random
import shutil
from pathlib import Path

CLIPS_DIR = Path("D:/Audion-Data/TTS/host_clips")
OUT_DIR = Path("D:/Audion-Data/TTS/vits_dataset_v2")
N_EVAL = 12

with open("D:/Audion-Data/TTS/vits_finetuned_host/vocab.json", encoding="utf-8") as f:
    VOCAB_CHARS = set(json.load(f).keys())
STRIP_CHARS = str.maketrans("", "", "،؟۔ًُ")

with open(OUT_DIR / "manifest_full.json", encoding="utf-8") as f:
    records = json.load(f)

for r in records:
    r["text"] = r["text"].translate(STRIP_CHARS).strip()

oov = [r for r in records if set(r["text"]) - VOCAB_CHARS]
print(f"[rebuild] {len(oov)}/{len(records)} clips still have out-of-vocab characters, dropping them")
records = [r for r in records if not (set(r["text"]) - VOCAB_CHARS) and r["text"]]

total_kept_min = sum(r["duration_s"] for r in records) / 60
print(f"[rebuild] {len(records)} clips kept, {total_kept_min:.1f} min total")

random.seed(42)
shuffled = records[:]
random.shuffle(shuffled)
eval_names = {r["file_name"] for r in shuffled[:N_EVAL]}

train_dir = OUT_DIR / "train"
val_dir = OUT_DIR / "validation"
shutil.rmtree(train_dir, ignore_errors=True)
shutil.rmtree(val_dir, ignore_errors=True)
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

with open(OUT_DIR / "manifest_filtered.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=1)

print(f"[done] train={len(train_rows)} eval={len(val_rows)}")

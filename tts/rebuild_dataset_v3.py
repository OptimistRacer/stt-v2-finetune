"""Re-transcribe all 156 host clips with a strong Urdu-biasing prompt (full of
already-transliterated loanwords) to push Whisper toward phoneticizing embedded
English into Urdu script instead of leaving it in Latin script -- recovering more
of the clips that got dropped in v2 for having out-of-vocab (untranslated English)
characters.
"""
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\Sistems-AI\stt-v2-finetune")
from scripts.transcribe import transcribe

CLIPS_DIR = Path("D:/Audion-Data/TTS/host_clips")
OUT_DIR = Path("D:/Audion-Data/TTS/vits_dataset_v3")
N_EVAL = 12

with open("D:/Audion-Data/TTS/vits_finetuned_host/vocab.json", encoding="utf-8") as f:
    VOCAB_CHARS = set(json.load(f).keys())
STRIP_CHARS = str.maketrans("", "", "،؟۔ًُ")

URDU_BIAS_PROMPT = (
    "یہ بزنس، ایجوکیشن اور ٹیکنالوجی کے بارے میں بات ہے۔ سجسٹ کرنا، بیک گراؤنڈ "
    "مارکیٹنگ، پری ریکوزٹس، اسٹارٹ اپ، چیلنج، کانفیڈنس، ایکسپیرینس اور اپورچیونٹی "
    "جیسے الفاظ استعمال ہوتے ہیں۔ آپ بزنس میں آسانی سے جا سکتے ہیں اور سبسکرائب "
    "کر سکتے ہیں۔"
)


def main():
    clips = sorted(CLIPS_DIR.glob("*.wav"))
    print(f"[build] {len(clips)} clips to transcribe")

    records = []
    for i, clip in enumerate(clips):
        result = transcribe(
            str(clip), language="ur", device="cuda", compute_type="float16",
            beam_size=5, vad=False, initial_prompt=URDU_BIAS_PROMPT, prompt_chunk_s=0,
        )
        text = result["text"].translate(STRIP_CHARS).strip()
        records.append({"file_name": clip.name, "text": text, "duration_s": result["duration_s"]})
        if (i + 1) % 20 == 0:
            print(f"[build] {i + 1}/{len(clips)} done")

    records = [r for r in records if r["text"]]
    oov = [r for r in records if set(r["text"]) - VOCAB_CHARS]
    print(f"[build] {len(oov)}/{len(records)} clips still have out-of-vocab characters, dropping them")
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

    print(f"[done] train={len(train_rows)} eval={len(val_rows)}")


if __name__ == "__main__":
    main()

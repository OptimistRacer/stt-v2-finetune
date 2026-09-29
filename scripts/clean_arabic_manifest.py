"""Strip the formal writing style from Common Voice Arabic training transcripts.

The first Arabic run (configs/day3_lora_ar_cv.yaml) improved read-aloud MSA (15.67 ->
12.40 WER on the CV holdout) but regressed every Gulf set (Saudi 17.61 -> 21.39, etc.).
Cause, from comparing outputs: 23% of CV Arabic training sentences are fully
diacritized (tashkeel), and the model learned to switch into that formal mode --
diacritized, MSA-leaning output that mis-hears dialect ("مكان الذهب" -> "مَا كَانَتْ ذَهَبُ").
Everyday Arabic text, the Gulf test transcripts and the app are all undiacritized.

So: remove diacritics and tatweel, and re-attach a standalone "و" to the next word
("و كيف" -> "وكيف", 1% of sentences; it is written joined in normal Arabic). Nothing
else changes, so a difference in results is attributable to this.

    python scripts/clean_arabic_manifest.py
"""
import json
import re
from pathlib import Path

SRC = Path(r"D:\Audion-Data\Arabic\open_datasets\manifest_common_voice_ar.jsonl")
DST = Path(r"D:\Audion-Data\Arabic\open_datasets\manifest_common_voice_ar_clean.jsonl")
DIACRITICS = re.compile(r"[\u064B-\u0652\u0670\u0640]")  # harakat, superscript alef, tatweel
LONE_WA = re.compile(r"(^|\s)و\s+(?=\S)")


def clean(text: str) -> str:
    text = DIACRITICS.sub("", text)
    text = LONE_WA.sub(r"\1و", text)
    return re.sub(r"\s+", " ", text).strip()


def main():
    changed = total = 0
    with open(SRC, encoding="utf-8") as src, open(DST, "w", encoding="utf-8") as dst:
        for line in src:
            rec = json.loads(line)
            if "draft_transcript" in rec:
                new = clean(rec["draft_transcript"])
                total += 1
                changed += new != rec["draft_transcript"]
                rec["draft_transcript"] = new
            dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[clean] {changed}/{total} transcripts changed -> {DST}")


if __name__ == "__main__":
    main()

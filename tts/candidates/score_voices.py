"""Score every voice in out/<model>/ by what a recognizer hears back, and build a blind
listening page.

Clarity judge: Urdu -> our served Urdu model (stt-v2-finetune scripts/transcribe.py
DEFAULT_MODEL); Arabic -> base large-v3, not our Arabic fine-tune, so the judge doesn't
favour any voice. Whisper fills in unclear words from context, so this score is
generous; the listening test is the real decision.

Writes results.json and listen.html (voices shuffled and labelled A, B, C...; the key is
in listen_key.json, so the listener doesn't see which model is which).

    python score_voices.py        (main Python environment)
"""
import html
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

import jiwer
from faster_whisper import WhisperModel

HERE = Path(__file__).parent
sys.path.insert(0, r"C:\Users\Sistems-AI\stt-v2-finetune")
from scripts.transcribe import DEFAULT_MODEL as URDU_STT  # noqa: E402

SENTENCES = json.loads((HERE / "sentences.json").read_text(encoding="utf-8"))
JUDGES = {"ur": URDU_STT, "ar": "large-v3"}
DIACRITICS = re.compile(r"[\u064B-\u0652\u0670]")
PUNCT = re.compile(r"[،۔.؟?!:;\"«»()\-_,]")


def normalize(text, lang):
    text = DIACRITICS.sub("", unicodedata.normalize("NFC", text or ""))
    text = PUNCT.sub(" ", text)
    if lang == "ar":
        text = re.sub("[أإآٱ]", "ا", text).replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    return re.sub(r"\s+", " ", text).strip()


def main():
    models = sorted(p.name for p in (HERE / "out").iterdir() if p.is_dir())
    results = {}
    for lang, judge_path in JUDGES.items():
        judge = WhisperModel(judge_path, device="cuda", compute_type="float16")
        for model in models:
            refs, hyps, clips = [], [], []
            for i, text in enumerate(SENTENCES[lang]):
                wav = HERE / "out" / model / f"{lang}_{i}.wav"
                if not wav.exists():
                    continue
                segs, _ = judge.transcribe(str(wav), language=lang, beam_size=5, vad_filter=True)
                heard = " ".join(s.text.strip() for s in segs)
                r, h = normalize(text, lang), normalize(heard, lang)
                refs.append(r); hyps.append(h)
                clips.append({"clip": wav.name, "heard": heard, "wer": round(100 * jiwer.wer(r, h), 1)})
            if refs:
                results.setdefault(lang, {})[model] = {"wer": round(100 * jiwer.wer(refs, hyps), 1),
                                                        "cer": round(100 * jiwer.cer(refs, hyps), 1),
                                                        "clips": clips}
                print(f"[score] {lang} {model:12s} WER {results[lang][model]['wer']:5.1f}%  "
                      f"CER {results[lang][model]['cer']:5.1f}%", flush=True)
        del judge

    (HERE / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    build_listening_page(results)


def build_listening_page(results):
    rng = random.Random(7)
    key, sections = {}, []
    for lang, by_model in results.items():
        models = list(by_model)
        rng.shuffle(models)
        labels = {m: chr(ord("A") + k) for k, m in enumerate(models)}
        key[lang] = {labels[m]: m for m in models}
        rows = []
        for i, text in enumerate(SENTENCES[lang]):
            cells = "".join(
                f'<div class="v"><b>{labels[m]}</b><audio controls preload="none" '
                f'src="out/{m}/{lang}_{i}.wav"></audio></div>'
                for m in sorted(models, key=lambda m: labels[m]))
            rows.append(f'<div class="s"><p dir="rtl">{html.escape(text)}</p>{cells}</div>')
        title = "Urdu" if lang == "ur" else "Arabic"
        sections.append(f"<h2>{title}</h2>" + "".join(rows))
    page = ("<!doctype html><meta charset=utf-8><title>Voice listening test</title>"
            "<style>body{font-family:system-ui;max-width:900px;margin:24px auto;padding:0 16px}"
            ".s{border-top:1px solid #ccc;padding:12px 0}.s p{font-size:1.3em;margin:4px 0 8px}"
            ".v{display:inline-flex;align-items:center;gap:6px;margin:4px 16px 4px 0}</style>"
            "<h1>Voice listening test</h1><p>For each language, listen to every sentence and note "
            "which letter sounds clearest and most natural. The letters are shuffled; the key is "
            "kept separately.</p>" + "".join(sections))
    (HERE / "listen.html").write_text(page, encoding="utf-8")
    (HERE / "listen_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(f"[score] listening page -> {HERE / 'listen.html'}", flush=True)


if __name__ == "__main__":
    main()

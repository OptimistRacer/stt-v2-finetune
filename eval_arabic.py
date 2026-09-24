"""Score transcription accuracy on a folder of wav + txt pairs.

Written for Gulf Arabic, but works for any language faster-whisper supports: point it
at a directory where each `foo.wav` has a `foo.txt` holding the correct transcript.

    python eval_arabic.py --dir gulf_sample --language ar
    python eval_arabic.py --dir gulf_sample --language ar --prompt "شلون حالك؟ وش تبغى؟"

Arabic orthography varies without changing the words (أ/ا, ى/ي, optional diacritics),
so text is normalised before scoring -- otherwise spelling variants inflate the error
rate and hide the real mistakes.
"""
import argparse
import glob
import json
import os
import re
import unicodedata

import jiwer
from faster_whisper import WhisperModel

DIACRITICS = re.compile(r"[ً-ْٰ]")
PUNCT = re.compile(r"[،.؟?!:;\"«»()\-_]")


def normalize(text: str, language: str = "ar") -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = DIACRITICS.sub("", text)
    text = PUNCT.sub(" ", text)
    if language == "ar":
        text = re.sub("[أإآٱ]", "ا", text).replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    return re.sub(r"\s+", " ", text).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="folder of foo.wav + foo.txt pairs")
    ap.add_argument("--language", default="ar")
    ap.add_argument("--model", default="large-v3", help="model size or a local CT2 export")
    ap.add_argument("--prompt", default=None, help="initial_prompt, e.g. dialect phrasing")
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--no_vad", action="store_true")
    ap.add_argument("--out", default=None, help="write per-clip results as JSON")
    args = ap.parse_args()

    pairs = []
    for wav in sorted(glob.glob(os.path.join(args.dir, "*.wav"))):
        txt = wav[:-4] + ".txt"
        if os.path.exists(txt):
            pairs.append((wav, open(txt, encoding="utf-8").read().strip()))
    if not pairs:
        raise SystemExit(f"no wav+txt pairs found in {args.dir}")
    print(f"[eval] {len(pairs)} clips | language={args.language} | model={args.model} "
          f"| prompt={'yes' if args.prompt else 'no'}")

    model = WhisperModel(args.model, device="cuda", compute_type="float16")
    refs, hyps, rows = [], [], []
    for wav, ref in pairs:
        segments, _ = model.transcribe(wav, language=args.language, beam_size=args.beam_size,
                                       vad_filter=not args.no_vad, initial_prompt=args.prompt)
        hyp = " ".join(s.text.strip() for s in segments).strip()
        r, h = normalize(ref, args.language), normalize(hyp, args.language)
        refs.append(r); hyps.append(h)
        rows.append({"clip": os.path.basename(wav), "reference": ref, "heard": hyp,
                     "wer": round(100 * jiwer.wer(r, h), 1)})
        print(f"  {rows[-1]['clip']:24s} WER {rows[-1]['wer']:5.1f}%")

    summary = {"n": len(refs), "wer": round(100 * jiwer.wer(refs, hyps), 2),
               "cer": round(100 * jiwer.cer(refs, hyps), 2),
               "language": args.language, "model": args.model, "prompt": args.prompt}
    print(f"[overall] WER {summary['wer']}%  CER {summary['cer']}%")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "clips": rows}, f, ensure_ascii=False, indent=2)
        print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()

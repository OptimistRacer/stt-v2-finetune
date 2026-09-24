"""Turn Urdu text into speech.

    python synthesize.py "آپ کا نام کیا ہے؟" -o hello.wav
    python synthesize.py --file script.txt -o demo.wav

Uses D:/Audion-Data/TTS/urdu_tts_final, which is Meta's MMS Urdu voice with the
playback settings that measured clearest (noise_scale 0.33, noise_scale_duration 0.5,
speaking_rate 0.9 -- 0.085 word error vs 0.132 at the defaults). Those are baked into
the model folder, so nothing needs to be passed at runtime.

Note: the MMS checkpoint is CC-BY-NC 4.0, i.e. non-commercial use.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import torch
from transformers import pipeline

from text_normalize import describe, has_sayable, normalize_for_tts

MODEL = "D:/Audion-Data/TTS/urdu_tts_final"
PAUSE_S = 0.28


def split_sentences(text):
    """The model was trained on single sentences and degrades on long input: on an
    89-word paragraph, synthesizing sentence by sentence halved the word error
    (0.135 -> 0.067) versus feeding the whole thing in at once."""
    parts = [p.strip() for p in re.split(r"(?<=[۔?!])\s*|\n+", text) if p and p.strip()]
    parts = parts or [text]
    # A chunk the voice can't say anything for (e.g. a stray all-English line) becomes
    # an empty token sequence and crashes VITS, so drop those here.
    return [p for p in parts if has_sayable(p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help="Urdu text to speak")
    ap.add_argument("--file", help="read the text from a UTF-8 file instead")
    ap.add_argument("-o", "--out", default="output.wav")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--seed", type=int, default=1234, help="same seed gives the same audio")
    ap.add_argument("--no-split", action="store_true",
                    help="feed the text in as one piece instead of sentence by sentence")
    ap.add_argument("--no-normalize", action="store_true",
                    help="skip spelling out numbers and converting English words")
    args = ap.parse_args()

    text = Path(args.file).read_text(encoding="utf-8").strip() if args.file else args.text
    if not text:
        raise SystemExit("give text as an argument or use --file")

    if not args.no_normalize:
        text, report = normalize_for_tts(text)
        summary = describe(report)
        if summary:
            print(f"[text] {summary}")

    tts = pipeline("text-to-speech", args.model, device=0 if torch.cuda.is_available() else -1)
    chunks = [text] if args.no_split else split_sentences(text)

    pieces, sampling_rate = [], None
    for i, chunk in enumerate(chunks):
        torch.manual_seed(args.seed + i)
        out = tts(chunk)
        sampling_rate = out["sampling_rate"]
        pieces.append(out["audio"].squeeze().astype(np.float32))
        if i < len(chunks) - 1:
            pieces.append(np.zeros(int(PAUSE_S * sampling_rate), dtype=np.float32))

    audio = np.concatenate(pieces)
    scipy.io.wavfile.write(args.out, rate=sampling_rate, data=audio)
    print(f"[tts] {len(chunks)} sentence(s), {len(audio) / sampling_rate:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()

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
from pathlib import Path

import scipy.io.wavfile
import torch
from transformers import pipeline

MODEL = "D:/Audion-Data/TTS/urdu_tts_final"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help="Urdu text to speak")
    ap.add_argument("--file", help="read the text from a UTF-8 file instead")
    ap.add_argument("-o", "--out", default="output.wav")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--seed", type=int, default=1234, help="same seed gives the same audio")
    args = ap.parse_args()

    text = Path(args.file).read_text(encoding="utf-8").strip() if args.file else args.text
    if not text:
        raise SystemExit("give text as an argument or use --file")

    tts = pipeline("text-to-speech", args.model, device=0 if torch.cuda.is_available() else -1)
    torch.manual_seed(args.seed)
    out = tts(text)
    audio = out["audio"].squeeze()
    scipy.io.wavfile.write(args.out, rate=out["sampling_rate"], data=audio)
    print(f"[tts] {len(audio) / out['sampling_rate']:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()

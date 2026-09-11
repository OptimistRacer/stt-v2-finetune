"""Inference entry point: transcribe an audio file with the exported model.

This is the deployable surface -- what a caller actually uses. Everything upstream
(Day 1-4) works on pre-cut ~30s clips, but real input is a whole recording, so this
hands the file to faster-whisper and lets it do its own VAD segmentation rather than
requiring the caller to pre-chunk anything.

Outputs text, JSON (segments with timestamps), or SRT subtitles.

Usage:
    python scripts/transcribe.py audio.mp3
    python scripts/transcribe.py audio.mp3 --format srt --out subs.srt
    python scripts/transcribe.py audio.mp3 --format json --vad
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faster_whisper import WhisperModel

DEFAULT_MODEL = r"D:\Audion-Data\Urdu\checkpoints\stt_v2_lora_augmented\export\ct2"


def _ts(seconds: float, sep: str = ",") -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(segments: list[dict]) -> str:
    out = []
    for i, seg in enumerate(segments, 1):
        out.append(f"{i}\n{_ts(seg['start'])} --> {_ts(seg['end'])}\n{seg['text'].strip()}\n")
    return "\n".join(out)


def transcribe(
    audio_path: str,
    model_dir: str = DEFAULT_MODEL,
    language: str = "ur",
    device: str = "cuda",
    compute_type: str = "float16",
    beam_size: int = 5,
    vad: bool = False,
) -> dict:
    if not Path(model_dir).exists():
        raise FileNotFoundError(f"No model at {model_dir} -- run scripts/run_day5_export.py first")

    model = WhisperModel(model_dir, device=device, compute_type=compute_type)
    t0 = time.time()
    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=beam_size,
        vad_filter=vad,
    )
    # faster-whisper yields lazily; materializing here is what actually runs decoding.
    segs = [
        {"start": s.start, "end": s.end, "text": s.text.strip()}
        for s in segments
    ]
    elapsed = time.time() - t0
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    return {
        "audio": str(audio_path),
        "language": getattr(info, "language", language),
        "duration_s": round(duration, 2),
        "elapsed_s": round(elapsed, 2),
        "rtf": round(elapsed / duration, 4) if duration else None,
        "n_segments": len(segs),
        "text": " ".join(s["text"] for s in segs).strip(),
        "segments": segs,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("--model_dir", default=DEFAULT_MODEL)
    p.add_argument("--language", default="ur")
    p.add_argument("--device", default="cuda")
    p.add_argument("--compute_type", default="float16")
    p.add_argument("--beam_size", type=int, default=5)
    p.add_argument("--vad", action="store_true", help="Drop non-speech with VAD before decoding")
    p.add_argument("--format", default="text", choices=["text", "json", "srt"])
    p.add_argument("--out", default=None, help="Write to a file instead of stdout")
    a = p.parse_args()

    result = transcribe(a.audio, a.model_dir, a.language, a.device, a.compute_type, a.beam_size, a.vad)

    if a.format == "text":
        body = result["text"]
    elif a.format == "json":
        body = json.dumps(result, ensure_ascii=False, indent=2)
    else:
        body = to_srt(result["segments"])

    if a.out:
        Path(a.out).write_text(body, encoding="utf-8")
        print(f"[transcribe] {result['duration_s']}s audio in {result['elapsed_s']}s "
              f"(RTF {result['rtf']}), {result['n_segments']} segments -> {a.out}")
    else:
        # stdout on Windows is cp1252 and will die on Urdu; write bytes directly.
        sys.stdout.buffer.write(body.encode("utf-8"))
        sys.stdout.buffer.write(b"\n")

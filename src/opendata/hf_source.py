"""Materialize a Hugging Face speech dataset split as local 16kHz mono WAV clips.

Common Voice and FLEURS arrive pre-segmented (one utterance per example) and
pre-transcribed with ground-truth text, read in a flat/neutral tone -- unlike the
podcast pipeline in `run_day1_prep.py` / `run_day1_gpu.py`, there's no denoise,
diarization, or draft-transcription step needed here.

Audio is decoded manually with soundfile/librosa rather than via `datasets`'
`Audio(sampling_rate=...)` cast: that cast requires the `torchcodec` package (pulling
in `torch`) in current `datasets` versions, which conflicts with keeping heavy ML deps
off the local machine (see README) -- decoding is cast with `decode=False` instead,
which just hands back raw bytes/a file path.
"""
import io
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, load_dataset


def load_split(hf_name: str, hf_config: str, split: str) -> Dataset:
    ds = load_dataset(hf_name, hf_config, split=split)
    return ds.cast_column("audio", Audio(decode=False))


def transcript_of(example: dict) -> str:
    # Common Voice uses "sentence", FLEURS uses "transcription"
    return example.get("sentence") or example.get("transcription") or ""


def speaker_of(example: dict) -> str | None:
    return example.get("client_id") or example.get("speaker_id")


def _read_raw_audio(audio_field: dict) -> tuple[np.ndarray, int]:
    if audio_field.get("bytes"):
        data, sr = sf.read(io.BytesIO(audio_field["bytes"]), dtype="float32")
    else:
        data, sr = sf.read(audio_field["path"], dtype="float32")
    if data.ndim > 1:  # stereo -> mono
        data = data.mean(axis=1)
    return data, sr


def save_clip(example: dict, out_path: Path, sample_rate: int) -> float:
    """Decode `example`'s audio to `out_path` as mono WAV at `sample_rate`. Returns duration in seconds."""
    data, sr = _read_raw_audio(example["audio"])
    if sr != sample_rate:
        data = librosa.resample(data, orig_sr=sr, target_sr=sample_rate)
    sf.write(out_path, data, sample_rate)
    return round(len(data) / sample_rate, 3)


def duration_of(path: Path) -> float:
    """Read duration from an already-materialized clip, no decode needed -- used to skip
    re-downloading/re-saving on a resumed run."""
    return round(sf.info(str(path)).duration, 3)

import os

import soundfile as sf
import torch
from huggingface_hub import get_token
from pyannote.audio import Pipeline


def load_pipeline(hf_token: str | None = None):
    """Load the pretrained speaker-diarization pipeline. Requires a HuggingFace token
    that has accepted the pyannote/speaker-diarization-3.1 model terms.

    Token resolution order: explicit arg, HF_TOKEN env var, then the token saved by
    `hf auth login` (huggingface_hub.get_token()) -- so a prior CLI login is enough
    and you don't have to also export HF_TOKEN."""
    token = hf_token or os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError("No HF token. Run `hf auth login`, set HF_TOKEN, or pass hf_token.")
    return Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)


def diarize_file(pipeline, audio_path: str) -> list[dict]:
    """Diarize `audio_path`. Audio is decoded here with soundfile and passed to the
    pipeline as an in-memory waveform dict rather than a file path: pyannote.audio 4.x
    decodes file paths via torchcodec, which needs FFmpeg's shared libraries on the
    system, and those aren't reliably present on Windows (only the ffmpeg executable
    is). Since the Day-1 prep step already wrote clean 16kHz mono WAVs, soundfile reads
    them with no FFmpeg dependency, sidestepping torchcodec entirely."""
    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim == 1:
        data = data[None, :]  # (time,) -> (channel=1, time)
    else:
        data = data.T  # (time, channel) -> (channel, time)
    waveform = torch.from_numpy(data)
    output = pipeline({"waveform": waveform, "sample_rate": sr})
    # pyannote.audio 4.x returns a DiarizeOutput wrapper; the Annotation with
    # .itertracks() is its .speaker_diarization attribute (3.x returned the Annotation
    # directly). Support both so this doesn't silently break on a version change.
    annotation = getattr(output, "speaker_diarization", output)
    return [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]


def assign_speaker(turns: list[dict], clip_start_s: float, clip_end_s: float) -> str | None:
    """Assign the speaker whose diarization turn has the most overlap with the clip window."""
    best_speaker, best_overlap = None, 0.0
    for t in turns:
        overlap = min(t["end"], clip_end_s) - max(t["start"], clip_start_s)
        if overlap > best_overlap:
            best_overlap, best_speaker = overlap, t["speaker"]
    return best_speaker

import os

from pyannote.audio import Pipeline


def load_pipeline(hf_token: str | None = None):
    """Load the pretrained speaker-diarization pipeline. Requires a HuggingFace token
    that has accepted the pyannote/speaker-diarization-3.1 model terms."""
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN env var or pass hf_token explicitly.")
    return Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)


def diarize_file(pipeline, audio_path: str) -> list[dict]:
    diarization = pipeline(audio_path)
    return [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]


def assign_speaker(turns: list[dict], clip_start_s: float, clip_end_s: float) -> str | None:
    """Assign the speaker whose diarization turn has the most overlap with the clip window."""
    best_speaker, best_overlap = None, 0.0
    for t in turns:
        overlap = min(t["end"], clip_end_s) - max(t["start"], clip_start_s)
        if overlap > best_overlap:
            best_overlap, best_speaker = overlap, t["speaker"]
    return best_speaker

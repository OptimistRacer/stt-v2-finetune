from pathlib import Path

from pydub import AudioSegment
from pydub.silence import detect_nonsilent


def find_speech_regions(
    audio: AudioSegment,
    silence_thresh_db: float = -40.0,
    min_silence_ms: int = 400,
) -> list[tuple[int, int]]:
    """Return (start_ms, end_ms) spans of non-silent audio, split on silence gaps."""
    return detect_nonsilent(audio, min_silence_len=min_silence_ms, silence_thresh=silence_thresh_db)


def pack_into_clips(
    regions: list[tuple[int, int]],
    target_max_ms: int = 30000,
    floor_ms: int = 2000,
) -> list[tuple[int, int]]:
    """Greedily merge adjacent speech regions up to target_max_ms, dropping anything under floor_ms.

    A clip boundary only ever falls on a detected silence gap, never mid-word or
    mid-sentence -- so a single unbroken region longer than target_max_ms (someone
    talking for 40s with no pause) is kept whole as one over-length clip rather than
    force-cut. That's a rare case; a slightly long clip is preferable to a broken
    transcript-audio alignment.
    """
    clips: list[tuple[int, int]] = []
    cur_start: int | None = None
    cur_end: int | None = None
    for start, end in regions:
        if cur_start is None:
            cur_start, cur_end = start, end
            continue
        if end - cur_start <= target_max_ms:
            cur_end = end
        else:
            if cur_end - cur_start >= floor_ms:
                clips.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    if cur_start is not None and cur_end - cur_start >= floor_ms:
        clips.append((cur_start, cur_end))
    return clips


def export_clips(audio: AudioSegment, clips: list[tuple[int, int]], out_dir: Path, source_id: str) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i, (start, end) in enumerate(clips):
        clip = audio[start:end]
        filename = f"{source_id}_clip{i:04d}.wav"
        path = out_dir / filename
        clip.export(path, format="wav")
        records.append({
            "clip_id": f"{source_id}_clip{i:04d}",
            "path": str(path),
            "start_ms": start,
            "end_ms": end,
            "duration_s": round((end - start) / 1000, 3),
        })
    return records

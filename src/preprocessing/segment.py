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


def _split_long_region(start: int, end: int, target_max_ms: int) -> list[tuple[int, int]]:
    """A single unbroken speech region can itself exceed target_max_ms (e.g. no silence
    gap for 40s of continuous talking). Hard-split it into target_max_ms-sized pieces --
    this is the one case where a clip boundary can fall mid-word, since there is no
    silence gap available to split on."""
    if end - start <= target_max_ms:
        return [(start, end)]
    pieces = []
    cur = start
    while cur < end:
        piece_end = min(cur + target_max_ms, end)
        pieces.append((cur, piece_end))
        cur = piece_end
    return pieces


def pack_into_clips(
    regions: list[tuple[int, int]],
    target_max_ms: int = 30000,
    floor_ms: int = 2000,
) -> list[tuple[int, int]]:
    """Greedily merge adjacent speech regions up to target_max_ms, dropping anything under floor_ms.

    Merging (rather than cutting at a fixed duration) keeps clip boundaries on silence gaps,
    so no clip is cut mid-word, except for the rare region that alone exceeds target_max_ms.
    """
    normalized_regions = []
    for start, end in regions:
        normalized_regions.extend(_split_long_region(start, end, target_max_ms))

    clips: list[tuple[int, int]] = []
    cur_start: int | None = None
    cur_end: int | None = None
    for start, end in normalized_regions:
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

"""Diarize the 5 podcast episodes to find the consistent single speaker (the host) for
XTTS voice-cloning, since the source manifest labeled everything SPEAKER_0 with
diarization disabled.

For each episode: run pyannote diarization, get real per-speaker turns, and extract a
voice embedding per local speaker label (mean of pyannote's own embedding model over
that speaker's segments) so speakers can be compared *across* episodes -- pyannote's
labels (SPEAKER_00, SPEAKER_01...) are local to each file and don't correspond to the
same person between files on their own.

Output: speakers.json with, per episode, each local speaker's total talk time and an
embedding; and a cross-episode similarity matrix to identify which speaker cluster
recurs across all 5 (the host).
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pyannote.audio import Model, Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from huggingface_hub import get_token

RAW_DIR = Path("D:/Audion-Data/TTS/raw_episodes")
OUT_PATH = Path("D:/Audion-Data/TTS/speakers.json")


def load_waveform(path: Path, sr: int = 16000):
    data, file_sr = sf.read(path, dtype="float32")
    assert file_sr == sr, f"{path} is {file_sr}Hz, expected {sr}"
    if data.ndim > 1:
        data = data.mean(axis=1)
    return torch.from_numpy(data)[None, :], sr


def main():
    token = get_token()
    print("[diarize] loading pyannote diarization pipeline...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(torch.device(device))

    print("[diarize] loading speaker embedding model...")
    # wespeaker, not pyannote/embedding: the latter needs its own separate gate accepted
    # on top of speaker-diarization-3.1's, and wespeaker is fully open.
    embed_model = PretrainedSpeakerEmbedding(
        "pyannote/wespeaker-voxceleb-resnet34-LM", device=torch.device(device), token=token
    )

    wavs = sorted(RAW_DIR.glob("*.wav"))
    print(f"[diarize] {len(wavs)} episodes to process")

    results = {}
    for wav_path in wavs:
        print(f"\n[diarize] === {wav_path.name} ===")
        waveform, sr = load_waveform(wav_path)
        audio_dict = {"waveform": waveform, "sample_rate": sr}

        diarization = pipeline(audio_dict)
        annotation = getattr(diarization, "speaker_diarization", diarization)

        turns = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            turns.append({"start": turn.start, "end": turn.end, "speaker": speaker})
        print(f"  {len(turns)} turns, {len(set(t['speaker'] for t in turns))} local speakers")

        # Total talk time and an embedding per local speaker label.
        by_speaker = {}
        for t in turns:
            by_speaker.setdefault(t["speaker"], []).append((t["start"], t["end"]))

        speaker_stats = {}
        np_wave = waveform[0].numpy()
        for spk, spans in by_speaker.items():
            total = sum(e - s for s, e in spans)
            # Embed from the speaker's longest few turns (short turns give noisy embeddings).
            spans_sorted = sorted(spans, key=lambda x: -(x[1] - x[0]))[:8]
            chunks = []
            for s, e in spans_sorted:
                if e - s < 1.0:
                    continue
                seg = np_wave[int(s * sr):int(e * sr)]
                if len(seg) > sr * 0.5:
                    chunks.append(torch.from_numpy(seg)[None, None, :])
            emb = None
            if chunks:
                embs = [embed_model(c) for c in chunks]
                emb = np.mean(np.stack(embs), axis=0).tolist()
            speaker_stats[spk] = {"total_talk_s": round(total, 1), "n_turns": len(spans), "embedding": emb}
            print(f"    {spk}: {total:.1f}s talk time across {len(spans)} turns")

        results[wav_path.name] = {
            "duration_s": len(np_wave) / sr,
            "turns": turns,
            "speakers": speaker_stats,
        }

        # Free VRAM between long episodes.
        del waveform, np_wave
        torch.cuda.empty_cache()

    OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[done] -> {OUT_PATH}")


if __name__ == "__main__":
    main()

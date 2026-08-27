"""Day 1, GPU part: speaker diarization + draft transcription (Whisper base).
Run this in Colab, after run_day1_prep.py has produced outputs/manifest.jsonl
and outputs/clean/*.wav (upload/sync the outputs/ folder to Drive first).

Usage:
    python scripts/run_day1_gpu.py --config configs/day1.yaml --hf-token <token>
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.diarization.diarize import assign_speaker, diarize_file, load_pipeline
from src.transcription.draft_transcribe import load_model, transcribe_clip


def run(config_path: str, hf_token: str | None) -> None:
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    out_dir = Path(cfg["output_dir"])
    manifest_in = out_dir / "manifest.jsonl"
    manifest_out = out_dir / "manifest_day1_complete.jsonl"

    records = [json.loads(line) for line in open(manifest_in, encoding="utf-8")]

    dia_pipeline = load_pipeline(hf_token)
    whisper_model = load_model(cfg.get("whisper_model", "base"), device=cfg.get("device", "cuda"))

    turns_cache: dict[str, list[dict]] = {}
    for rec in records:
        source_id = rec["source_id"]
        if source_id not in turns_cache:
            clean_path = out_dir / "clean" / f"{source_id}.wav"
            print(f"[diarize] {source_id}")
            turns_cache[source_id] = diarize_file(dia_pipeline, str(clean_path))

        rec["speaker"] = assign_speaker(turns_cache[source_id], rec["start_ms"] / 1000, rec["end_ms"] / 1000)
        rec["draft_transcript"] = transcribe_clip(whisper_model, rec["path"], language=cfg.get("language", "ur"))
        rec["verified"] = False

    with open(manifest_out, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[done] -> {manifest_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/day1.yaml")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()
    run(args.config, args.hf_token)

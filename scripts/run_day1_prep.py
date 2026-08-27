"""Day 1, CPU part: denoise, normalize, segment into clips, write manifest skeleton.
Runs anywhere with ffmpeg installed -- no GPU needed. Run this locally first.

Usage:
    python scripts/run_day1_prep.py --config configs/day1.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from pydub import AudioSegment

from src.manifest.build_manifest import sha256_of, write_manifest
from src.preprocessing.audio_prep import resample_and_normalize
from src.preprocessing.segment import export_clips, find_speech_regions, pack_into_clips


def run(config_path: str) -> None:
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    data_dir = Path(cfg["data_dir"])
    out_dir = Path(cfg["output_dir"])

    all_records = []
    for source in cfg["sources"]:
        src_path = data_dir / source["filename"]
        clean_path = out_dir / "clean" / f"{source['id']}.wav"
        print(f"[prep] {src_path.name}")
        resample_and_normalize(src_path, clean_path, sample_rate=cfg.get("sample_rate", 16000))

        audio = AudioSegment.from_wav(clean_path)
        regions = find_speech_regions(
            audio,
            silence_thresh_db=cfg.get("silence_thresh_db", -40.0),
            min_silence_ms=cfg.get("min_silence_ms", 400),
        )
        clips = pack_into_clips(
            regions,
            target_max_ms=cfg.get("target_max_ms", 30000),
            floor_ms=cfg.get("floor_ms", 2000),
        )
        clip_dir = out_dir / "clips" / source["id"]
        clip_records = export_clips(audio, clips, clip_dir, source["id"])

        for rec in clip_records:
            rec["source_id"] = source["id"]
            rec["source_file"] = source["filename"]
            rec["accent"] = None
            rec["speaker"] = None
            rec["verified"] = False
            rec["checksum_sha256"] = sha256_of(Path(rec["path"]))
        all_records.extend(clip_records)
        print(f"  -> {len(clip_records)} clips")

    manifest_path = out_dir / "manifest.jsonl"
    write_manifest(all_records, manifest_path, dataset_version=cfg.get("dataset_version", "v2-day1"))
    print(f"[done] manifest -> {manifest_path} ({len(all_records)} clips total)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/day1.yaml")
    args = parser.parse_args()
    run(args.config)

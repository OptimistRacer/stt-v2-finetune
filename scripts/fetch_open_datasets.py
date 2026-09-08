"""Fetch open, pre-transcribed, neutral-emotion Urdu speech (Common Voice, FLEURS) and
fold it into the same manifest shape `run_day1_prep.py` produces for podcast audio, so
both can feed the same Day 3 fine-tuning step. CPU-only; no GPU needed.

Usage:
    python scripts/fetch_open_datasets.py --config configs/opendata_urdu.yaml
    python scripts/fetch_open_datasets.py --config configs/opendata_urdu.yaml --only fleurs_ur
    python scripts/fetch_open_datasets.py --config configs/opendata_urdu.yaml --limit 50   # smoke test
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.manifest.build_manifest import sha256_of, write_manifest
from src.opendata.hf_source import duration_of, load_split, save_clip, speaker_of, transcript_of


def run(config_path: str, only: str | None, limit: int | None) -> None:
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    out_dir = Path(cfg["output_dir"])
    sample_rate = cfg.get("sample_rate", 16000)

    all_records = []
    for ds_cfg in cfg["datasets"]:
        if only and ds_cfg["id"] != only:
            continue
        print(f"[opendata] {ds_cfg['id']} ({ds_cfg['hf_name']}/{ds_cfg['hf_config']})")
        clip_dir = out_dir / "clips" / ds_cfg["id"]
        clip_dir.mkdir(parents=True, exist_ok=True)

        for split in ds_cfg["splits"]:
            split_ds = load_split(ds_cfg["hf_name"], ds_cfg["hf_config"], split)
            n = len(split_ds) if not limit else min(limit, len(split_ds))
            print(f"  {split}: {n} clips")

            for i in range(n):
                clip_id = f"{ds_cfg['id']}_{split}_{i:06d}"
                clip_path = clip_dir / f"{clip_id}.wav"
                example = split_ds[i]
                if clip_path.exists():
                    duration_s = duration_of(clip_path)
                else:
                    duration_s = save_clip(example, clip_path, sample_rate)

                all_records.append({
                    "clip_id": clip_id,
                    "path": str(clip_path),
                    "start_ms": 0,
                    "end_ms": int(duration_s * 1000),
                    "duration_s": duration_s,
                    "source_id": ds_cfg["id"],
                    "source_file": f"{ds_cfg['hf_name']}/{ds_cfg['hf_config']}:{split}",
                    "speaker": speaker_of(example),
                    "accent": None,
                    "emotion": "neutral",
                    "verified": True,
                    "draft_transcript": transcript_of(example),
                    "license": ds_cfg.get("license"),
                    "checksum_sha256": sha256_of(clip_path),
                })

    manifest_path = out_dir / "manifest_opendata.jsonl"
    write_manifest(all_records, manifest_path, dataset_version=cfg.get("dataset_version", "v2-opendata-neutral"))
    print(f"[done] manifest -> {manifest_path} ({len(all_records)} clips total)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/opendata_urdu.yaml")
    parser.add_argument("--only", default=None, help="Fetch only this dataset id from the config")
    parser.add_argument("--limit", type=int, default=None, help="Cap clips per split (smoke test)")
    args = parser.parse_args()
    run(args.config, args.only, args.limit)

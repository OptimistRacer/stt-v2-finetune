"""Fetch Common Voice Urdu from the CC0 mirror and write it in the project's manifest shape.

Why a separate fetcher: Mozilla moved Common Voice off Hugging Face in October 2025
(`mozilla-foundation/common_voice_17_0`, which configs/opendata_urdu.yaml points at, now
holds only a README pointing to Mozilla Data Collective). `fsicoli/common_voice_19_0`
is a CC0 mirror, but it ships raw tar archives + TSVs rather than a loadable dataset,
so scripts/fetch_open_datasets.py can't read it.

The mirror only carries audio for the train/dev/test splits: 17,212 of the 66,973
validated clips (20.8 h of 76.4 h). The full set needs a Mozilla Data Collective
account; this gets the part that's available without one.

Splits: CV train+dev plus most of CV test go to training. HOLDOUT clips from CV test are
kept back as an extra read-speech benchmark on a corpus the model hasn't seen -- FLEURS
test alone is only 299 clips.

    python scripts/fetch_common_voice_mirror.py
    python scripts/fetch_common_voice_mirror.py --limit 50     # smoke test
"""
import argparse
import csv
import random
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import get_token, hf_hub_download

from src.manifest.build_manifest import sha256_of, write_manifest

REPO = "fsicoli/common_voice_19_0"
OUT_DIR = Path(r"D:\Audion-Data\Urdu\open_datasets")
CLIP_DIR = OUT_DIR / "clips" / "common_voice_ur"
RAW_DIR = OUT_DIR / "raw" / "common_voice_ur"
SPLITS = ["train", "dev", "test"]
HOLDOUT = 1000
SAMPLE_RATE = 16000

csv.field_size_limit(10**7)


def fetch(path: str) -> Path:
    return Path(hf_hub_download(REPO, path, repo_type="dataset", token=get_token()))


def read_tsv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def to_wav(src: Path, dst: Path) -> None:
    """Common Voice ships 48 kHz mp3; training expects 16 kHz mono wav."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-ac", "1",
                    "-ar", str(SAMPLE_RATE), str(dst)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap clips per split (smoke test)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    durations = {}
    for row in read_tsv(fetch("transcript/ur/clip_durations.tsv")):
        values = list(row.values())
        durations[values[0]] = float(values[1]) / 1000.0

    rows = []
    for split in SPLITS:
        split_rows = read_tsv(fetch(f"transcript/ur/{split}.tsv"))
        if args.limit:
            split_rows = split_rows[: args.limit]
        for r in split_rows:
            r["_cv_split"] = split
        rows += split_rows
        print(f"[cv] {split}: {len(split_rows)} clips")

        tar_path = fetch(f"audio/ur/{split}/ur_{split}_0.tar")
        wanted = {r["path"] for r in split_rows}
        with tarfile.open(tar_path) as tar:
            for member in tar:
                name = Path(member.name).name
                if name in wanted and not (RAW_DIR / name).exists():
                    with tar.extractfile(member) as src, open(RAW_DIR / name, "wb") as dst:
                        dst.write(src.read())

    def convert(r):
        wav = CLIP_DIR / (Path(r["path"]).stem + ".wav")
        if not wav.exists():
            to_wav(RAW_DIR / r["path"], wav)
        return wav

    print(f"[cv] converting {len(rows)} clips to 16 kHz wav ...")
    with ThreadPoolExecutor(args.workers) as pool:
        wavs = list(pool.map(convert, rows))

    test_ids = [i for i, r in enumerate(rows) if r["_cv_split"] == "test"]
    held_out = set(random.Random(0).sample(test_ids, min(HOLDOUT, len(test_ids))))

    records = []
    for i, (r, wav) in enumerate(zip(rows, wavs)):
        duration_s = durations.get(r["path"], 0.0)
        records.append({
            "clip_id": f"common_voice_ur_{Path(r['path']).stem}",
            "path": str(wav),
            "start_ms": 0,
            "end_ms": int(duration_s * 1000),
            "duration_s": duration_s,
            "source_id": "common_voice_ur",
            "source_file": f"{REPO}/ur:{r['_cv_split']}",
            "speaker": r["client_id"],
            "accent": r.get("accents") or None,
            "emotion": "neutral",
            "verified": True,
            "draft_transcript": r["sentence"].strip(),
            "license": "CC0-1.0",
            "checksum_sha256": sha256_of(wav),
            "split": "test" if i in held_out else "train",
        })

    out = OUT_DIR / "manifest_common_voice_ur.jsonl"
    write_manifest(records, out, dataset_version="v3-common-voice-mirror")
    n_train = sum(1 for r in records if r["split"] == "train")
    hours = sum(r["duration_s"] for r in records if r["split"] == "train") / 3600
    print(f"[done] {len(records)} clips -> {out}")
    print(f"[done] train {n_train} clips ({hours:.1f} h), held-out read-speech test {len(records) - n_train}")


if __name__ == "__main__":
    main()

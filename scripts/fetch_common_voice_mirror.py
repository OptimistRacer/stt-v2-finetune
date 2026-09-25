"""Fetch Common Voice (Urdu or Arabic) from the CC0 mirror and write it in the project's
manifest shape.

Why a separate fetcher: Mozilla moved Common Voice off Hugging Face in October 2025
(`mozilla-foundation/common_voice_17_0`, which configs/opendata_urdu.yaml points at, now
holds only a README pointing to Mozilla Data Collective). `fsicoli/common_voice_19_0`
is a CC0 mirror, but it ships raw tar archives + TSVs rather than a loadable dataset,
so scripts/fetch_open_datasets.py can't read it.

The mirror only carries audio for the train/dev/test splits: for Urdu 17,212 of the
66,973 validated clips (20.8 h of 76.4 h). The full set needs a Mozilla Data Collective
account; this gets the part that's available without one.

Splits: CV train+dev plus most of CV test go to training. HOLDOUT clips from CV test are
kept back as an extra read-speech benchmark on a corpus the model hasn't seen -- FLEURS
test alone is only 299 clips.

Arabic (--lang ar) differs in two ways, both because its train split is 32.5 h from
only 24 speakers:
- --max_per_speaker caps how many clips any one voice contributes, so the model learns
  Arabic rather than those 24 people.
- --speaker_disjoint holds out whole *speakers* (from CV test, which has ~1,000) for the
  test and validation splits. The Urdu run held out random clips, so some held-out
  speakers also had clips in training, which flatters its holdout score.

    python scripts/fetch_common_voice_mirror.py
    python scripts/fetch_common_voice_mirror.py --lang ar --max_per_speaker 300 --speaker_disjoint
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
OUT_DIRS = {
    "ur": Path(r"D:\Audion-Data\Urdu\open_datasets"),
    "ar": Path(r"D:\Audion-Data\Arabic\open_datasets"),
}
SPLITS = ["train", "dev", "test"]
HOLDOUT = 1000
VALIDATION = 300
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


def assign_splits(rows: list[dict], speaker_disjoint: bool, max_per_speaker: int | None) -> dict[int, str]:
    """Row index -> manifest split. Rows left out (over the per-speaker cap) are absent."""
    rng = random.Random(0)
    assigned = {}
    if speaker_disjoint:
        test_speakers = sorted({r["client_id"] for r in rows if r["_cv_split"] == "test"})
        rng.shuffle(test_speakers)
        by_speaker = {}
        for i, r in enumerate(rows):
            by_speaker.setdefault(r["client_id"], []).append(i)
        for split, quota in (("test", HOLDOUT), ("validation", VALIDATION)):
            n = 0
            while n < quota and test_speakers:
                for i in by_speaker[test_speakers.pop()]:
                    assigned[i] = split
                    n += 1
    else:
        test_ids = [i for i, r in enumerate(rows) if r["_cv_split"] == "test"]
        for i in rng.sample(test_ids, min(HOLDOUT, len(test_ids))):
            assigned[i] = "test"

    per_speaker = {}
    for i, r in enumerate(rows):
        if i in assigned:
            continue
        per_speaker[r["client_id"]] = per_speaker.get(r["client_id"], 0) + 1
        if max_per_speaker and per_speaker[r["client_id"]] > max_per_speaker:
            continue
        assigned[i] = "train"
    return assigned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="ur", choices=sorted(OUT_DIRS))
    ap.add_argument("--limit", type=int, default=None, help="cap clips per split (smoke test)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_per_speaker", type=int, default=None, help="cap training clips per speaker")
    ap.add_argument("--speaker_disjoint", action="store_true",
                    help="hold out whole speakers for test/validation instead of random clips")
    args = ap.parse_args()

    lang = args.lang
    out_dir = OUT_DIRS[lang]
    clip_dir = out_dir / "clips" / f"common_voice_{lang}"
    raw_dir = out_dir / "raw" / f"common_voice_{lang}"
    clip_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    durations = {}
    for row in read_tsv(fetch(f"transcript/{lang}/clip_durations.tsv")):
        values = list(row.values())
        durations[values[0]] = float(values[1]) / 1000.0

    rows = []
    for split in SPLITS:
        split_rows = read_tsv(fetch(f"transcript/{lang}/{split}.tsv"))
        if args.limit:
            split_rows = split_rows[: args.limit]
        for r in split_rows:
            r["_cv_split"] = split
        rows += split_rows
        print(f"[cv] {split}: {len(split_rows)} clips")

    # Decide splits before extracting, so capped-out clips are never unpacked or converted.
    assigned = assign_splits(rows, args.speaker_disjoint, args.max_per_speaker)
    keep = sorted(assigned)
    splits_out = [assigned[i] for i in keep]
    rows = [rows[i] for i in keep]
    print(f"[cv] keeping {len(rows)} clips")

    for split in SPLITS:
        wanted = {r["path"] for r in rows if r["_cv_split"] == split}
        tar_path = fetch(f"audio/{lang}/{split}/{lang}_{split}_0.tar")
        with tarfile.open(tar_path) as tar:
            for member in tar:
                name = Path(member.name).name
                if name in wanted and not (raw_dir / name).exists():
                    with tar.extractfile(member) as src, open(raw_dir / name, "wb") as dst:
                        dst.write(src.read())

    def convert(r):
        wav = clip_dir / (Path(r["path"]).stem + ".wav")
        if not wav.exists():
            to_wav(raw_dir / r["path"], wav)
        return wav

    print(f"[cv] converting {len(rows)} clips to 16 kHz wav ...")
    with ThreadPoolExecutor(args.workers) as pool:
        wavs = list(pool.map(convert, rows))

    records = []
    for r, wav, split in zip(rows, wavs, splits_out):
        duration_s = durations.get(r["path"], 0.0)
        records.append({
            "clip_id": f"common_voice_{lang}_{Path(r['path']).stem}",
            "path": str(wav),
            "start_ms": 0,
            "end_ms": int(duration_s * 1000),
            "duration_s": duration_s,
            "source_id": f"common_voice_{lang}",
            "source_file": f"{REPO}/{lang}:{r['_cv_split']}",
            "speaker": r["client_id"],
            "accent": r.get("accents") or None,
            "emotion": "neutral",
            "verified": True,
            "draft_transcript": r["sentence"].strip(),
            "license": "CC0-1.0",
            "checksum_sha256": sha256_of(wav),
            "split": split,
        })

    out = out_dir / f"manifest_common_voice_{lang}.jsonl"
    write_manifest(records, out, dataset_version="v3-common-voice-mirror")
    counts = {s: sum(1 for r in records if r["split"] == s) for s in ("train", "validation", "test")}
    hours = sum(r["duration_s"] for r in records if r["split"] == "train") / 3600
    print(f"[done] {len(records)} clips -> {out}")
    print(f"[done] train {counts['train']} clips ({hours:.1f} h), validation {counts['validation']}, "
          f"held-out read-speech test {counts['test']}")


if __name__ == "__main__":
    main()

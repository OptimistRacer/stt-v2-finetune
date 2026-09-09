"""Stage C: filter podcast clips to neutral, split them train/val/test, and merge with
the FLEURS manifest into one combined training manifest.

Split design: stratified *within* each source_id, so all three podcasts appear in all
three splits. The alternative -- holding out whole podcasts -- would make the eval
splits measure "does this generalize to an unseen speaker/recording" rather than "did
the model learn this domain", and with only 3 sources it would also swing the numbers
wildly depending on which podcast landed where.

Split key is the clip's position within its source, taken deterministically (every Nth
clip to val/test) rather than randomly, so reruns are reproducible without carrying a
seed around. Clips from one source are contiguous in time, so this also spreads each
split across the whole episode instead of clumping it at one end.

The merged manifest keeps FLEURS's records byte-identical and appends podcast records
with a `split` field, so downstream code can filter either corpus the same way.
"""
import json
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_of_fleurs(rec: dict) -> str:
    """FLEURS records carry their split in source_file, e.g. '...ur_pk:train'."""
    return rec["source_file"].rsplit(":", 1)[-1]


def assign_splits(records: list[dict], val_every: int = 10, test_every: int = 10) -> list[dict]:
    """Deterministic ~80/10/10 within each source: every 10th clip -> validation, every
    10th offset by 5 -> test, rest -> train."""
    by_source: dict[str, list[dict]] = {}
    for rec in records:
        by_source.setdefault(rec["source_id"], []).append(rec)

    out = []
    for source_id in sorted(by_source):
        for i, rec in enumerate(by_source[source_id]):
            if i % val_every == 0:
                rec["split"] = "validation"
            elif i % test_every == 5:
                rec["split"] = "test"
            else:
                rec["split"] = "train"
            out.append(rec)
    return out


def filter_neutral(records: list[dict]) -> tuple[list[dict], dict]:
    kept, dropped = [], {"non_neutral": 0, "unverified": 0, "empty": 0}
    for rec in records:
        if not (rec.get("draft_transcript") or "").strip():
            dropped["empty"] += 1
            continue
        if not rec.get("verified"):
            dropped["unverified"] += 1
            continue
        if rec.get("emotion") != "neutral":
            dropped["non_neutral"] += 1
            continue
        kept.append(rec)
    return kept, dropped


def merge(fleurs_path: str | Path, podcast_records: list[dict], out_path: str | Path) -> dict:
    """Write FLEURS records (with their native split) plus podcast records into one
    manifest. Both end up carrying a `split` field so a single loader handles them.

    Clip paths are resolved to absolute: Day 1 wrote podcast clip paths relative to the
    repo root (its output_dir is relative), which would silently break training run
    from any other working directory. FLEURS paths are already absolute."""
    fleurs = load_jsonl(fleurs_path)
    for rec in fleurs:
        rec["split"] = split_of_fleurs(rec)

    for rec in podcast_records:
        rec["path"] = str(Path(rec["path"]).resolve())

    merged = fleurs + podcast_records
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    counts: dict[str, dict[str, int]] = {"fleurs": {}, "podcast": {}, "total": {}}
    for rec in fleurs:
        counts["fleurs"][rec["split"]] = counts["fleurs"].get(rec["split"], 0) + 1
    for rec in podcast_records:
        counts["podcast"][rec["split"]] = counts["podcast"].get(rec["split"], 0) + 1
    for rec in merged:
        counts["total"][rec["split"]] = counts["total"].get(rec["split"], 0) + 1
    counts["n_total"] = len(merged)
    counts["out_path"] = str(out_path)
    return counts

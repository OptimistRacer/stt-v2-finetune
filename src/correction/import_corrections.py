"""Stage B, import half: merge corrected review files back into the manifest.

Validates the round-trip before touching anything: every clip_id that was exported must
come back exactly once, no invented clip_ids, every clip must have a valid emotion. On
any mismatch it reports and refuses to write, rather than silently dropping or
corrupting records -- a dropped training example is invisible later, so fail loud here.
"""
import json
from pathlib import Path

from src.correction.glossary import apply_glossary, load_glossary

VALID_EMOTIONS = {"neutral", "non-neutral"}


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def import_corrections(
    manifest_path: str | Path,
    corrected_paths: list[str | Path],
    out_path: str | Path,
    glossary_path: str | Path | None = None,
) -> dict:
    manifest_path = Path(manifest_path)
    out_path = Path(out_path)
    glossary = load_glossary(glossary_path) if glossary_path else {}

    manifest = _read_jsonl(manifest_path)
    by_id = {r["clip_id"]: r for r in manifest}

    corrections: dict[str, dict] = {}
    problems: list[str] = []
    for cp in corrected_paths:
        cp = Path(cp)
        for rec in _read_jsonl(cp):
            cid = rec.get("clip_id")
            if cid is None:
                problems.append(f"{cp.name}: a record has no clip_id")
                continue
            if cid not in by_id:
                problems.append(f"{cp.name}: clip_id not in manifest (invented?): {cid}")
                continue
            if cid in corrections:
                problems.append(f"{cp.name}: duplicate clip_id: {cid}")
                continue
            emotion = (rec.get("emotion") or "").strip()
            if emotion not in VALID_EMOTIONS:
                problems.append(f"{cp.name}: clip {cid} has invalid emotion {emotion!r} (want one of {sorted(VALID_EMOTIONS)})")
                continue
            corrections[cid] = rec

    # Which manifest clips were exported for review? export_for_review.py skips both
    # needs_manual_split clips and empty-draft clips, so only the rest are expected
    # back -- don't flag the skipped ones as "missing".
    expected_ids = {
        cid for cid, r in by_id.items()
        if not r.get("needs_manual_split") and (r.get("draft_transcript") or "").strip()
    }
    missing = expected_ids - corrections.keys()
    if missing:
        problems.append(f"{len(missing)} exported clip(s) missing from corrections, e.g. {sorted(missing)[:5]}")

    if problems:
        return {"status": "error", "problems": problems, "written": 0}

    merged = []
    for rec in manifest:
        cid = rec["clip_id"]
        if cid in corrections:
            text = corrections[cid]["draft_transcript"]
            rec["draft_transcript"] = apply_glossary(text, glossary)
            rec["emotion"] = corrections[cid]["emotion"]
            rec["verified"] = True
        merged.append(rec)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    verified = sum(1 for r in merged if r.get("verified"))
    neutral = sum(1 for r in merged if r.get("emotion") == "neutral")
    return {
        "status": "ok",
        "written": len(merged),
        "verified": verified,
        "neutral": neutral,
        "non_neutral": verified - neutral,
        "out_path": str(out_path),
    }

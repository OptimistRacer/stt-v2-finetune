"""Merge an approved candidates file into the code-switch dictionary.

The human-in-the-loop half of scripts/harvest_codeswitch.py. Refuses anything still
containing the `????` placeholder rather than accepting a half-filled entry, and reports
what it skipped -- a silently dropped line would look identical to an accepted one.

Usage:
    python scripts/apply_codeswitch_candidates.py --file codeswitch_candidates.txt
    python scripts/apply_codeswitch_candidates.py --file candidates.txt --dry-run
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.postprocess.codeswitch import DEFAULT_PATH

ENTRY = re.compile(r"^\s*([^#=]+?)\s*=\s*([^#]+?)\s*(?:#.*)?$")


def run(file: str, dict_path: str, dry_run: bool) -> None:
    path = Path(dict_path)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    terms: dict[str, str] = data.get("terms", {})
    protected: list[str] = data.get("protected", [])

    added, updated, skipped, prot_added = {}, {}, [], []
    in_protected = False

    for raw in Path(file).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("---"):
            in_protected = "PROTECTED" in line.upper()
            continue
        if line.startswith("#") or line.startswith("=") or line.upper().startswith("CODE-SWITCH"):
            continue

        if in_protected and "=" not in line:
            if line not in protected:
                protected.append(line)
                prot_added.append(line)
            continue

        m = ENTRY.match(line)
        if not m:
            continue
        key, val = m.group(1).strip(), m.group(2).strip()
        if "?" in key or "?" in val or not key or not val:
            skipped.append(line)
            continue
        if key in terms and terms[key] != val:
            updated[key] = val
        elif key not in terms:
            added[key] = val
        terms[key] = val

    print(f"[apply] added {len(added)}, updated {len(updated)}, protected +{len(prot_added)}")
    if skipped:
        print(f"[apply] SKIPPED {len(skipped)} unfilled line(s) still containing '????':")
        for s in skipped[:5]:
            print(f"          {s}")
        if len(skipped) > 5:
            print(f"          ... and {len(skipped) - 5} more")

    if dry_run:
        print("[apply] --dry-run: nothing written")
        return

    data["terms"] = terms
    data["protected"] = protected
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[apply] dictionary now has {len(terms)} terms, {len(protected)} protected -> {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True)
    p.add_argument("--dict", default=str(DEFAULT_PATH))
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    run(a.file, a.dict, a.dry_run)

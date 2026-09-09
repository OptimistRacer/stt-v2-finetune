"""Code-switch glossary: consistent Latin-script rendering of recurring English/loan
terms in corrected Urdu transcripts.

The correction convention (see the CORRECTION_GUIDE.md that export_for_review.py
writes) is: Urdu words stay in Urdu/Nastaliq script, English and loan words are written
in Latin script rather than phonetically transliterated into Urdu. A full dictionary
can't cover novel words, so the convention is applied as a *rule* by whoever corrects
the text; this glossary only pins the handful of terms that must render the same way
every time they appear -- proper nouns, brand names, recurring tech terms -- so they
don't drift between clips.

Deliberately starts near-empty (configs/codeswitch_glossary.json). It's meant to grow
as recurring terms surface during correction, not to be pre-filled with guesses.
"""
import json
from pathlib import Path


def load_glossary(path: str | Path) -> dict[str, str]:
    """Return the term->canonical-form mapping. Missing/empty file -> {} (not an error;
    an empty glossary is the normal starting state)."""
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("terms", {})


def apply_glossary(text: str, glossary: dict[str, str]) -> str:
    """Replace each glossary key with its canonical form. Whole-substring replacement,
    longest keys first so a longer term isn't partially clobbered by a shorter one it
    contains. No-op when the glossary is empty."""
    for key in sorted(glossary, key=len, reverse=True):
        text = text.replace(key, glossary[key])
    return text

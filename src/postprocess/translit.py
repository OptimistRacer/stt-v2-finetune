"""Crude Urdu->Latin romanization, used to tell transliterations from translations.

The code-switch harvester diffs two transcriptions and proposes (urdu, english) pairs.
Most are genuine transliterations -- سکرین/"screen" -- but the model sometimes
*translates* instead, yielding pairs like تقریباً/"almost", تحجد/"time" and ہے/"possible".
Those are catastrophic to add: تقریباً is ordinary Urdu for "almost", so the entry would
rewrite a real Urdu word as English everywhere it appears.

The two cases are easy to separate by sound. A transliteration keeps the English word's
consonants (سکرین really is s-k-r-y-n); a translation shares nothing phonetically. So
romanize the Urdu roughly, compare against the English, and require some similarity.

This is deliberately approximate -- Urdu omits short vowels, and one letter can map to
several sounds. It only has to be good enough to separate "sounds like the same word"
from "sounds nothing like it", and it is only ever a filter on *proposals* a human still
reviews.
"""
import difflib
import re

# Consonant skeletons matter; short vowels are usually unwritten in Urdu anyway.
_MAP = {
    "ا": "a", "آ": "a", "ب": "b", "پ": "p", "ت": "t", "ٹ": "t", "ث": "s",
    "ج": "j", "چ": "ch", "ح": "h", "خ": "kh", "د": "d", "ڈ": "d", "ذ": "z",
    "ر": "r", "ڑ": "r", "ز": "z", "ژ": "zh", "س": "s", "ش": "sh", "ص": "s",
    "ض": "z", "ط": "t", "ظ": "z", "ع": "a", "غ": "gh", "ف": "f", "ق": "q",
    "ک": "k", "گ": "g", "ل": "l", "م": "m", "ن": "n", "ں": "n", "و": "o",
    "ہ": "h", "ھ": "h", "ء": "", "ی": "y", "ئ": "y", "ے": "e",
}
_VOWELS = set("aeiouy")


def romanize(urdu: str) -> str:
    return "".join(_MAP.get(ch, "") for ch in urdu)


def _norm_english(s: str) -> str:
    """Fold English spellings toward how they are pronounced, since the Urdu side is
    written by sound: 'physical' starts with an f, and Urdu spells it that way."""
    s = s.lower()
    for a, b in (("ph", "f"), ("qu", "kw"), ("ck", "k"), ("x", "ks"),
                 ("tion", "shn"), ("sion", "shn"), ("ce", "se"), ("ci", "si"),
                 ("c", "k"), ("wh", "w")):
        s = s.replace(a, b)
    return re.sub(r"(.)\1+", r"\1", s)  # collapse doubles: "apps" -> "aps"


def _skeleton(s: str) -> str:
    """Consonants only, deduplicated -- the part that survives transliteration."""
    s = re.sub(r"[^a-z]", "", s.lower())
    out = [c for c in s if c not in _VOWELS]
    dedup = [c for i, c in enumerate(out) if i == 0 or c != out[i - 1]]
    return "".join(dedup)


def similarity(urdu: str, english: str) -> float:
    """0..1. High when the English word plausibly transliterates to the Urdu one."""
    rom, eng = romanize(urdu), _norm_english(english)
    if not rom or not eng:
        return 0.0
    full = difflib.SequenceMatcher(None, rom, eng).ratio()
    skel = difflib.SequenceMatcher(None, _skeleton(rom), _skeleton(eng)).ratio()
    score = max(full, skel)
    # Very short strings score high by coincidence -- a single shared consonant between
    # a 1-letter and 3-letter skeleton already reads as 0.5. Require a real overlap
    # before trusting anything that short.
    if min(len(_skeleton(rom)), len(_skeleton(eng))) < 3:
        score *= 0.6
    return score


def looks_like_transliteration(urdu: str, english: str, threshold: float = 0.45) -> bool:
    return similarity(urdu, english) >= threshold

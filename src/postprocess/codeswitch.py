"""Rewrite transliterated English back into Latin script in an Urdu transcript.

Whisper transliterates English/loan words into Urdu script when the surrounding speech
is Urdu-dominant ("ٹرگر" for "trigger"). The convention wanted here is Urdu words in
Urdu script, English words in Latin. Two layers handle that: an initial_prompt nudges
the model at decode time (see scripts/transcribe.py), and this dictionary deterministically
cleans up whatever the prompt missed.

Three rules keep it from doing damage, because a wrong replacement is worse than a
missed one -- a missed word merely looks untranslated, a wrong one corrupts real Urdu:

1. WHOLE WORDS ONLY. Substring replacement would rewrite the inside of unrelated words;
   Urdu script has no capitalization to disambiguate and many short transliterations are
   substrings of ordinary Urdu words.
2. A PROTECTED LIST that is never converted. Plenty of English-origin words are simply
   normal written Urdu now -- منٹ (minute), پروگرام, فون, ٹیم. Converting those makes the
   text read *worse*, not better. In one 35-minute lecture منٹ alone appeared 19 times.
3. NOTHING IS ADDED AUTOMATICALLY. New terms arrive only via the harvester
   (scripts/harvest_codeswitch.py), which proposes candidates for a human to approve.
   Auto-adding guesses would silently poison every future transcript.

Dictionary keys are the Urdu spellings Whisper *actually produces*, harvested from real
transcripts -- not guessed. The same English word often has several transliterations
(both "ڈیجٹل" and "ڈیجیٹل" appear for "digital" in this project's own data), so guessing
would miss the real spellings while inventing ones that never occur.
"""
import json
import re
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "codeswitch_dict.json"

# Urdu/Arabic letters. Used to build word boundaries, since Python's \b is defined on
# ASCII word characters and does not behave usefully around Urdu script.
_URDU = r"؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿"


class CodeSwitcher:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else DEFAULT_PATH
        data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.terms: dict[str, str] = data.get("terms", {})
        self.protected: set[str] = set(data.get("protected", []))
        # Longest first so a multi-word term wins over its own first word.
        active = [t for t in sorted(self.terms, key=len, reverse=True) if t not in self.protected]
        self._patterns = [
            (re.compile(rf"(?<![{_URDU}\w]){re.escape(t)}(?![{_URDU}\w])"), self.terms[t])
            for t in active
        ]

    @property
    def n_terms(self) -> int:
        return len(self._patterns)

    def apply(self, text: str) -> str:
        for pattern, replacement in self._patterns:
            text = pattern.sub(replacement, text)
        return text

    def explain(self, text: str) -> list[tuple[str, str, int]]:
        """Which terms would fire, and how often. For reviewing a change before trusting it."""
        hits = []
        for pattern, replacement in self._patterns:
            n = len(pattern.findall(text))
            if n:
                hits.append((pattern.pattern, replacement, n))
        return hits

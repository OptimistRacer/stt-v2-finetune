"""Prepare Urdu text so the voice can actually say it.

The voice's vocabulary is Arabic-script Urdu characters, digits and space -- nothing
else. Measured failures before this layer existed:

    "میں نے 25 کتابیں خریدیں"   -> heard as "مینے صورت کتابیں خری لیں"  (digits garbled)
    "اس کی قیمت 1500 روپے ہے"   -> heard as "اس کی قیمت 5000 روپے ہے"  (WRONG number)
    "یہ ایک اچھا podcast ہے"    -> heard as "یہ گیٹ اچھا ہے"            (English dropped)

The 1500 -> 5000 case is the reason this exists: the audio sounds fluent while stating
a different figure, which is worse than an obvious error.

So: digits become Urdu number words, English words are looked up in the mined
pronunciation lexicon (see build_tts_lexicon.py), and anything still unsayable is
reported rather than silently dropped, so the caller can warn instead of shipping
audio with words missing.
"""
import json
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEXICON_PATH = HERE / "tts_lexicon.json"
VOCAB_PATH = HERE / "urdu_tts_final" / "vocab.json"

# The Arabic voice's vocabulary contains no digits at all (the Urdu one does), so an
# unconverted number doesn't mispronounce -- it disappears. Arabic number words come
# from num2words rather than a hand-written table.
ARABIC_VOICE = "facebook/mms-tts-ara"

# Optional diacritics the vocabulary lacks; Urdu is normally written without them.
STRIP_CHARS = str.maketrans("", "", "ًٌٍَُِّْٓٔ")
LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")
NUMBER = re.compile(r"\d[\d,]*")
ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

ONES = ["صفر", "ایک", "دو", "تین", "چار", "پانچ", "چھ", "سات", "آٹھ", "نو"]
# Urdu has a distinct word for every number to 99 rather than a regular pattern, so
# they are listed rather than composed. Spellings cross-checked against the
# human-written FLEURS Urdu transcripts where those numbers occur.
TENS_UNITS = [
    "دس", "گیارہ", "بارہ", "تیرہ", "چودہ", "پندرہ", "سولہ", "سترہ", "اٹھارہ", "انیس",
    "بیس", "اکیس", "بائیس", "تئیس", "چوبیس", "پچیس", "چھببیس", "ستائیس", "اٹھائیس", "انتیس",
    "تیس", "اکتیس", "بتیس", "تینتیس", "چونتیس", "پینتیس", "چھتیس", "سینتیس", "اڑتیس", "انتالیس",
    "چالیس", "اکتالیس", "بیالیس", "تینتالیس", "چوالیس", "پینتالیس", "چھیالیس", "سینتالیس", "اڑتالیس", "انچاس",
    "پچاس", "اکاون", "باون", "ترپن", "چون", "پچپن", "چھپن", "ستاون", "اٹھاون", "انسٹھ",
    "ساٹھ", "اکسٹھ", "باسٹھ", "ترسٹھ", "چوسٹھ", "پینسٹھ", "چھیاسٹھ", "سڑسٹھ", "اڑسٹھ", "انہتر",
    "ستر", "اکہتر", "بہتر", "تہتر", "چوہتر", "پچہتر", "چھہتر", "ستتر", "اٹھتر", "اناسی",
    "اسی", "اکیاسی", "بیاسی", "تراسی", "چوراسی", "پچاسی", "چھیاسی", "ستاسی", "اٹھاسی", "نواسی",
    "نوے", "اکانوے", "بانوے", "ترانوے", "چورانوے", "پچانوے", "چھیانوے", "ستانوے", "اٹھانوے", "ننانوے",
]
SCALES = [(10_000_000, "کروڑ"), (100_000, "لاکھ"), (1_000, "ہزار"), (100, "سو")]
MAX_NUMBER = 10_000_000 * 100  # beyond this, leave the digits alone and say so


def number_to_urdu(n: int) -> str | None:
    """None when the number is out of range, so the caller can warn instead of
    inventing something wrong."""
    if n < 0 or n >= MAX_NUMBER:
        return None
    if n < 10:
        return ONES[n]
    if n < 100:
        return TENS_UNITS[n - 10]
    for value, name in SCALES:
        if n >= value:
            head, rest = divmod(n, value)
            words = f"{number_to_urdu(head)} {name}"
            return words if rest == 0 else f"{words} {number_to_urdu(rest)}"
    return None


def load_lexicon(path=LEXICON_PATH):
    if not Path(path).exists():
        return {}
    return {k: v["urdu"] for k, v in json.loads(Path(path).read_text(encoding="utf-8")).items()}


_VOCAB_CACHE = {}


def load_vocab(path=VOCAB_PATH, language="ur"):
    if language == "ar":
        if "ar" not in _VOCAB_CACHE:
            from transformers import AutoTokenizer

            _VOCAB_CACHE["ar"] = set(AutoTokenizer.from_pretrained(ARABIC_VOICE).get_vocab().keys())
        return _VOCAB_CACHE["ar"]
    if not Path(path).exists():
        return None
    return set(json.loads(Path(path).read_text(encoding="utf-8")).keys())


def arabic_number(n: int) -> str | None:
    from num2words import num2words

    try:
        return num2words(n, lang="ar")
    except (OverflowError, NotImplementedError, ValueError):
        return None


def normalize_for_tts(text: str, lexicon=None, vocab=None, language="ur"):
    """Returns (normalized_text, report). The report lists what was converted and --
    more importantly -- what could not be, so callers can warn rather than emit audio
    with words silently missing.

    The English->script lexicon is Urdu-only; Arabic has no equivalent yet, so English
    words in Arabic text are reported as unsayable rather than mangled."""
    lexicon = (load_lexicon() if language == "ur" else {}) if lexicon is None else lexicon
    vocab = load_vocab(language=language) if vocab is None else vocab
    spell = arabic_number if language == "ar" else number_to_urdu
    report = {"numbers": [], "words": [], "unknown_words": [], "unsayable_chars": []}

    text = unicodedata.normalize("NFC", text or "").translate(ARABIC_INDIC).translate(STRIP_CHARS)

    def sub_number(m):
        raw = m.group(0)
        words = spell(int(raw.replace(",", "")))
        if words is None:
            report["numbers"].append({"from": raw, "to": None})
            return raw
        report["numbers"].append({"from": raw, "to": words})
        return words

    text = NUMBER.sub(sub_number, text)

    def sub_word(m):
        raw = m.group(0)
        urdu = lexicon.get(raw.lower().strip("'-"))
        if urdu:
            report["words"].append({"from": raw, "to": urdu})
            return urdu
        report["unknown_words"].append(raw)
        return raw

    text = LATIN_WORD.sub(sub_word, text)
    text = re.sub(r"[ \t]+", " ", text).strip()

    if vocab:
        # Sentence punctuation (۔ for Urdu, . for Arabic, plus ؟ ! ،) is used for
        # splitting and then harmlessly ignored by the tokenizer, so don't report it --
        # only flag characters that would vanish mid-word and lose meaning.
        leftover = {c for c in text if c not in vocab and not c.isspace() and c not in "۔؟?!،.:;"}
        report["unsayable_chars"] = sorted(leftover)

    return text, report


def has_sayable(text: str, vocab=None, language="ur") -> bool:
    """Whether the voice would produce anything at all for this text.

    The tokenizer silently drops characters outside its vocabulary, so a chunk of only
    Latin letters reduces to an empty sequence and VITS then dies inside
    `_relative_position_to_absolute_position` with 'narrow(): length must be
    non-negative'. Callers use this to skip such chunks instead of crashing."""
    vocab = load_vocab(language=language) if vocab is None else vocab
    if not vocab:
        return bool(text.strip())
    return any(c in vocab and not c.isspace() for c in text)


def describe(report) -> str:
    """One-line human summary, for the CLI and the web UI warning."""
    bits = []
    if report["numbers"]:
        ok = [n for n in report["numbers"] if n["to"]]
        if ok:
            bits.append(f"{len(ok)} number(s) spelled out")
        bad = [n["from"] for n in report["numbers"] if not n["to"]]
        if bad:
            bits.append(f"number(s) too large to say: {', '.join(bad)}")
    if report["words"]:
        bits.append(f"{len(report['words'])} English word(s) converted")
    if report["unknown_words"]:
        bits.append("cannot say (will be skipped): " + ", ".join(sorted(set(report["unknown_words"]))))
    if report["unsayable_chars"]:
        bits.append("unsupported characters: " + " ".join(report["unsayable_chars"]))
    return "; ".join(bits)


if __name__ == "__main__":
    import sys

    for sample in sys.argv[1:] or [
        "میں نے 25 کتابیں خریدیں۔",
        "اس کی قیمت 1500 روپے ہے۔",
        "یہ ایک اچھا podcast ہے۔",
        "ہم نے 2026 میں leadership پر بات کی۔",
    ]:
        out, rep = normalize_for_tts(sample)
        print(f"in : {sample}")
        print(f"out: {out}")
        print(f"     {describe(rep) or 'no changes needed'}\n")

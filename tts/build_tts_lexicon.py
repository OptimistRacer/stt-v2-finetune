"""Build an English -> Urdu-script pronunciation lexicon for the TTS front-end.

The voice can only say characters in its vocabulary (Arabic-script Urdu, digits,
space). Latin-script words are effectively dropped: "یہ ایک اچھا podcast ہے" came back
from the recognizer as "یہ گیٹ اچھا ہے" -- the English word vanished.

The lexicon is derived from data rather than guessed: the same 156 host clips were
transcribed twice, once with English kept in Latin script and once with a prompt that
pushed Whisper to spell English phonetically in Urdu. Aligning the two transcriptions
of the same audio yields real pronunciation pairs (start -> سٹارٹ, journey -> جرنی).
The STT project's code-switch dictionary, which maps the same relationship in the
opposite direction, is folded in as well.

    python build_tts_lexicon.py
"""
import collections
import difflib
import json
import re
from pathlib import Path

TTS = Path("D:/Audion-Data/TTS")
CODESWITCH_DICT = Path(r"C:\Users\Sistems-AI\stt-v2-finetune\configs\codeswitch_dict.json")
OUT = TTS / "tts_lexicon.json"

# v1 kept English in Latin script; v3 was re-transcribed with an Urdu-biasing prompt.
LATIN_SIDE = TTS / "vits_dataset" / "manifest_full.json"
URDU_SIDE = TTS / "vits_dataset_v3" / "manifest_full.json"

LATIN_WORD = re.compile(r"^[A-Za-z][A-Za-z'\-]*$")


def load(path):
    return {r["file_name"]: r["text"] for r in json.loads(Path(path).read_text(encoding="utf-8"))}


def mine_pairs(latin_side, urdu_side):
    """A Latin word replaced by Urdu words at the same position in the other
    transcription of the same audio is a pronunciation pair."""
    pairs = collections.Counter()
    for name in sorted(set(latin_side) & set(urdu_side)):
        a, b = latin_side[name].split(), urdu_side[name].split()
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
            if tag != "replace":
                continue
            src, dst = a[i1:i2], b[j1:j2]
            if len(src) == 1 and LATIN_WORD.match(src[0]) and 1 <= len(dst) <= 2 \
                    and not any(LATIN_WORD.match(w) for w in dst):
                pairs[(src[0].lower().strip("'-"), " ".join(dst))] += 1
    return pairs


def main():
    pairs = mine_pairs(load(LATIN_SIDE), load(URDU_SIDE))

    # Keep the most frequent Urdu spelling per English word.
    best = {}
    for (en, ur), count in pairs.most_common():
        if en not in best:
            best[en] = {"urdu": ur, "count": count, "source": "paired-transcripts"}

    added = 0
    if CODESWITCH_DICT.exists():
        cs = json.loads(CODESWITCH_DICT.read_text(encoding="utf-8"))
        for urdu, english in cs.get("terms", {}).items():
            key = english.lower()
            if key not in best:
                best[key] = {"urdu": urdu, "count": 0, "source": "codeswitch-dict"}
                added += 1

    vocab = set(json.loads((TTS / "urdu_tts_final" / "vocab.json").read_text(encoding="utf-8")).keys())
    unpronounceable = {en: v for en, v in best.items() if set(v["urdu"]) - vocab}
    for en in unpronounceable:
        del best[en]

    OUT.write_text(json.dumps(dict(sorted(best.items())), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[lexicon] {len(best)} entries "
          f"({len(best) - added} mined from paired transcripts, {added} from the code-switch dictionary)")
    if unpronounceable:
        print(f"[lexicon] dropped {len(unpronounceable)} entries whose Urdu spelling the voice can't say")
    print(f"[done] -> {OUT}")


if __name__ == "__main__":
    main()

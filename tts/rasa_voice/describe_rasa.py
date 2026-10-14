"""Turn DataSpeech's measurements into the one-sentence voice descriptions Parler-TTS
trains on, one per clip.

Input: the dataset DataSpeech's main.py saved with --output_dir (columns from it:
utterance_pitch_mean/std, snr, c50, speech_duration), built from rasa_male_hf/.
Output: rasa_male_desc/data/{train,test}-*.parquet with `id` and `description`, which the
training script joins to rasa_male_hf/ by id (--id_column_name id).

Wording:
- all wording uses DataSpeech's v02 words -- the vocabulary Parler-TTS models were
  trained to understand. Noise, room sound and pitch use its v02 bin edges too;
  expressiveness (pitch spread) uses this speaker's own quantiles, like speaking rate.
- speaking rate is NOT taken from DataSpeech: it counts English phonemes and is
  meaningless for Urdu. Here it is Urdu letters per second of speech, binned by this
  speaker's own quantiles, so "slightly fast" means fast *for him*.
- the clip's Rasa style becomes a tone phrase ("in a happy, cheerful tone"), which is
  what lets the voice be steered away from a flat reading at inference time.
- every description names the voice (NAME), as the Parler single-speaker recipe does,
  so the name in a request brings back this voice.

    ..\\venv_dataspeech\\Scripts\\python.exe describe_rasa.py --tags <main.py output_dir>
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import Dataset, load_from_disk

HERE = Path(__file__).resolve().parent
DATASPEECH = HERE.parent / "dataspeech" / "examples" / "tags_to_annotations"
OUT = HERE / "rasa_male_desc"
# One name per speaker: a request's description must contain the name to get that voice.
# Parvaaz (male) is the product voice; the female speaker only adds Urdu training data (v2).
NAMES = {"Male": "Parvaaz", "Female": "Sana"}
PRONOUN = {"Male": "his", "Female": "her"}

STYLE = {
    "CONV": "in a natural, conversational tone",
    "WIKI": "in a calm, informative reading voice",
    "NEWS": "like a news presenter",
    "BOOK": "like a storyteller narrating a book",
    "HAPPY": "in a happy, cheerful tone",
    "SAD": "in a sad, sorrowful tone",
    "ANGER": "in an angry tone",
    "FEAR": "in a fearful, anxious tone",
    "DISGUST": "in a disgusted tone",
    "SURPRISE": "in a surprised tone",
}
ASSISTANT_STYLES = {"ALEXA", "BB", "UMANG", "DIGI"}  # requests spoken to a voice assistant
RATE = {"very slowly": "very slowly", "slowly": "slowly", "slightly slowly": "slightly slowly",
        "moderate speed": "at a moderate speed", "slightly fast": "slightly fast", "fast": "fast", "very fast": "very fast"}
URDU_LETTER = re.compile(r"[ء-غف-يٱ-ۓ]")

TEMPLATES = [
    "{name} speaks {style}, {rate}, with a {pitch} voice that is {mono}. The recording is {noise} and {reverb}.",
    "In a {pitch} voice that is {mono}, {name} speaks {rate} {style}. The audio is {noise} and {reverb}.",
    "{name} delivers {pron} words {rate} {style}; {pron} {pitch} voice is {mono}. The recording sounds {noise} and {reverb}.",
]


def binned(values, edges, words):
    idx = np.clip(np.digitize(values, edges[1:-1]), 0, len(words) - 1)
    return [words[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True, help="measure_rasa.py output folder")
    ap.add_argument("--out", default=None, help="output folder (default rasa_male_desc/)")
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = HERE / a.out

    edges = json.loads((DATASPEECH / "v02_bin_edges.json").read_text())
    words = json.loads((DATASPEECH / "v02_text_bins.json").read_text())
    ds = load_from_disk(a.tags)
    splits = list(ds.keys())
    cols = ["id", "text", "style", "gender", "duration", "utterance_pitch_mean", "utterance_pitch_std", "snr", "c50",
            "speech_duration"]
    tables = {s: ds[s].select_columns([c for c in cols if c in ds[s].column_names]).to_pandas() for s in splits}
    for df in tables.values():
        if "gender" not in df:  # the v1 male-only dataset has no gender column
            df["gender"] = "Male"

    def rate_of(df):
        letters = df["text"].map(lambda t: len(URDU_LETTER.findall(t)))
        secs = df["speech_duration"].where(df["speech_duration"] > 0.5, df["duration"])
        return letters / secs

    train = tables["train"]
    # Speaking rate and expressiveness are graded against each speaker's own range: on
    # v02's absolute edges (from English multi-speaker audiobooks) nearly every clip is
    # "monotone", which would leave nothing to learn the difference from. Pitch level uses
    # v02's per-gender bins at the speaker level, as DataSpeech does.
    spk = {}
    for g, tg in train.groupby("gender"):
        mean_pitch = float(tg["utterance_pitch_mean"].mean())
        spk[g] = {
            "rate_edges": [0.0, *np.percentile(rate_of(tg), [5, 20, 40, 60, 80, 95]).tolist(), 1e9],
            "mono_edges": [0.0, *np.percentile(tg["utterance_pitch_std"].dropna(), [10, 35, 65, 90]).tolist(), 1e9],
            "pitch": binned([mean_pitch], edges[f"pitch_bins_{g.lower()}"], words["speaker_level_pitch_bins"])[0],
            "mean_pitch": mean_pitch,
            "medians": {c: float(tg[c].median()) for c in ("snr", "c50", "utterance_pitch_std")},
        }
    report = {g: {"name": NAMES[g], "speaker_mean_pitch_hz": round(s["mean_pitch"], 1), "pitch_bin": s["pitch"],
                  "rate_edges_letters_per_s": [round(x, 2) for x in s["rate_edges"][1:-1]],
                  "expressiveness_edges_pitch_std_hz": [round(x, 2) for x in s["mono_edges"][1:-1]]}
              for g, s in spk.items()}
    rng = random.Random(0)
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    for split, df in tables.items():
        df = df.reset_index(drop=True)
        rates = rate_of(df)
        out, counts = [], Counter()
        for i, row in df.iterrows():
            g, s = row["gender"], spk[row["gender"]]
            # A few clips get no SNR/C50 (brouhaha found no speech frames to average over);
            # fill with the speaker's typical value rather than 0, which would read as "very noisy".
            val = lambda c: row[c] if row[c] == row[c] else s["medians"][c]  # NaN check
            rate = binned([rates[i]], s["rate_edges"], words["speaker_rate_bins"])[0]
            noise = binned([val("snr")], edges["noise"], words["snr_bins"])[0]
            reverb = binned([val("c50")], edges["reverberation"], words["reverberation_bins"])[0]
            mono = binned([val("utterance_pitch_std")], s["mono_edges"], words["utterance_level_std"])[0]
            style = STYLE.get(row["style"], "in a clear voice, as if speaking to a voice assistant"
                              if row["style"] in ASSISTANT_STYLES else "in a natural tone")
            pitch = s["pitch"].replace("-pitch", "-pitched").replace("moderate pitch", "moderate-pitched")
            d = rng.choice(TEMPLATES).format(name=NAMES[g], pron=PRONOUN[g], style=style, rate=RATE[rate], pitch=pitch,
                                             mono=mono, noise=noise, reverb=reverb)
            out.append({"id": row["id"], "description": d[0].upper() + d[1:]})
            counts[(g, "rate", rate)] += 1
            counts[(g, "monotony", mono)] += 1
            counts[(g, "noise", noise)] += 1
        for old in (OUT / "data").glob(f"{split}-*.parquet"):
            old.unlink()
        Dataset.from_list(out).to_parquet(str(OUT / "data" / f"{split}-00000-of-00001.parquet"))
        report[split] = {"rows": len(out),
                         "by_speaker": {g: {k: {w: n for (gg, kk, w), n in counts.items() if gg == g and kk == k}
                                            for k in ("rate", "monotony", "noise")} for g in spk},
                         "examples": [o["description"] for o in out[:2]] + [o["description"] for o in out[-2:]]}
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

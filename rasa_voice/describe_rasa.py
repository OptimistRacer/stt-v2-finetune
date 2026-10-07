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
NAME = "Parvaaz"

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
    "{name} delivers his words {rate} {style}; his {pitch} voice is {mono}. The recording sounds {noise} and {reverb}.",
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
    cols = ["id", "text", "style", "duration", "utterance_pitch_mean", "utterance_pitch_std", "snr", "c50", "speech_duration"]
    tables = {s: ds[s].select_columns([c for c in cols if c in ds[s].column_names]).to_pandas() for s in splits}

    def rate_of(df):
        letters = df["text"].map(lambda t: len(URDU_LETTER.findall(t)))
        secs = df["speech_duration"].where(df["speech_duration"] > 0.5, df["duration"])
        return letters / secs

    train = tables["train"]
    rate_edges = [0.0, *np.percentile(rate_of(train), [5, 20, 40, 60, 80, 95]).tolist(), 1e9]
    # Expressiveness is also graded against this speaker's own range: on v02's absolute
    # edges (from English multi-speaker audiobooks) nearly every clip is "monotone", which
    # would leave nothing to learn the difference from.
    mono_edges = [0.0, *np.percentile(train["utterance_pitch_std"].dropna(), [10, 35, 65, 90]).tolist(), 1e9]
    speaker_pitch = float(train["utterance_pitch_mean"].mean())
    pitch_word = binned([speaker_pitch], edges["pitch_bins_male"], words["speaker_level_pitch_bins"])[0]

    report = {"name": NAME, "speaker_mean_pitch_hz": round(speaker_pitch, 1), "pitch_bin": pitch_word,
              "rate_edges_letters_per_s": [round(x, 2) for x in rate_edges[1:-1]],
              "expressiveness_edges_pitch_std_hz": [round(x, 2) for x in mono_edges[1:-1]]}
    rng = random.Random(0)
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    for split, df in tables.items():
        rate = binned(rate_of(df), rate_edges, words["speaker_rate_bins"])
        # A few clips get no SNR/C50 (brouhaha found no speech frames to average over);
        # fill with the speaker's typical value rather than 0, which would read as "very noisy".
        noise = binned(df["snr"].fillna(train["snr"].median()), edges["noise"], words["snr_bins"])
        reverb = binned(df["c50"].fillna(train["c50"].median()), edges["reverberation"], words["reverberation_bins"])
        mono = binned(df["utterance_pitch_std"].fillna(train["utterance_pitch_std"].median()), mono_edges, words["utterance_level_std"])
        out = []
        for i, row in df.reset_index(drop=True).iterrows():
            style = STYLE.get(row["style"], "in a clear voice, as if speaking to a voice assistant"
                              if row["style"] in ASSISTANT_STYLES else "in a natural tone")
            d = rng.choice(TEMPLATES).format(name=NAME, style=style, rate=RATE[rate[i]], pitch=pitch_word.replace("-pitch", "-pitched").replace("moderate pitch", "moderate-pitched"),
                                             mono=mono[i], noise=noise[i], reverb=reverb[i])
            out.append({"id": row["id"], "description": d[0].upper() + d[1:]})
        for old in (OUT / "data").glob(f"{split}-*.parquet"):
            old.unlink()
        Dataset.from_list(out).to_parquet(str(OUT / "data" / f"{split}-00000-of-00001.parquet"))
        report[split] = {"rows": len(out), "rate": dict(Counter(rate)), "noise": dict(Counter(noise)),
                         "reverb": dict(Counter(reverb)), "monotony": dict(Counter(mono)),
                         "examples": [o["description"] for o in out[:3]]}
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

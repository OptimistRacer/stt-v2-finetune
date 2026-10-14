# Arabic: what works, what's measured, what's missing

## Working

- **Transcription**: language selector in the web app. Urdu uses the fine-tune, Arabic
  uses base `large-v3` (already cached). The Urdu code-switch dictionary is forced off
  for Arabic.
- **Voice**: `facebook/mms-tts-ara`, playback tuned by measurement to
  noise_scale 0.5 / duration 0.5 / rate 0.9 (0.107 word error vs 0.196 at library
  defaults). Note this differs from Urdu's best (0.33/0.5/0.9) -- tune per voice.
- **Numbers**: mandatory for Arabic. The Arabic voice's vocabulary has **no digits at
  all**, so an unconverted number doesn't mispronounce, it silently vanishes. Handled
  via `num2words` (real Arabic support), not a hand-written table.
- **English words inside Arabic**: *not* supported. The pronunciation lexicon was mined
  from Urdu recordings; there is no Arabic equivalent, so English is reported as
  unsayable rather than mangled.

## Measured: Gulf dialect

First real measurement, on 5 free sample clips from Nexdata's Saudi corpus
(`gulf_sample/`, human transcripts, 16 kHz, quiet indoor). Reproduce with:

    python eval_arabic.py --dir gulf_sample --language ar

| | WER | CER |
|---|---|---|
| Gulf Arabic (real dialect) | **22.8%** | 8.3% |
| Urdu, for comparison | 18.45% | 7.57% |

**The errors are dialect-specific, not general weakness.** The worst clip (37.5%) fails
exactly on Gulf vocabulary: `شلون` -> `شون`, and `تغى`/`تبغى`/`تبي` all collapsed to
`تبقى`. Other clips sit at 16-18% with mostly harmless spelling variants
(`المسحية`/`المسيحية`).

Caveat: 5 clips, ~60 words each. Treat as a first reading, not a settled number.

## Tried and rejected: dialect prompting (free, didn't work)

| prompt | WER |
|---|---|
| none | **22.8%** |
| Gulf phrasing (short) | 25.3% |
| Gulf phrasing (long) | 25.3% |
| formal Arabic (control) | 40.5% |

Both Gulf prompts were slightly *worse* than no prompt. The control is the informative
result: a formal-Arabic prompt nearly doubled the error, so prompting genuinely does
steer register -- it just can't teach vocabulary the model lacks. Don't retry prompt
wordings on 5 clips; that's fitting to noise.

## Data sources checked (don't re-investigate)

| source | audio | Gulf | conversational | usable |
|---|---|---|---|---|
| ArSyra Gulf (Kaggle + HF `ArSyra/arsyra-gulf`) | no | yes | yes | **no** -- text only, gated, mostly paid |
| Kaggle "Arabic letters as audio" | yes | no | no | **no** -- isolated letters, ~1s each |
| LDC2006T15 | no | yes | yes | transcripts only; needs its audio pair |
| LDC2006S43 (+T15) | yes | yes | yes | yes, if licence/fee acceptable (research-use terms) |
| **Nexdata 849h Saudi** | yes | yes | yes | yes -- paid; GitHub repo has free samples (used above) |
| Quran corpora (everyayah etc.) | yes | no | no | **no** -- classical register; would recite, not speak. Same trap as the Iqbal poetry |
| YouTube podcasts (third-party) | yes | varies | yes | **no** -- not ours; downloading breaks YouTube's terms |
| Kaggle "Organic Gulf Arabic Dialect" (ebubekr53) | no | yes | yes | **no** -- text only (translation-app queries), small sample, CC BY-NC 4.0 |
| HF `oddadmix` Arabic Voice Collection / `dialectal-arabic-lahgtna-v2` | yes | yes (sa, bh, ye) | yes | **rights unclear** -- ~3,000 h, 337 GB, but no licence stated, audio re-hosted from others' YouTube channels/podcasts (`original_video_id`), AI-generated transcripts. Test split (3,056 rows) is the lowest-risk use: internal eval only |
| Common Voice Arabic (CC0 mirror `fsicoli/common_voice_19_0`) | yes | ~none | no | yes -- 57.8 h with audio, read speech, mostly MSA; train split is only 24 speakers |

## What's needed next

Real Gulf audio **with human-corrected transcripts**. Machine transcripts are what
ruined the Urdu voice clone; the same rule applies here.

- **20-30 min** -> a trustworthy Gulf measurement (replaces the 5-clip reading)
- **2-3 h** -> enough to fine-tune and likely beat 22.8%

Best candidates: Virtuosoft's own Arabic podcasts (`configs/day1_arabic.yaml` already
names `Arabic_Podcast1.mp3` / `Arabic_Podcast2.mp3`, not on disk), recording Gulf
speakers directly, Creative-Commons-licensed material, or buying Nexdata.
Check the speakers are actually Gulf -- Egyptian or Levantine audio won't fix the
`شلون`/`تبغى` failures.

The STT project's existing correction workflow (`src/correction/export_for_review.py`
-> correct -> `import_corrections.py`) means transcripts can be machine-drafted and
then human-fixed, which is far faster than typing from scratch.

## Tried: fine-tune on Common Voice Arabic (2026-09-29) -- rejected, retry running

`stt_ar_lora_cv` (stt-v2-finetune `configs/day3_lora_ar_cv.yaml`, 31 h, 943 speakers,
2 epochs). WER vs base large-v3 (VAD on, same clips):

| test | base large-v3 | fine-tune |
|---|---|---|
| CV holdout (MSA read speech, 1,000 clips, speaker-disjoint) | 15.67% | **12.40%** |
| Saudi (Lahgtna test, 273) | **17.61%** | 21.39% |
| Bahraini (193) | **31.63%** | 34.39% |
| Yemeni (115) | **29.87%** | 32.10% |
| Nexdata Saudi (5) | **22.78%** | 27.85% |

Better at MSA, worse at every Gulf set, so the app keeps base large-v3. Cause: 23% of
CV Arabic training sentences carry full tashkeel, and the fine-tune learned to slip into
a diacritized, formal mode that mis-hears dialect (it also picked up Egyptian/Levantine
forms like كده, رح from CV speakers). Retry with diacritics stripped from the targets
(`scripts/clean_arabic_manifest.py`, `configs/day3_lora_ar_cv_clean.yaml`). Lahgtna
references are AI-generated, so read absolute numbers as approximate; the
before/after comparison on identical clips is the signal.

# Parvaaz: an Urdu voice we can use in a product

Status: **v2 (step 1200) live in the app** since 2026-10-14 as Voice "Parvaaz (ours)", the
default and only Urdu voice, with the self-check on. v1 (step 300, in the app 2026-10-12 to
2026-10-14) is kept in `..\parvaaz_voice\` for rollback.

## Why this exists

The app's Urdu voices are either robotic (Standard = Meta MMS) or not allowed in a product
(Natural = OmniVoice). Both are CC-BY-NC, and buying a licence is not an option. So we
fine-tuned a commercially licensed model on commercially licensed data:

| piece | what | licence |
|---|---|---|
| base model | `ai4bharat/indic-parler-tts` (0.9B, 44.1 kHz, Urdu "unofficially" supported) | Apache-2.0 |
| training data | AI4Bharat **Rasa**, Urdu **male** speaker (`ai4bharat/Rasa`) | CC-BY-4.0 |
| training code | `huggingface/parler-tts` (+ local patch below), `huggingface/dataspeech` | Apache-2.0 / MIT |

CC-BY-4.0 needs attribution wherever the voice is used, e.g.:
*"Voice trained on the Rasa dataset by AI4Bharat (CC BY 4.0), fine-tuned from Indic
Parler-TTS (Apache-2.0)."* Have someone confirm the exact wording and voice-likeness
questions before a commercial launch.

The voice is named **"Parvaaz"** in the training descriptions; a request's description
must contain that name to get this voice.

## Results

Word / character error, judged by our Urdu STT model (stt_v3) hearing the audio back:

| model | 6 shared test sentences | 20 held-out Rasa sentences |
|---|---|---|
| real Rasa recordings (judge's floor) | - | 12.1% / 5.4% |
| Meta (current Standard) | 5.8% / 1.8% | - |
| OmniVoice (current Natural) | 0.0% / 0.0% | - |
| Indic Parler, untrained | 11.5% / 7.5% | - |
| Parvaaz step 150 | 14.4% / 5.9% | 20.8% / 9.3% |
| **Parvaaz step 300** | **2.9% / 0.7%** | **15.2% / 6.4%** |
| Parvaaz step 450 | 9.6% / 7.9% | 15.8% / 6.5% |
| Parvaaz step 600 | 3.8% / 1.4% | 18.9% / 7.9% |
| Parvaaz step 654 (final) | 5.8% / 2.5% | 15.2% / 7.4% |

Each sentence is generated once with sampling, so single runs are noisy -- step 300 and
654 were re-checked with two more seeds (see "Seed re-check" below).

**Blind listening test** (`eval_out/blind_test_s300_ratings.json`), naturalness 1-5, 6 sentences,
voices shuffled per sentence: **Parvaaz (step 300) 4.0 = OmniVoice 4.0**, Meta 1.83.
That meets the plan's pass bar (as natural as OmniVoice).

### Seed re-check

Steps 300 and 654 re-generated with `TTS_SEED=1` and `2` (word error, three draws each):

| checkpoint | 6 test sentences | mean (word / char) | 20 held-out | mean (word / char) |
|---|---|---|---|---|
| step 300 | 2.9, 5.8, 7.7 | 5.5% / 2.2% | 15.2, 17.7, 17.7 | 16.9% / 7.5% |
| step 654 (final) | 5.8, 2.9, 1.9 | 3.5% / 1.7% | 15.2, 15.5, 18.9 | 16.5% / 8.1% |

**A tie within noise** -- step 300's single-draw lead was partly luck. Both sit close to
the judge's 12.1% floor on the real speaker. Choose between them by ear:
`C:\Users\Sistems-AI\stt-v2-finetune\outputs\blind_test_s654.html` is the same blind test
for the final checkpoint (step 300's is `blind_test_s300.html`, already rated 4.0).

## Pipeline (in order)

All under `D:\Audion-Data\TTS\rasa_voice\`. Environments: `..\venv_dataspeech` (Python 3.11,
torch 2.8 cu128, datasets 3.6, pyannote 3.3 via brouhaha) and `..\candidates\venv_parler`
(Python 3.11, torch 2.11 cu128, transformers 4.46.1, parler-tts editable from `..\parler-tts`).

1. **Download** the Urdu male shards of Rasa (34 files, 8.7 GB) into `..\rasa_urdu\`;
   `..\rasa_urdu\shard_index.json` records which shards hold which speaker.
2. `prepare_rasa_male.py` -> `rasa_male_hf\` (local HF dataset): drops PROPER NOUN rows
   (lists of Indian names), clips outside 2-20 s, and text with other-script letters.
   **10,473 train clips (21.25 h), 1,168 test clips (2.43 h)**, Rasa's own split kept.
3. `measure_rasa.py` -> `rasa_male_tags\`: DataSpeech's pitch (penn, argmax decoder --
   its Viterbi add-on has no RTX 5090 kernels) and brouhaha SNR / C50 / speech duration.
4. `describe_rasa.py` -> `rasa_male_desc\`: one description per clip from DataSpeech's
   v02 words. Speaking rate (Urdu letters/s) and expressiveness are graded on this
   speaker's own quantiles; the Rasa style becomes a tone phrase ("in a happy, cheerful tone").
5. `train_parvaaz.sh full` (launched detached by `run_train_full.cmd`, log in
   `train_full.log`): batch 4 x 8 accumulation, lr 1e-4 constant with warmup, 2 epochs =
   654 updates, bf16, frozen text encoder, checkpoint every 150. Took 1h25 on the RTX 5090
   including ~60 min of thermal rests (fans stayed ~33%, peak 78 C).
6. Evaluation: `make_eval_models.ps1` (hard-links checkpoints into loadable folders under
   `eval_models\`), `..\candidates\gen_candidate.py parvaaz_<tag> <folder>` (6 shared
   sentences), `gen_heldout.py` (20 held-out sentences), `score_parvaaz.py` (scores ->
   `eval_out\scores.json`), `build_blind_test.py` (blind A/B/C page with 1-5 ratings).
   `TTS_SEED=<n>` re-draws the sampling in both generators.

### Local patch to parler-tts (`parler-tts-local.patch`, on top of upstream d108732)

- audio codes: Indic Parler uses transformers' `DacModel`, whose `encode()` takes
  `n_quantizers` (no `bandwidth`/`padding_mask`) and returns (bsz, codebooks, seq) --
  the script assumed Encodec's (1, bsz, codebooks, seq).
- no crash when training without an eval split.
- GPU thermal pacing after every micro-batch (target 65 C, hard pause 75 C, resume 65 C;
  `THERMAL_*` env vars), logged on each step line.

Re-apply after a fresh clone: `git -C parler-tts apply ..\rasa_voice\parler-tts-local.patch`.

## Known limitations

- **English must be in Urdu script.** Rasa's transcripts have no Latin letters (English
  loanwords are written in Urdu script), so the voice has never read Latin text. The app's
  `text_normalize.py` already maps English words via `tts_lexicon.json` (190 words) and
  spells out numbers; anything else is reported as unsayable.
- Rasa's sentences are Indian-context (names, government services); the speaker's accent
  was judged fine for a Pakistani audience (2026-10-07).
- Generation runs about real time (fp32, no compile) -- fine for a demo, worth speeding up
  (bf16 / SDPA / torch.compile) before heavy use.

## Serving (2026-10-12)

- `export_parvaaz.py --step 300` -> `..\parvaaz_voice\` (3.76 GB safetensors, tokenizer, generation
  config, `ATTRIBUTION.md`, `voice.json` with the description template). No optimizer state.
- `parvaaz_worker.py`: local helper on 127.0.0.1:8097 in `..\candidates\venv_parler`, started by
  the app on first use (about 20 s), log `parvaaz_worker.log`. Speed is asked for in the
  description ("slightly slowly" / "at a moderate speed" / "slightly fast" for the app's
  Slower / Normal / Faster); fixed seed per sentence.
- `..\webapp\server.py`: engine `parvaaz` (Urdu only; numbers spelled out and English words
  mapped by the existing normaliser; English words it doesn't know are read as written, and
  the note says so). `..\webapp\index.html`: Voice option "Parvaaz (ours)", disabled for Arabic.
- `..\deploy\stop_webapp.ps1` also stops this helper.
- Checked through the running app: two Urdu sentences heard back word for word; "کل صبح 10 بجے
  ہماری meeting ہے" -> "dus" spelled out and "meeting" spoken as میٹنگ; Arabic refused cleanly.
- To serve step 654 instead: `export_parvaaz.py --step 654`, then restart the app.

## Settings sweep and self-check (2026-10-12)

The user asked for clearer pronunciation and better voice quality. `sweep_parvaaz.py` tried 7
settings on the 6 shared + 20 held-out sentences, two takes each (`sweep_score.py`, Urdu judge):

| setting | one take | best of 2 takes |
|---|---|---|
| step 654, today's wording, temperature 1.0 | **13.1%** | 10.6% |
| step 654, "very clean recording" | 15.4% | 12.4% |
| step 654, temperature 0.7 | 19.7% | 11.0% |
| step 654, very clean + 0.7 | 17.7% | 12.4% |
| step 654, 0.7 + "slightly slowly" | 25.0% | 16.4% |
| step 654, base model's "high quality" wording + 0.7 | 15.7% | 10.1% |
| step 300, very clean + 0.7 | 21.2% | 14.8% |

Lower sampling temperature makes this model mispronounce *more*, and slowing it down is worst.
Blind voice-quality test (`listening_tests/blind_test_parvaaz_quality_ratings.json`): OmniVoice
4.33, **Parvaaz today (step 300) 4.00**, the three step-654 variants 3.67 each. Decision: keep
step 300 with today's settings; improve pronunciation with the **self-check** instead.

Self-check (in `webapp/server.py`, `best_take`): each sentence is heard back by the app's own
recogniser and regenerated, up to 3 takes, if words come out wrong; the clearest take is kept.
English words left in Latin script are allowed one mismatch each. Keeping the best of 3 takes
measured 3.6% -> 1.2% on the shared sentences and 18.0% -> 12.8% on unseen ones (step 654
seed data). Tested offline; goes live with the next app restart.

Disk clean-up (2026-10-12): smoke-test folders and the optimizer/scheduler/RNG state in every
checkpoint deleted (30 GB). All checkpoint weights (`pytorch_model.bin`) are kept; checkpoints
can no longer resume training, only be evaluated or exported.

## Parvaaz v2: more Urdu data (2026-10-12 to 2026-10-14)

Aim: better voice quality than v1 by training on twice the data. The user chose Rasa's female
Urdu speaker (same studio quality, CC BY 4.0) over IndicVoices-R (49 GB, variable quality).

- Data: `prepare_rasa_male.py --genders Male Female --out rasa_mf_hf` -> **20,107 train clips,
  42.15 h** (male 21.25 h = v1's set, female 20.9 h) + 2,237 test clips. Descriptions
  (`describe_rasa.py`, now per speaker) name the female voice "Sana" (training data only,
  not a product voice); her pitch is graded on DataSpeech's female scale (209.5 Hz,
  "slightly high"). The generalised scripts reproduce v1's descriptions exactly.
- Run unattended: `run_v2_pipeline.cmd` -> `run_v2_pipeline.sh` (download, build, measure,
  describe, `train_parvaaz.sh v2`); logs `v2_pipeline.log`, `train_v2.log`.
- Training: 1,256 updates (2 epochs, batch 4 x 8), 13:49-16:45, ~7.7 s per update;
  checkpoints `parvaaz_v2_train\checkpoint-300/600/900/1200/1256` (optimizer state still
  inside, ~4 GB each -- delete once a checkpoint is chosen).
- **GPU memory lesson:** mid-run it slowed 7x (67 s per update) because the app's idle voice
  helpers (Parvaaz 6 GB, OmniVoice 5 GB) pushed GPU memory past 32 GB, spilling to system RAM
  (signature: ~100% "utilisation" at ~130 W). Stopping the two helpers (user approved)
  restored full speed. Before any GPU job, check for idle helpers.
- Evaluation (`eval_v2.sh`, all checkpoints, two takes each; scores in `eval_out/scores.json`)
  and a blind test (`listening_tests/blind_test_parvaaz_v2_ratings.json`, 6 sentences, 1-5):

  | voice | blind test | word error, shared sentences | word error, unseen sentences |
  |---|---|---|---|
  | **v2 step 1200** | **4.67** | **1.9%** | **14.2%** |
  | v2 step 1256 (final) | 4.33 | | |
  | v1 step 300 (was in the app) | 4.17 | 5.5% | 16.9% |
  | OmniVoice (non-commercial yardstick) | 4.67 | | |
  | real Rasa recordings (recogniser's floor) | | | 12.1% |

  v2 step 1200 ties OmniVoice by ear and beats v1 on both error measures, so it replaced v1.
- Export: `export_parvaaz.py --run v2 --step 1200` -> `..\parvaaz_voice_v2\` (3.76 GB,
  ATTRIBUTION.md now credits both Rasa Urdu speakers' data). `parvaaz_worker.py --model_dir`
  picks the export; the app passes `parvaaz_voice_v2`.

## Deployment (2026-10-14)

- The app (`webapp/server.py`) now offers **only our voices**: Parvaaz v2 for Urdu (helper on
  port 8097) and the Arabic voice (`..\arabic_voice\`, Chatterbox helper on 8098). An empty
  Voice picks our voice for the language; Meta and OmniVoice requests are refused (licence
  non-commercial), and the page hides them. The self-check is on for both voices.
- Tested after the restart: Urdu and Arabic defaults, refusals, page menus. First request
  after a restart loads the helper (~26 s Urdu, ~60 s Arabic); later ones are faster.
- GPU memory with both helpers loaded: ~18-19 GB. **Stop the helpers before any training.**
- Disk clean-up: optimizer/scheduler/RNG state deleted from every v2 checkpoint (19 GB). The
  weights are kept for evaluation or export, not for resuming training.

## Next steps

1. Voice-likeness check of Parvaaz (Rasa's male speaker) before a commercial launch; use the
   credit in `..\parvaaz_voice_v2\ATTRIBUTION.md`.
2. Voice quality: the Parvaaz host's own shorts (`..\parvaaz\audit_shorts.py`, waiting for
   files) or IndicVoices-R.
3. Speed: generation is roughly real time on fp32; bf16 / SDPA / torch.compile could help.

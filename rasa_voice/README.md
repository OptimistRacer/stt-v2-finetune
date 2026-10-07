# Parvaaz: an Urdu voice we can use in a product

Status: **trained and passing**, not yet in the app (2026-10-07).

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

## Next steps (pick up here)

1. Choose the checkpoint by ear: step 300 or step 654 tie on clarity (see Seed re-check).
   Rate `outputs\blind_test_s654.html` the same way as the step-300 test and compare.
2. Export it as a clean, standalone model folder (weights + config + tokenizer +
   generation config + an ATTRIBUTION note) -- the checkpoints also hold ~4 GB of
   optimizer state each that a served model doesn't need.
3. Add it to the speech app as a third voice (`webapp/server.py` engine + Voice menu
   option; its own "sayable" check, since `has_sayable` uses Meta's character list).
   **This needs the deployed app (scheduled task UrduSpeechWebapp, port 8095) restarted --
   ask first.**
4. Disk clean-up once the checkpoint is chosen: `smoke_train\` (11 GB), optimizer states.
5. Later: Arabic voice the same way; optionally the Parvaaz host's own voice from the
   channel's shorts (`..\parvaaz\audit_shorts.py` is ready, waiting for the files).

# STT v2 Fine-Tuning — Urdu Whisper

Local codebase for the STT v2 7-Day Plan (Auton8/DeepPulse pre-task, feeds into Audion's
self-hosted STT). Code lives here and is git-tracked; audio data lives alongside it in
`E:\Audion-Data\Urdu\` (not tracked in git — see `.gitignore`).

## Current state (2026-10-02)

- **Urdu: done and serving.** `stt_v3_lora_cv` (LoRA on Whisper large-v3, trained on
  FLEURS + podcasts + ~20 h of Common Voice) is what `scripts/transcribe.py` and the web
  UI load by default. 17.87% WER on the served path -- see "More Urdu data" below.
- **Arabic: experiment finished, nothing shipped.** Fine-tuning on Common Voice Arabic
  clearly helps read Modern Standard Arabic (15.67% -> 11.47% WER) but costs accuracy on
  Gulf dialect speech, and the cost grows with training steps. Base large-v3 remains the
  safe Arabic model; an early checkpoint (500) is the only fine-tune that comes close on
  Gulf. See "Arabic fine-tune experiment".
- **Still open:** accent tagging and the accent-wise report (need human labels), and
  measurement on real demo-environment audio. See "Open items" at the bottom.

Sections below are in the order the work happened, with the original 7-Day Plan status
snapshot (2026-09-11) after the Urdu work and the newer sections after it.

## Why split into "prep" and "gpu" scripts

- `scripts/run_day1_prep.py` — denoise, loudness-normalize, segment into clips, build a
  manifest skeleton. CPU-only, needs only `ffmpeg` + `requirements-local.txt`. Run this
  on your local machine.
- `scripts/run_day1_gpu.py` — speaker diarization + Whisper-base draft transcription.
  Needs a GPU and `requirements-colab.txt` (`faster-whisper`, `pyannote.audio`). Run this
  in Colab, pointed at the `outputs/` folder produced by the prep step.

This way the heavy ML dependencies never need to touch the local machine, and the local
prep step is fast to iterate on and debug without burning Colab GPU time.

## Data found (2026-08-27)

`E:\Audion-Data\Urdu\`:
- `Urdu-Podcast1-Male-Neutral.mp3` — ~61 min
- `Urdu-Podcast2-maleAndFemal-Neutral.mp3` — ~63 min
- `Urdu-Podcast3-Male-Neutral.mpeg` — ~54 min

Total ~2h57m. The STT v2 7-Day Plan assumed 2 podcasts (2+2h) + ~1h fresh recordings
(~5h total) — there's no separate "fresh recording" file here yet, and there are 3
podcast files rather than 2. Not blocking for Day 1, but worth flagging if the ~5h
assumption matters for Day 3-4 fine-tuning data volume.

## Running Day 1

**1. Local (CPU) — prep + segment:**
```bash
pip install -r requirements-local.txt
pip install audioop-lts   # only needed on Python 3.13+, stdlib audioop was removed
python scripts/run_day1_prep.py --config configs/day1.yaml
```
Produces `outputs/clean/*.wav` (denoised, normalized, 16kHz mono), `outputs/clips/<source>/*.wav`
(15-30s clips, silence-boundary cuts), and `outputs/manifest.jsonl`.

**2. Colab (GPU) — diarize + draft transcribe:**
- Upload the `outputs/` folder (or just `outputs/clean/` + `outputs/manifest.jsonl` +
  `outputs/clips/`) to Google Drive.
- Copy this whole repo folder to Drive too (or push to a private GitHub repo and
  `git clone` in Colab — ask before pushing anywhere).
- Open `notebooks/colab_runner.ipynb`, set `REPO_DIR` and `HF_TOKEN`, run all cells.
- Produces `outputs/manifest_day1_complete.jsonl` with `speaker` and `draft_transcript`
  filled in per clip.

`pyannote/speaker-diarization-3.1` requires a HuggingFace token that has accepted the
model's terms at https://huggingface.co/pyannote/speaker-diarization-3.1 — do this once,
manually, before running Day 1 GPU step.

## Manifest fields

Each line in `manifest.jsonl` / `manifest_day1_complete.jsonl` is one clip record:

| field | meaning |
|---|---|
| `clip_id` | `<source_id>_clip####` |
| `path` | absolute path to the clip WAV |
| `start_ms` / `end_ms` | offset into the cleaned source file |
| `duration_s` | actual clip length |
| `source_id` / `source_file` | which podcast this came from |
| `speaker` | diarization label (filled by `run_day1_gpu.py`) |
| `accent` | `null` until Day 2 accent tagging |
| `verified` | `false` until a human corrects the draft transcript (Day 2) |
| `draft_transcript` | Whisper-base output (filled by `run_day1_gpu.py`) |
| `checksum_sha256` | integrity check on the clip file |
| `dataset_version` | from `configs/day1.yaml` |

These fields exist specifically to close gaps flagged in the Technical Spec Review of the
developer guidelines doc: no checksum/version field in the manifest, no `verified` flag
despite drafts entering the pipeline, no defined duration validation, no minimum segment
length. `accent` and `speaker` are left as explicit fields rather than free text so Day 2
tagging has somewhere defined to write.

## Open-dataset Urdu data (added 2026-09-08)

Demo scope was narrowed to **Urdu, neutral emotion only**. Rather than pulling more
podcasts (copyright risk, and mostly conversational/mixed-emotion audio that would
need filtering back down to neutral), we're pulling open, pre-transcribed read-speech
corpora instead — read-aloud sentences are inherently flat/neutral, and skip the
denoise/diarize/draft-transcribe steps `run_day1_prep.py` + `run_day1_gpu.py` do for
raw podcast audio, since these arrive already single-speaker and transcribed.

- **FLEURS (`google/fleurs`, config `ur_pk`)** — fully open, no login needed.
- **Common Voice 17 (`mozilla-foundation/common_voice_17_0`, config `ur`)** — gated,
  wired up in `configs/opendata_urdu.yaml` but **not used for this demo** (decided
  2026-09-08: FLEURS alone, 8.6h/2,675 clips, is the full dataset for this pass). The
  3 podcast files noted above are also out of scope for now — they need Day 1
  prep/diarization first and live on an `E:` drive not mounted on this machine.

FLEURS's own `train`/`validation`/`test` splits (2109/267/299 ≈ 79/10/11%) are used
as-is — that's already the Day 2 80/10/10 split the 7-Day Plan calls for, so no
separate splitting step is needed for this data.

Note: this machine has no `E:` drive mounted, so open-dataset audio lands on
`D:\Audion-Data\Urdu\open_datasets\` instead of the `E:\Audion-Data\Urdu\` the podcast
data uses — see `configs/opendata_urdu.yaml`. Move it under `E:\` and update the
config's `output_dir` if/when that drive is available.

```bash
pip install -r requirements-opendata.txt
python scripts/fetch_open_datasets.py --config configs/opendata_urdu.yaml --only fleurs_ur
# once logged in and terms accepted:
python scripts/fetch_open_datasets.py --config configs/opendata_urdu.yaml
```

Produces `<output_dir>/clips/<dataset_id>/*.wav` and `<output_dir>/manifest_opendata.jsonl`
(`<output_dir>` is `configs/opendata_urdu.yaml`'s `output_dir`, currently
`D:\Audion-Data\Urdu\open_datasets\` — not git-tracked, same as the podcast data), in
the same record shape as
`manifest.jsonl` plus two extra fields: `emotion` (hardcoded `"neutral"` — no separate
labeling pass needed for this data) and `license` (per-dataset, for provenance since
this gets merged with podcast-derived data before fine-tuning). `verified` is `True`
since these transcripts are ground truth, not a Whisper draft.

`datasets`' own `Audio(sampling_rate=...)` cast needs the `torchcodec` package (pulls
in `torch`) in current versions — `src/opendata/hf_source.py` decodes with
`soundfile`/`librosa` instead to keep heavy ML deps off the local machine, per the
prep/gpu split above.

## Day 3: LoRA fine-tuning (added 2026-09-08)

Runs **locally on this machine's RTX 5090**, not Colab -- unlike Day 1's GPU step, the
local GPU here is faster than what Colab offers and skips the upload round-trip, so
the original prep/gpu split doesn't apply to Day 3 onward.

```bash
pip install -r requirements-train.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128   # match your GPU's CUDA capability
python scripts/run_day3_train.py --config configs/day3_lora.yaml
```

Base model: `openai/whisper-large-v3`, LoRA on `q_proj`/`v_proj` (r=8, alpha=16,
dropout=0.1, weight_decay=0.01) via PEFT — ~0.25% of params trainable. Trains/evals
against `manifest_opendata.jsonl`'s `train`/`validation` splits (FLEURS's own split,
not a separately-generated one). Checkpoints and the final adapter land under
`configs/day3_lora.yaml`'s `training.output_dir`
(`D:\Audion-Data\Urdu\checkpoints\stt_v2_lora\`, not git-tracked).

Smoke-test a config change fast without waiting on a full epoch:
```bash
python scripts/run_day3_train.py --config configs/day3_lora.yaml --max_train_examples 8 --max_steps 2
```

Current numbers (full validation split, 267 clips, `run_day4_eval.py`):

| | WER | CER |
|---|---|---|
| base whisper-large-v3 (zero-shot) | 23.95% | 8.55% |
| Day 3 LoRA adapter | **20.9%** | **6.95%** |

### Post-mortem: repeated "successful" runs that were actually all broken

Every Day 3 run up through the one below trained "successfully" by every training-time
signal (loss dropped smoothly every time) but produced an adapter that scored *worse*
than the un-tuned base model on real `run_day4_eval.py` evaluation -- the reason this
module exists separately from training-time eval, since training's own eval couldn't
have caught any of this (see below). Real output degenerated into repeating a single
token after a few correct words (`تتتتتتتتت...`, `رررررر...`), regardless of:
- learning rate (tried 1e-3, then 1e-4 -- lower delayed the collapse, checkpoint-200
  looked fine at 1e-4 on a 200-example subset, but the same lr still collapsed on the
  full 2109-example set by step 200)
- LoRA rank/regularization (tried r=32 down to r=8, added weight_decay=0.01 and
  dropout=0.1 -- still collapsed, full validation set, 198% WER)
- total training steps (checkpoint-200 was already *more* broken than later
  checkpoints in one run, ruling out "trained too long")

**Actual root cause**, found by inspecting the label pipeline directly after
hyperparameters were exhausted as an explanation: `src/training/dataset.py`'s
`WhisperCollator` was supposed to strip the leading `<|startoftranscript|>` token from
labels before the model's own `shift_tokens_right` re-adds it via
`decoder_start_token_id` -- otherwise every decoder position is off by one for the
model's entire target sequence. The strip condition compared against
`processor.tokenizer.bos_token_id`, which for Whisper is 50257 (`<|endoftext|>` --
Whisper's tokenizer sets `bos_token_id == eos_token_id`; it is *not*
`<|startoftranscript|>`, a distinct token, 50258). That comparison was never true, so
the strip never fired, for the entire history of this repo's training runs: every
training example carried a duplicate `<|startoftranscript|>` at the start of
`decoder_input_ids`. Teacher-forced loss still dropped normally -- the model just
learned a consistently *shifted* mapping -- but real generation, which starts from a
single correctly-placed prompt, hit a decoder-input distribution the model never
actually trained on, and degraded into repetition. This explains every earlier
observation: why loss always looked fine while WER didn't, why no hyperparameter fixed
it, and why an *untrained* PEFT-wrapped model (LoRA at init = zero delta, verified to
match the base model's output exactly) proved the corruption came from training on
mislabeled targets, not from the PEFT/`generate()` integration itself.

Fixed by comparing against the correct token
(`processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")`) and asserting
on it rather than silently no-op'ing if it's ever wrong again. Confirmed on the full
validation set: 20.9% WER, actually beating the base model, no repetition anywhere in
the output.

### Crash post-mortem: this machine rebooted itself, three times

Separately from the label bug above, three full-machine crashes (unclean reboot,
`nvlddmkm` driver-fault storms in the Windows Event Log right before each one) hit
while chasing the collapse -- every single time inside a batched `model.generate()`
call, never once during plain training forward/backward:
1. A full-validation-set `run_day4_eval.py` run (batch_size=8), mid-generate on the
   base model.
2. `Seq2SeqTrainer`'s own `predict_with_generate` eval, at step 200 of a training run.
3. Same as #2, on a retry.

One crash log showed `CUBLAS_STATUS_INTERNAL_ERROR` immediately before the fatal
error, pointing at fused cuBLASLt/SDPA attention kernels being unstable on this RTX
5090 (Blackwell, sm_120) under this driver/torch version. Fixes, all evidence-based
rather than precautionary blanket changes:
- `predict_with_generate` is now forced off in `run_day3_train.py` regardless of
  config -- training eval only computes loss (the one thing that never crashed across
  all 3 incidents). All WER/CER evaluation moved to `run_day4_eval.py`.
- `run_day4_eval.py` (where the actual `generate()` risk lives) loads models with
  `attn_implementation="eager"` (the unfused, stable attention path) and
  `torch.backends.cuda.matmul.allow_tf32 = False`, plus a smaller batch size (2),
  per-batch retry with a CUDA cache clear, and incremental JSONL writes so a crash
  loses at most one in-flight batch and reruns resume automatically.
- Training itself (`src/training/lora_setup.py`) does **not** use eager attention --
  applying it there was tried first, on the reasoning that "sustained heavy load"
  caused the crashes, but that's not what the evidence showed (0/3 crashes were during
  training forward/backward) -- and it backfired: eager materializes the full
  O(seq_len²) attention score matrix per head/layer instead of a fused kernel, and
  Whisper's encoder always processes a fixed 30s/1500-token sequence regardless of
  actual clip length, which pushed training VRAM to ~98% and caused allocator
  thrashing (GPU-reported 100% "utilization" but power draw collapsed to ~115-130W,
  step time climbing 38s→97s/step) at both batch=8/fp32 and batch=4/bf16 -- switching
  batch size and dtype didn't fix it because neither was the actual lever,
  `attn_implementation` was. Caught both times by a watchdog polling
  `nvidia-smi` for exactly that signature (near-VRAM-ceiling memory + abnormally low
  power despite high reported utilization) before it escalated to a crash.
- Training also auto-resumes from the latest checkpoint (`get_last_checkpoint`) if one
  exists in `output_dir`, and checkpoints every 100 steps (was 200) -- so a future
  interruption, from any cause, loses at most ~100 steps, not the whole run.

## Podcast data: Day 1-2 pipeline run (2026-09-09)

The 3 Urdu podcasts (`D:\Audion-Data\Urdu\*.mp3/.mpeg`, ~2h57m) went through the full
Day 1 + Day 2 pipeline and are now merged with FLEURS into a combined training set.

```bash
python scripts/run_day1_prep.py       --config configs/day1.yaml   # denoise/segment  -> 373 clips
python scripts/run_day1_gpu.py        --config configs/day1.yaml   # diarize + draft transcribe
python scripts/run_day2_retranscribe.py --config configs/day1.yaml # re-do drafts with large-v3
python scripts/run_day2_autocorrect.py  --config configs/day1.yaml # glossary + emotion -> manifest_verified
python scripts/run_day2_split_merge.py                             # neutral filter, split, merge with FLEURS
python scripts/run_day3_train.py --config configs/day3_lora_combined.yaml
python scripts/run_day4_eval.py  --config configs/day3_lora_combined.yaml --split test --domain fleurs
```

Combined manifest: **2406 train / 306 validation / 336 test** (3048 clips = 2675 FLEURS
+ 373 podcast).

### Why the drafts were re-transcribed with large-v3

Day 1 drafts come from Whisper-**base**, and they were too garbled to correct from text
alone -- "fixing" them without listening would be plausible-guessing, not correction
(`کو حراج نہیں` -> `کوئی حرج نہیں`, `انسان کی فترہ تھا ہے` -> `انسان کی فطرت ہے`).
large-v3 fixes far more than a human editor could infer from the base output, at zero
manual cost, and recovered all 21 clips base had failed on entirely.

**Accepted limitation:** these are pseudo-labels from the same model being fine-tuned,
so they carry little *new text* signal. The podcast data's value here is acoustic --
conversational delivery, mic, and background that FLEURS's clean read speech doesn't
cover. Genuinely better labels need a human listening per clip.

### What the podcasts actually contain

Not what the filenames suggest. Measured Latin-vs-Urdu character ratio per clip:

| podcast | clips | hours | mean English | >50% English | pure Urdu |
|---|---|---|---|---|---|
| podcast1 | 129 | 0.99h | 0% | 0 | 129 |
| podcast2 | 132 | 1.02h | 24% | 21 | 52 |
| podcast3 | 112 | 0.87h | 40% | 42 | 24 |

podcast1 is pure Urdu; podcast3 is substantially an English-language podcast. All 373
clips were kept (decision 2026-09-09) on the basis that heavily code-switched speech is
legitimate target domain for Pakistani Urdu.

The code-switching convention (Urdu words in Urdu script, English in Latin) turned out
to need no tooling: large-v3 already applies it natively (1208 distinct Latin-script
words across the corpus, e.g. `دونوں کے جو biology وہ same ہے`). Only ~22 occurrences
corpus-wide were English still in Urdu script, mostly naturalized loanwords that
written Urdu renders that way anyway -- so `configs/codeswitch_glossary.json` is
deliberately empty, with the reasoning recorded in the file.

### Results so far

| model | test set | WER | CER |
|---|---|---|---|
| base whisper-large-v3 | FLEURS-only (299) | 22.77% | 8.01% |
| LoRA, FLEURS-only training | FLEURS-only (299) | **20.65%** | **6.79%** |
| base whisper-large-v3 | mixed (336) | 21.89% | 10.14% |
| LoRA, FLEURS+podcast training | mixed (336) | **19.62%** | **8.55%** |

Those two rows are **not directly comparable** -- different test sets. Scored
like-for-like with `--domain` (2026-09-11):

**On FLEURS test (299 clips, real human labels) -- the trustworthy benchmark:**

| model | WER | CER |
|---|---|---|
| base whisper-large-v3 | 22.77% | 8.01% |
| LoRA, FLEURS-only training | 20.65% | 6.79% |
| LoRA, FLEURS+podcast training | 20.55% | 6.79% |

Adding ~3h of podcast audio left FLEURS performance unchanged (0.1pp is noise).

**On podcast test (37 clips), split by how much English the clip contains:**

| subset | base | FLEURS-only LoRA | combined LoRA |
|---|---|---|---|
| pure Urdu (16) | 11.1% | 16.9% | 14.2% |
| English-heavy (17) | 26.2% | **56.2%** | **17.9%** |
| all (37) | 19.6% | 38.3% | 17.1% |

### The finding that actually mattered

**Fine-tuning on FLEURS alone destroyed code-switched/English handling.** On
English-heavy clips the FLEURS-only adapter scores 56.2% WER against base's 26.2% --
more than double the error of not fine-tuning at all. FLEURS is pure Urdu read-aloud
sentences, so training exclusively on it taught the model that English doesn't belong
in the output. Spot-checking confirms it isn't the repetition collapse from the earlier
BOS bug: the Urdu it produces is clean (sometimes better than the reference --
`بدقسمتی سے` where the pseudo-label has `بھتکسمتی سے`), it just mangles the English.

The FLEURS benchmark is completely blind to this: the FLEURS-only model looks *best*
there (20.65%). Real Pakistani Urdu code-switches constantly, so shipping that model on
the strength of its benchmark number would have meant shipping a regression.

Adding the podcast data fixes it (17.9% on English-heavy, better than base) at no cost
to FLEURS. **Use `stt_v2_lora_combined`, not `stt_v2_lora`.**

**Caveat on the podcast columns:** those labels were generated by large-v3 itself
(see above), so base has a built-in advantage -- it partly agrees with itself. That is
why base looks strongest on the pure-Urdu podcast row; don't read that row as
"fine-tuning hurt". The English-heavy degradation is far too large to be explained by
that circularity.

## Audio augmentation for noisy input (2026-09-11)

Both corpora are clean -- measured, not assumed. FLEURS is crowd-sourced read speech
(median ~28 dB SNR, p5 19 dB, worst 14 dB, 9% of clips under 20 dB); the podcasts are
studio-grade (~46-50 dB *before* our denoise step). Neither contains traffic, cafe
noise, phone-line compression, or reverb, so nothing in training resembles a demo run
on a laptop mic or over a phone.

```bash
python scripts/make_noisy_testset.py                                      # fixed-seed degraded test set
python scripts/run_day3_train.py --config configs/day3_lora_augmented.yaml
python scripts/run_day4_eval.py  --config configs/day3_lora_augmented.yaml \
    --manifest D:/Audion-Data/Urdu/open_datasets/manifest_noisy_test.jsonl --split test
```

`src/training/augment.py` synthesizes degradations rather than pulling a multi-GB noise
corpus: colored noise at 5-25 dB SNR, babble mixed from other clips in this same corpus,
a decaying-noise RIR for room reverb, and telephone band-limiting + mu-law. Two choices
that matter: augmentation is wired to the **train split only** (augmenting eval would
make a real gain indistinguishable from an easier test set), and `p_clean=0.35` keeps a
third of batches pristine so clean accuracy isn't traded away.

`make_noisy_testset.py` writes the degraded clips to disk with a fixed seed so every
model is scored on byte-identical audio, using a different seed from training so it
isn't a memorization test of noise already seen.

### Results

Overall, 336-clip test set (WER):

| audio | base large-v3 | combined | combined + augmentation |
|---|---|---|---|
| clean | 21.89% | 19.62% | **19.06%** |
| degraded | 31.90% | 27.52% | **25.40%** |

Better on **both**. The degraded gain (-2.1pp) was the goal; the clean gain (-0.6pp) is
a bonus -- the augmentation regularized rather than traded off.

Clean test by domain:

| subset | combined | + augmentation |
|---|---|---|
| FLEURS (299, human labels) | 20.55% | 20.64% |
| podcast (37) | 17.14% | **14.88%** |
| podcast English-heavy (17) | 17.86% | **12.62%** |

FLEURS is flat (0.09pp is noise) and code-switched audio improved markedly.

**`stt_v2_lora_augmented` is the model to ship.** Still untested on genuinely
in-the-wild audio -- the degraded set is synthetic, so it validates robustness to
noise/reverb/band-limiting but not to real room acoustics, real overlapping speakers,
or real phone codecs.

## Day 5: export for serving (2026-09-11)

```bash
python scripts/run_day5_export.py        --config configs/day3_lora_augmented.yaml
python scripts/run_day5_verify_export.py --config configs/day3_lora_augmented.yaml --split test
```

Merges the LoRA deltas into the base projections and converts to CTranslate2 --
faster-whisper's format, already a dependency here. Output:
`<training.output_dir>/export/ct2/`, 2.9GB fp16 (from a 5.8GB fp32 merged model),
self-contained with no PEFT or adapter loading at runtime.

```python
from faster_whisper import WhisperModel
model = WhisperModel(r"D:\Audion-Data\Urdu\checkpoints\stt_v2_lora_augmented\export\ct2",
                     device="cuda", compute_type="float16")
segments, info = model.transcribe("clip.wav", language="ur")
```

### Verified, not just built

| runtime | WER | CER |
|---|---|---|
| PyTorch (HF transformers) | 19.06% | 8.10% |
| CTranslate2 fp16 (served) | **18.45%** | **7.57%** |

No quantization loss -- slightly better, because faster-whisper's decoding defaults
(beam search, temperature fallback) differ from the plain HF `generate()` path used in
run_day4_eval.py and happen to suit real transcription better. Worth knowing when
comparing numbers across the two runtimes: they are not the same decoder.

**Throughput: 12.5x realtime** (66 min of audio in 318s, RTF 0.080) at 5.9GB VRAM.

### End to end

| model | clean WER | |
|---|---|---|
| base whisper-large-v3 | 22.77% | starting point |
| + FLEURS fine-tune | 20.65% | but destroyed code-switching (56% WER on English-heavy clips) |
| + podcast data | 19.62% | code-switching restored |
| + augmentation | 19.06% | and robust to noise/reverb/phone |
| + CT2 export | **18.45%** | deployable, 12.5x realtime |

~19% relative error reduction over base, code-switching preserved, noise robustness
added. Note the FLEURS row: on its own benchmark it looked like the second-best model,
while being the one that would have failed hardest in a real demo.

## Day 7: end-to-end integration test (2026-09-11)

`scripts/transcribe.py` is the inference entry point -- the deployable surface a caller
actually uses. Everything upstream works on pre-cut ~30s clips; this takes a whole
recording and lets faster-whisper do its own segmentation.

```bash
python scripts/transcribe.py audio.mp3                          # plain text
python scripts/transcribe.py audio.mp3 --format srt --out s.srt # subtitles
python scripts/transcribe.py audio.mp3 --format json --vad      # segments + timestamps
```

Tested on a raw, never-segmented 61-minute podcast MP3 -- the case the clip-level
evaluation could not cover:

| check | result |
|---|---|
| decodes raw MP3 directly | yes, no pre-conversion (faster-whisper's decoder, not torchcodec) |
| 61 min in a single call | yes, 1566 auto-segmented chunks |
| speed | **12.0x realtime** (61 min of audio in 305s) |
| language detection | correctly `ur` |
| timestamps monotonic | yes, no overlaps |
| coverage | 100% of audio, 0 empty segments |

Sample output, subtitle-ready:

```
[   0.20 ->    4.76] اس میں کوئی حرج نہیں ہے کہ بچہ انگریزی سیکھے گا
[  15.16 ->   18.12] ہم پرانے ڈرامے دیکھتے ہیں پی ٹی وی کے
```

Note `stdout` is written as raw UTF-8 bytes: the Windows console is cp1252 and raises
`UnicodeEncodeError` on Urdu through normal `print`.

## Code-switching: English words in Latin script (2026-09-14)

Whisper transliterates English into Urdu script when the surrounding speech is
Urdu-dominant (`ٹرگر` for "trigger"). The convention wanted here is Urdu words in Urdu
script, English words in Latin. Two layers, cheapest first.

**Layer 1 -- decode-time prompt.** `--codeswitch_prompt` passes an `initial_prompt`
mixing Urdu with Latin-script English, which biases the output style. Measured on the
demo lecture, same audio and model:

| | output |
|---|---|
| without | گناہ **ٹرگر** ہوتے ہیں … **فیزیکل** چیزوں |
| with | گناہ **trigger** ہوتے ہیں … **physical** چیزوں |

Free and needs no maintenance, but it is a bias, not a guarantee. The Urdu in the
prompt matters as much as the English: an all-English prompt pushes the model toward
translating rather than transcribing.

**Layer 2 -- dictionary.** `--fix_codeswitch` applies `configs/codeswitch_dict.json`
after decoding. Deterministic, and guarded three ways, because a wrong replacement is
worse than a missed one (a missed word merely looks untranslated; a wrong one corrupts
real Urdu):
- whole-word matching only -- verified that `ٹرگرانا`, `موبائلوں`, `ڈیزائنر` are left
  alone despite containing dictionary keys
- a `protected` list never converted: `منٹ`, `پروگرام`, `فون`, `ٹیم` and similar are
  naturalized written Urdu now. `منٹ` alone appeared 19 times in the 35-minute lecture;
  rendering it "minute" mid-sentence would be wrong, not right
- nothing is ever added automatically

```bash
python scripts/transcribe.py audio.mp3 --codeswitch_prompt --fix_codeswitch
```

### How the dictionary grows

`scripts/harvest_codeswitch.py` transcribes the audio **twice**, with and without the
prompt, and diffs the aligned word sequences. Where one run has Urdu script and the
other Latin for the same position, that pair is an entry -- discovered by the model, not
guessed. Verified working: on a 4-minute clip it recovered `ٹارگٹ -> target` and
`ڈیزائن -> design` unprompted, then correctly suppressed them as already known.

Candidates go to a text file for a human to approve, then
`scripts/apply_codeswitch_candidates.py` merges them (refusing any line still holding
the `????` placeholder, and reporting what it skipped).

**A heuristic version of this was built first and scrapped.** Ranking high-frequency
Latin words returned "you", "the", "and" -- already Latin, needing no entry. Flagging
Urdu words absent from a reference corpus returned `رمضان`, `دعا`, `قرآن`, `صاحب` --
ordinary Urdu, missing from FLEURS only because FLEURS is general-domain and the test
audio is a religious lecture. Both produced mostly noise, which is worse than nothing
when a human has to review it.

### Expected coverage

Measured held-out (dictionary built on some podcasts, tested on one never seen), not
in-sample:

| dictionary size | coverage on a *new* recording |
|---|---|
| 100 | ~27% |
| 300 | ~53% |
| 500 | ~66% |
| 1,000 | ~80% |
| 1,150 | ~85% |

Plateaus around 85-90%: half of all distinct English words in this project's data appear
exactly once (names, one-off terms), accounting for ~12% of occurrences. Combined with
the prompt layer, expect ~90% of English words rendered in Latin. The 2.9h of
code-switched podcast audio already on disk is enough to harvest ~1,200 terms.

Caveat: that curve comes from *podcast* speech, which is ~8% English by word. The demo
lecture was 0.6%, so for that kind of content there is simply far less to get wrong.

## Real-world evaluation (2026-09-14)

First measurement on audio that isn't FLEURS, the podcasts, or synthetic noise: a
35-minute Urdu religious lecture (M4A/AAC, mono 48kHz) supplied as the demo sample.

```bash
python scripts/transcribe.py "<audio>" --format json --out out.json
python scripts/make_correction_sheet.py --json out.json --audio "<audio>" --out CORRECT_ME.txt
# human corrects 5 x 2-minute chunks, then:
python scripts/score_correction_sheet.py --sheet CORRECTED_ME.txt \
    --model_json out.json --compare_json base.json
```

Recording quality sat between the two training corpora on SNR (27.3 dB, vs FLEURS ~28
and podcasts ~46-50) but with a **noise floor ~16 dB higher than FLEURS** (-46.7 vs
-62.7 dB) -- noisier than anything trained on.

| | WER | CER |
|---|---|---|
| fine-tuned (augmented, CT2) | **8.30%** | **4.10%** |
| base whisper-large-v3 | 21.68% | 13.73% |

Fine-tuned won all 5 chunks individually (3.7/8.8/6.4/10.2/11.2 vs 10.0/22.9/15.2/22.9/36.0).

### Read these numbers with the bias in mind

**The reference was produced by correcting the fine-tuned model's own transcript.** Any
error the human did not notice stays in the reference as "correct", so the fine-tuned
model scores right by construction there. 8.30% is therefore closer to *"how much the
human changed"* -- a floor on its true error rate -- and base is graded against its
competitor's wording rather than a neutral reference. Same circularity as the podcast
pseudo-labels, and it inflates the gap.

A concrete instance of the unfairness: in chunk 5 the fine-tuned model duplicated a
whole sentence, the duplication survived into the reference, and base -- which said it
once, correctly -- was penalised for the difference.

**The direction still holds**, on evidence independent of the scoring: base hallucinated
text over the silent first 6 seconds of the file (-43 dB, then looped `پچیس لاکھ`), and
repeated `دعاوں میں جنت بہت مانگو` three times at the start of chunk 5. Those are real
Whisper failure modes, not artifacts. Digit formatting was ruled out as an explanation
(only 8 digit tokens corpus-wide; stripping them moves WER 8.30 -> 8.33 and 21.68 ->
21.76).

Honest summary: the fine-tuned model is clearly better on this audio and comfortably
beats off-the-shelf Whisper, but quote it as roughly **88-92% of words correct**, not
91.7%. An unbiased figure would need the base model's transcript corrected
independently and both scored against their own references.

`score_correction_sheet.py` also handles a correction sheet filled into the wrong box --
it picks whichever of the two boxes diverges from the original transcript, so a
misfiled chunk is recovered rather than lost (chunk 3 of this run was filled in the
"what the model heard" box).

## Web UI (2026-09-14)

`webapp/` is a local FastAPI server plus a single page: upload a file (click or
drag-drop) or record from the mic, Urdu rendered right-to-left with per-segment
timestamps, download `.txt` / `.srt`, optional code-switch pass. Everything runs on this
machine -- audio goes to a temp file, is transcribed, and is deleted. Options and notes
are in `webapp/README.md`.

```bash
python webapp/server.py --preload          # then open http://127.0.0.1:8080
```

The server is Urdu-only (it fixes the language to `ur`) and loads the served `stt_v3`
model; it does not use the Arabic experiment models.

Three double-click launchers sit in the repo root:
`INSTALL_REQUIREMENTS.bat` (CUDA torch + faster-whisper + web deps, checks ffmpeg),
`START_WEBAPP.bat` (starts the server and opens the browser), and `TRANSCRIBE_FILE.bat`
(drag an audio file onto it; writes `<name>.txt` and `<name>.srt` next to the audio).

## Evaluating the model as served (2026-09-24)

`run_day4_eval.py` scores the *training* runtime (HF `generate()`), but what ships runs
through faster-whisper on the CT2 export, and the two decoders are not the same.
`scripts/eval_served.py` scores the served path over the same manifest splits and
reports read-aloud and conversational subsets separately (FLEURS-only numbers once hid
the code-switching regression). It reproduces the 18.45% WER / 7.57% CER above exactly
(read-aloud 20.42%, conversational 13.23%).

It also makes decoding settings testable. Sweep on the 336-clip test split:

| variant | WER |
|---|---|
| `vad` | **18.23%** (`transcribe.py --vad`; the web UI does not enable it) |
| `beam8` | 18.41% |
| baseline | 18.45% |
| `no_temp_fallback` | 18.45% (fallback never triggers on these clips) |
| `patience2` | 18.45% (no effect at beam 5) |
| `beam8_no_context` | 18.52% |
| `no_context` | 18.62% |
| `best_of5_no_context` | 18.62% |

The VAD gain is 25 fewer errors in 11,644 words and holds in 93% of bootstrap
resamples -- likely real, but small. Decoding tuning is close to exhausted; meaningful
Urdu gains need more training data, not decoding changes.

```bash
python scripts/eval_served.py --split test                       # baseline
python scripts/eval_served.py --split test --params '{"vad_filter": true}'
python scripts/eval_served.py --split test --sweep --limit 120   # compare variants
```

## More Urdu data: Common Voice, and `stt_v3` (2026-09-25)

Common Voice left Hugging Face in October 2025, so `configs/opendata_urdu.yaml`'s
`mozilla-foundation/common_voice_17_0` entry now holds only a README. The CC0 mirror
`fsicoli/common_voice_19_0` ships raw tar archives plus TSVs, so
`scripts/fetch_common_voice_mirror.py` reads it directly and writes a project-format
manifest. The mirror only carries audio for train/dev/test: for Urdu 17,212 of the
66,973 validated clips (20.8 h of 76.4 h). The rest needs a Mozilla Data Collective
account. 1,000 clips from CV test are held out as an extra read-speech benchmark.

`scripts/merge_cv_manifest.py` merges it with FLEURS + podcasts into
`manifest_combined_cv.jsonl`. CV would be ~87% of the data, so podcast *train* clips are
repeated 3x to keep conversational speech weighted -- the same read-speech-dominance
failure as the FLEURS-only run. Existing manifests are left untouched so earlier results
stay reproducible.

```bash
python scripts/fetch_common_voice_mirror.py
python scripts/merge_cv_manifest.py
python scripts/run_day3_train.py --config configs/day3_lora_cv.yaml
python scripts/run_day5_export.py --config configs/day3_lora_cv.yaml
```

`configs/day3_lora_cv.yaml` is the shipped recipe with two changes only (the new
manifest, and 2 epochs instead of 3 since there is ~3x more data), so differences are
attributable to the data. Acceptance rule: it replaces the shipped model only if
conversational (podcast) accuracy does not regress, not just the overall number.

Served-path WER (`eval_served.py`, VAD decoding), as recorded in the commit for this run:

| test set | `stt_v2_lora_augmented` | `stt_v3_lora_cv` |
|---|---|---|
| all | 18.23% | **17.87%** |
| podcast | 13.23% | **12.35%** |
| FLEURS | 20.13% | **19.96%** |
| CV holdout | 26.90% | **23.21%** |

Podcast improved, so the acceptance rule passes, and `stt_v3_lora_cv` is now
`DEFAULT_MODEL` in `scripts/transcribe.py` (the previous model is kept on disk). Read
the gains with care: the CV-holdout gain is the least meaningful row (the new model
trained on CV, and the Urdu holdout is random clips, not speaker-disjoint, so some
held-out speakers also appear in training), and the overall and FLEURS gains are small.

### GPU thermal pacing

This machine has crashed outright under GPU load before (see the crash post-mortem), and
the first attempt at this run had the RTX 5090 at 85 C with fans at 97%.
`src/training/thermal.py` adds a `ThermalGuard` callback, configured under `thermal:` in
the training YAML. Two layers: after each step it rests `gain_s_per_c` seconds per degree
above `target_c` (capped at `max_rest_s`) so the card settles near the target and the
fans stay steady rather than ramping; and a hard stop at `pause_at_c` waits until it falls
to `resume_at_c`. It fails open if `nvidia-smi` can't be read.

Measured: the Urdu run averaged 66 C with fans ~32%. On the Arabic run (below), over the
~2 h from step 1250 to 3000 the guard logged an average of 66 C and rested ~76 min in
total, while fans held at 31-33% in every sample taken. The 75 C hard stop did trigger --
75 hard pauses and a peak of 77 C, because temperature climbs ~5 C/s and is only checked
per step -- so the cost of a quiet run is time, not noise.

## Arabic fine-tune experiment (2026-09-25 to 2026-10-02)

Question: does the Urdu recipe (LoRA on large-v3 with Common Voice read speech) also
improve Arabic, in particular Gulf dialect? Acceptance rule, set before training: the
fine-tune only replaces base large-v3 for Arabic if Gulf accuracy does not regress,
measured on dialect test clips and not just the Common Voice holdout. Training on read
speech alone had already broken Urdu code-switching once, so this was a real risk.

### Data

Common Voice Arabic from the same CC0 mirror: 28,553 train / 10,352 dev / 10,474 test
clips, 26,648 kept. It is read speech, almost all Modern Standard Arabic (about ten clips
labelled Saudi/Gulf). The train split is 32.5 h from only 24 speakers, so
`fetch_common_voice_mirror.py --lang ar` adds two options: `--max_per_speaker 300` so the
model learns Arabic rather than those voices, and `--speaker_disjoint` to hold out whole
*speakers* from CV test. Result: 25,347 train clips (31.0 h, 943 speakers), 301
validation, 1,000 test, no speaker overlap. (The Urdu holdout is random clips, which
flatters its score.)

### Test sets

| set | clips | what it is |
|---|---|---|
| Saudi | 273 | Lahgtna v2 test split |
| Bahraini | 193 | Lahgtna v2 test split |
| Yemeni | 115 | Lahgtna v2 test split |
| Nexdata | 5 | Gulf sample in the TTS repo -- anecdotal at this size |
| CV holdout | 1,000 | speaker-disjoint Modern Standard Arabic read speech |

The three dialect sets come from `oddadmix/dialectal-arabic-lahgtna-v2`, built by
`fetch_gulf_eval.py` in the TTS repo into `D:\Audion-Data\Arabic\gulf_eval_lahgtna`,
outside every git repo. That dataset states no licence and re-hosts audio from other
people's channels, so it is used **for evaluation only**. Its transcripts are
AI-generated, so absolute WER against them is approximate; the signal is the
*comparison* between models on identical clips. Scoring is `eval_arabic.py` in the TTS
repo (faster-whisper on the CT2 export). It normalises text first -- drops diacritics and
punctuation, folds alef variants and `ى/ي`, `ؤ`, `ئ` -- so spelling variants don't count
as errors.

### Run 1: diacritized training text

`configs/day3_lora_ar_cv.yaml` -- same recipe as Urdu (LoRA r=8, lr 1e-4, 2 epochs,
augmentation, thermal pacing). Read Modern Standard Arabic improved (CV holdout 15.67 ->
12.40) but every Gulf set got worse (Saudi 17.61 -> 21.39, Bahraini 31.63 -> 34.39,
Yemeni 29.87 -> 32.10). Comparing outputs showed why: 23% of CV Arabic sentences are
written with full diacritics (tashkeel), and the model learned to switch into that formal
mode, producing diacritized, MSA-leaning text that mis-hears dialect (`مكان الذهب` became
`مَا كَانَتْ ذَهَبُ`). Everyday Arabic, the Gulf transcripts and the app are undiacritized.

### Run 2: diacritics stripped

`scripts/clean_arabic_manifest.py` removes diacritics and tatweel and re-attaches a
stand-alone `و` to the next word (`و كيف` -> `وكيف`): 9,834 of 26,648 transcripts changed,
nothing else. `configs/day3_lora_ar_cv_clean.yaml` is otherwise identical, and keeps every
250-step checkpoint so an earlier one can be tried if later steps drift toward formal
Arabic.

### Results (WER %)

| test set | base large-v3 | run 1 | run 2 final (3170) | run 2 ckpt-1000 | run 2 ckpt-500 |
|---|---|---|---|---|---|
| Saudi (273) | **17.61** | 21.39 | 19.47 | 18.69 | 18.25 |
| Bahraini (193) | **31.63** | 34.39 | 32.25 | 32.03 | 32.12 |
| Yemeni (115) | 29.87 | 32.10 | 31.71 | 30.00 | **29.66** |
| Nexdata (5) | **22.78** | 27.85 | 24.05 | 25.32 | 24.05 |
| CV holdout (1000) | 15.67 | 12.40 | **11.47** | 11.69 | 12.45 |

Whether the gaps are real, from a paired bootstrap over clips against base (positive =
worse than base; WER recomputed per clip with the scorer's normalisation, and it
reproduces every stored summary):

| Gulf subset | model | WER diff vs base | 95% CI |
|---|---|---|---|
| Saudi (273) | final | +1.85 | [+1.08, +2.64] |
| Saudi (273) | ckpt-500 | +0.63 | [-0.03, +1.34] |
| Bahraini (193) | final | +0.62 | [-0.40, +1.64] |
| Bahraini (193) | ckpt-500 | +0.49 | [-0.62, +1.60] |
| Yemeni (115) | final | +1.84 | [+0.64, +2.98] |
| Yemeni (115) | ckpt-500 | -0.22 | [-1.19, +0.77] |
| **Gulf pooled (581)** | **final** | **+1.49** | **[+0.92, +2.04]** |
| **Gulf pooled (581)** | **ckpt-1000** | **+0.71** | **[+0.17, +1.25]** |
| **Gulf pooled (581)** | **ckpt-500** | **+0.44** | **[-0.06, +0.97]** |

Reproduce with `scripts/bootstrap_vs_base.py`, which reads the per-clip JSON files and
checks its recomputed WER against each file's stored summary before resampling:

```bash
python scripts/bootstrap_vs_base.py --eval_dir D:/Audion-Data/Arabic/eval \
    --base base --candidates clean ckpt500 ckpt1000 --sets sa bh ye
```

### What it shows

- **Stripping diacritics helped but did not fix it.** Versus run 1, every Gulf set
  improved and the Saudi regression roughly halved (+3.8 -> +1.9 WER against base), but
  the final model is still significantly worse than base on Saudi and Yemeni and on the
  pooled Gulf set.
- **The drift grows with training steps.** Saudi 18.25 -> 18.69 -> 19.47 and Yemeni
  29.66 -> 30.00 -> 31.71 from step 500 to 1000 to 3170; Bahraini is flat. Most of the
  Modern Standard Arabic gain arrives early (12.45 at step 500 vs 11.47 at the end), so
  later steps mostly add Gulf cost. Likely cause: read-MSA-only training pulls the model
  toward formal Arabic -- an interpretation, not tested directly.
- **Checkpoint-500 is the only fine-tune near base on Gulf.** Pooled +0.44 WER, whose
  interval just touches zero, with Yemeni slightly better than base (CER too: 13.81 vs
  14.65) and read MSA ~3 points better. Strictly it is a hair worse on Saudi, and the
  5-clip Nexdata set is too small to lean on.
- **Selection bias.** ckpt-500 was picked after seeing these same Gulf sets, so its
  numbers are slightly flattering. Confirming it needs a Gulf test set it hasn't been
  compared on.

### Decision

Nothing is deployed and the Arabic model in use is unchanged. Under the acceptance rule
the final model fails. Checkpoint-500 is the one worth considering, and whether it is
worth using depends on traffic: mostly Modern Standard Arabic or mixed -> likely yes;
mostly Gulf dialect -> keep base large-v3.

### Reproduce

```bash
python scripts/fetch_common_voice_mirror.py --lang ar --max_per_speaker 300 --speaker_disjoint
python scripts/clean_arabic_manifest.py
python scripts/run_day3_train.py --config configs/day3_lora_ar_cv_clean.yaml

# export one checkpoint to its own folder (does not touch the final model's export)
python scripts/run_day5_export.py --config configs/day3_lora_ar_cv_clean.yaml \
    --adapter_dir D:/Audion-Data/Arabic/checkpoints/stt_ar_lora_cv_clean/checkpoint-500 \
    --out_root    D:/Audion-Data/Arabic/checkpoints/stt_ar_lora_cv_clean/export_ckpt500

# score it (eval_arabic.py lives in the TTS repo, D:\Audion-Data\TTS)
python eval_arabic.py --dir D:/Audion-Data/Arabic/gulf_eval_lahgtna/sa --language ar \
    --model "D:\Audion-Data\Arabic\checkpoints\stt_ar_lora_cv_clean\export_ckpt500\ct2" --out sa.json
```

Checkpoints are under `D:\Audion-Data\Arabic\checkpoints\` (`stt_ar_lora_cv`,
`stt_ar_lora_cv_clean`); per-clip results are in `D:\Audion-Data\Arabic\eval\` with
prefixes `base_`, `new_` (run 1), `clean_` (run 2 final), `ckpt500_`, `ckpt1000_`. The
helper shell scripts there are outside this repo.

### Operational notes

- **Resuming:** training auto-resumes from the latest checkpoint, so re-run the same
  command. This run was resumed twice: after the first launch stopped near step 1330 on
  2026-09-29 (cause not determined; no traceback in the log), and at step 3000 after a
  background-task time limit killed it at step 3014.
- **Don't run two exports against one folder.** `run_day5_export.py` writes fixed
  `merged_hf` and `ct2` paths and refuses to overwrite `ct2` without `--force`; use
  `--out_root` per run. A leftover waiter job and a manual launch once both exported and
  scored after training ended, which interleaved the log and produced a harmless
  `FileExistsError`. The numbers above come from the stored per-clip JSON files, not
  that log.

## Status against the 7-Day Plan (as of 2026-09-11)

> Snapshot from 2026-09-11. Since then the served model changed to `stt_v3_lora_cv` and
> the Arabic experiment was run -- see "Current state" at the top and the sections above.

The STT track is complete and deployable. Two STT items and the whole Emotion track are
not done -- recorded here rather than left implicit.

| Day | Task | Track | Status |
|---|---|---|---|
| 1 | Denoise, segment, diarize | Shared | done -- 373 clips, 2 speakers per podcast |
| 1 | Draft transcription (Whisper base) | Shared | done, then redone with large-v3 |
| 2 | Transcript correction & verification | STT | done via large-v3 re-transcription (pseudo-labels, see above) |
| 2 | **Accent tagging** (Karachi/Lahori/Punjabi) | STT | **NOT DONE** |
| 2 | Emotion labeling per segment | Emotion | partial -- blanket `neutral`, not a per-clip judgement |
| 2 | Train/val/test split 80/10/10 | Shared | done -- 2406/306/336 |
| 3 | Base model selection | STT | done -- whisper-large-v3 |
| 3 | LoRA/PEFT fine-tuning setup | STT | done |
| 3 | Kick off STT training | STT | done |
| 4 | Complete STT training | STT | done |
| 4 | Emotion classifier architecture | Emotion | not done (out of scope) |
| 4 | Emotion training pipeline | Emotion | not done (out of scope) |
| 5 | Complete emotion training | Emotion | not done (out of scope) |
| 5 | STT WER/CER evaluation | STT | done |
| 5 | **STT accent-wise breakdown** | STT | **NOT DONE** |
| 6 | Emotion accuracy/F1 per class | Emotion | not done (out of scope) |
| 6 | CTranslate2/ONNX export | STT | done -- 2.9GB CT2, verified 18.45% WER |
| 6 | Emotion model export | Emotion | not done (out of scope) |
| 7 | End-to-end integration test | Shared | done for STT; "both models" not applicable with no emotion model |
| 7 | Buffer: bug-fix, sign-off | Shared | done |

**Why the two STT gaps:** accent tagging needs someone who can distinguish
Karachi/Lahori/Punjabi-influenced Urdu by ear. FLEURS ships no accent labels and nothing
in this pipeline listens to audio, so neither the tagging nor the accent-wise report
could be produced. `run_day4_eval.py` notes the same thing at the top of the file.

**Why the Emotion track is absent:** scope was narrowed on 2026-09-08 to "Urdu, neutral
emotion only" for the demo. It is a deliberate decision, not an oversight -- but it does
mean Day 7's "both models in pipeline" is half-satisfied by construction.

### Biggest remaining risk

As of 2026-09-11, every number here came from FLEURS, 3 podcasts, and *synthetic* noise.
The one measurement since on genuinely different audio (the 35-minute lecture, see
"Real-world evaluation") used a reference built by correcting the model's own output, so
it understates error. The model has still never been scored on independently transcribed
audio from the actual demo environment. 10-15 minutes recorded through the real
mic/room/phone, with a rough human transcript, would convert "good model, assumed
conditions" into "good model, measured conditions". That remains the single
highest-value thing left to do for Urdu.

## Open items

- **Accent tagging and the accent-wise WER report** (Karachi/Lahori/Punjabi): need
  someone who can tell them apart by ear; nothing in this pipeline can.
- **Independent real-environment Urdu measurement**, as above.
- **Arabic decision:** choose between base large-v3 and checkpoint-500 (see "Arabic
  fine-tune experiment"), and if ckpt-500 is a candidate, confirm it on a Gulf test set
  not used to select it. Optionally score checkpoints 250 and 750 to see whether less
  training stays closer to base. If a better Arabic fine-tune is wanted, it probably needs
  dialectal training data rather than more read Modern Standard Arabic -- a hypothesis
  from the drift pattern, not something tested.
- **VAD in the web app:** `eval_served.py` found VAD the best decoding setting (18.23%),
  but the web UI does not enable it -- see "Evaluating the model as served". Turning it on
  is a one-line change if wanted.
- **Out of scope by decision (2026-09-08):** the Emotion track.

See `STT v2 7Day Plan.xlsx` for the original task list per day.

# STT v2 Fine-Tuning — Urdu Whisper

Local codebase for the STT v2 7-Day Plan (Auton8/DeepPulse pre-task, feeds into Audion's
self-hosted STT). Code lives here and is git-tracked; audio data lives alongside it in
`E:\Audion-Data\Urdu\` (not tracked in git — see `.gitignore`).

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

The two rows are **not directly comparable** -- different test sets. The like-for-like
comparisons (still to run) are `--domain fleurs` and `--domain podcast`, which answer
"did podcast data help or hurt on FLEURS?" and "did it improve podcast-like audio?"
respectively. The second is the one that matters for whether this exercise paid off.

## Roadmap (Days 2-7)

Not scaffolded yet — build each day's script once the prior day's output exists and the
approach is confirmed, rather than stubbing out unused code now. See
`STT v2 7Day Plan.xlsx` for the full task list per day (accent/emotion tagging, LoRA
fine-tuning, WER/CER + accent-wise eval, CTranslate2/ONNX export, integration test).

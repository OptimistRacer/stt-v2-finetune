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

Base model: `openai/whisper-large-v3`, LoRA on `q_proj`/`v_proj` (r=32, alpha=64) via
PEFT — ~1% of params trainable. Trains/evals against `manifest_opendata.jsonl`'s
`train`/`validation` splits (FLEURS's own split, not a separately-generated one).
Checkpoints and the final adapter land under `configs/day3_lora.yaml`'s
`training.output_dir` (`D:\Audion-Data\Urdu\checkpoints\stt_v2_lora\`, not
git-tracked).

Two dtype gotchas hit while building this, both fixed in `src/training/lora_setup.py`:
loading the model at fp32 explicitly is required (the whisper-large-v3 checkpoint is
stored in fp16, and `from_pretrained` now defaults to the checkpoint's native dtype)
because `TrainingArguments(bf16=True)` only autocasts train/eval forward passes, not
`model.generate()` during eval, which runs at the model's raw weight dtype instead —
so non-fp32 weights there mismatch against fp32 audio features.

Smoke-test a config change fast without waiting on a full epoch:
```bash
python scripts/run_day3_train.py --config configs/day3_lora.yaml --max_train_examples 8 --max_steps 2
```

### Post-mortem: the first full run collapsed (lr too high)

The first Day 3 run (`learning_rate: 1.0e-3`, the value in HF's own PEFT/Whisper LoRA
notebook) trained "successfully" -- loss dropped smoothly, 1.32 -> 0.45 -- but the
resulting adapter was badly broken: 259% WER, worse than the un-tuned base model's
22%. Root cause, found via `scripts/run_day4_eval.py`: teacher-forced training loss
doesn't catch this failure mode, because during training the decoder always sees the
*true* previous token. Free-running generation has no such crutch, and the adapter had
learned a shortcut that only works with that crutch -- given only its own predictions
to condition on, it degenerates into repeating a single token
(`تتتتتتتتت...`, `رررررر...`) after a few correct words. Confirmed it wasn't
"trained too long" either: `checkpoint-200` (1.5 epochs) was already *more* broken
than the final one (302% vs 259% WER), so cutting epochs wouldn't have fixed it.

Fix: `learning_rate: 1.0e-4` (10x lower) -- confirmed via a 15-minute diagnostic run
(200 examples, 40 steps) before committing another ~2.4h to a full retrain: adapter
beat the base model with no collapse (20.0% vs 21.95% WER). This is why
`src/evaluation/transcribe_eval.py` exists as a separate module from training-time
eval -- it's what caught this, by fixing `model.generation_config.language`/`task` up
front (Day 3 training's own eval had a *second*, unrelated bug: it never forces
language, so Whisper's per-call auto-detection produces noisy WER numbers regardless
of real model quality -- don't trust the WER Day 3 training prints, use
`run_day4_eval.py` on a saved checkpoint instead).

## Roadmap (Days 2-7)

Not scaffolded yet — build each day's script once the prior day's output exists and the
approach is confirmed, rather than stubbing out unused code now. See
`STT v2 7Day Plan.xlsx` for the full task list per day (accent/emotion tagging, LoRA
fine-tuning, WER/CER + accent-wise eval, CTranslate2/ONNX export, integration test).

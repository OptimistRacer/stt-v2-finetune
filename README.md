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

## Roadmap (Days 2-7)

Not scaffolded yet — build each day's script once the prior day's output exists and the
approach is confirmed, rather than stubbing out unused code now. See
`STT v2 7Day Plan.xlsx` for the full task list per day (accent/emotion tagging, LoRA
fine-tuning, WER/CER + accent-wise eval, CTranslate2/ONNX export, integration test).

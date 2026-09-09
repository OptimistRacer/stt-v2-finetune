"""Batched transcription + WER/CER scoring against a manifest split.

Split out from src/training/lora_setup.py's train-time eval: that path runs
generate() through Seq2SeqTrainer without ever forcing a language on the model's
generation_config, which lets Whisper fall back to per-call language
auto-detection -- flaky enough (esp. against a barely-trained LoRA) to produce
garbage transcripts that score 200%+ WER regardless of how well training actually
went. Forcing language/task here once, up front, is the fix.

Incremental writes + retry (2026-09-09): a full-validation-set eval run crashed this
machine outright (nvlddmkm driver faults in the Windows event log, then an unclean
reboot) partway through, losing all progress since nothing was written until the very
end. `transcribe_records` now appends each batch to `out_path` as it completes and can
resume past whatever's already there, and retries a batch after clearing the CUDA
cache before giving up on it -- most transient CUDA errors won't recur after a cache
clear, and even a hard crash now only loses the one in-flight batch, not the whole run.
"""
import json
import time
from pathlib import Path

import jiwer
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def force_language(model, language: str, task: str) -> None:
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None


# See src/training/lora_setup.py's module docstring: this machine crashed outright
# (driver fault -> unclean reboot) three times, always inside model.generate(), never
# during plain forward/backward. eager attention + disabling TF32 avoids the unstable
# fused kernel path that's the prime suspect.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def load_base_model(base_model: str, language: str, task: str, device: str = "cuda"):
    model = WhisperForConditionalGeneration.from_pretrained(
        base_model, dtype="float32", attn_implementation="eager"
    ).to(device)
    force_language(model, language, task)
    model.eval()
    return model


def load_lora_model(base_model: str, adapter_dir: str, language: str, task: str, device: str = "cuda"):
    from peft import PeftModel

    base = WhisperForConditionalGeneration.from_pretrained(
        base_model, dtype="float32", attn_implementation="eager"
    )
    model = PeftModel.from_pretrained(base, adapter_dir).to(device)
    force_language(model, language, task)
    model.eval()
    return model


def _load_done_ids(out_path: Path) -> dict[str, dict]:
    if not out_path.exists():
        return {}
    done = {}
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done[rec["clip_id"]] = rec
    return done


@torch.no_grad()
def transcribe_records(
    model,
    processor: WhisperProcessor,
    records: list[dict],
    out_path: Path,
    sample_rate: int = 16000,
    batch_size: int = 4,
    device: str = "cuda",
    max_new_tokens: int = 225,
    max_retries: int = 2,
) -> list[dict]:
    """Transcribes `records`, appending each batch to `out_path` (JSONL) as it
    completes -- a crash mid-run loses at most one in-flight batch, not everything.
    Records already present in `out_path` are skipped (resume), and a batch that
    raises a CUDA error gets one cache-clear + retry before being logged as failed
    and skipped, rather than taking down the whole run."""
    done = _load_done_ids(out_path)
    pending = [r for r in records if r["clip_id"] not in done]
    if done:
        print(f"  resuming: {len(done)} already done, {len(pending)} left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as out_f:
        for i in range(0, len(pending), batch_size):
            batch = pending[i : i + batch_size]
            audios = []
            for rec in batch:
                audio, sr = sf.read(rec["path"], dtype="float32")
                if sr != sample_rate:
                    raise ValueError(f"{rec['path']} is {sr}Hz, expected {sample_rate}Hz")
                audios.append(audio)

            inputs = processor.feature_extractor(audios, sampling_rate=sample_rate, return_tensors="pt")
            input_features = inputs.input_features.to(device)

            hyps = None
            for attempt in range(max_retries + 1):
                try:
                    generated_ids = model.generate(input_features, max_new_tokens=max_new_tokens)
                    hyps = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    break
                except Exception as e:
                    print(f"  batch at {i} failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    torch.cuda.empty_cache()
                    time.sleep(2)

            if hyps is None:
                print(f"  batch at {i} failed after {max_retries + 1} attempts, skipping {[r['clip_id'] for r in batch]}")
                for rec in batch:
                    out_f.write(json.dumps({**rec, "hypothesis": None, "failed": True}, ensure_ascii=False) + "\n")
                out_f.flush()
                continue

            for rec, hyp in zip(batch, hyps):
                out_f.write(json.dumps({**rec, "hypothesis": hyp.strip()}, ensure_ascii=False) + "\n")
            out_f.flush()
            print(f"  transcribed {len(done) + i + len(batch)}/{len(records)}")

    all_results = list(_load_done_ids(out_path).values())
    return [r for r in all_results if not r.get("failed")]


def score(results: list[dict]) -> dict:
    refs = [r["draft_transcript"] for r in results]
    hyps = [r["hypothesis"] for r in results]
    return {
        "n": len(results),
        "wer": round(100 * jiwer.wer(refs, hyps), 2),
        "cer": round(100 * jiwer.cer(refs, hyps), 2),
    }

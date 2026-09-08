"""Batched transcription + WER/CER scoring against a manifest split.

Split out from src/training/lora_setup.py's train-time eval: that path runs
generate() through Seq2SeqTrainer without ever forcing a language on the model's
generation_config, which lets Whisper fall back to per-call language
auto-detection -- flaky enough (esp. against a barely-trained LoRA) to produce
garbage transcripts that score 200%+ WER regardless of how well training actually
went. Forcing language/task here once, up front, is the fix.
"""
from pathlib import Path

import jiwer
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def force_language(model, language: str, task: str) -> None:
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None


def load_base_model(base_model: str, language: str, task: str, device: str = "cuda"):
    model = WhisperForConditionalGeneration.from_pretrained(base_model, dtype="float32").to(device)
    force_language(model, language, task)
    model.eval()
    return model


def load_lora_model(base_model: str, adapter_dir: str, language: str, task: str, device: str = "cuda"):
    from peft import PeftModel

    base = WhisperForConditionalGeneration.from_pretrained(base_model, dtype="float32")
    model = PeftModel.from_pretrained(base, adapter_dir).to(device)
    force_language(model, language, task)
    model.eval()
    return model


@torch.no_grad()
def transcribe_records(
    model,
    processor: WhisperProcessor,
    records: list[dict],
    sample_rate: int = 16000,
    batch_size: int = 8,
    device: str = "cuda",
    max_new_tokens: int = 225,
) -> list[dict]:
    """Returns records augmented with a `hypothesis` field."""
    results = []
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        audios = []
        for rec in batch:
            audio, sr = sf.read(rec["path"], dtype="float32")
            if sr != sample_rate:
                raise ValueError(f"{rec['path']} is {sr}Hz, expected {sample_rate}Hz")
            audios.append(audio)

        inputs = processor.feature_extractor(audios, sampling_rate=sample_rate, return_tensors="pt")
        input_features = inputs.input_features.to(device)

        generated_ids = model.generate(input_features, max_new_tokens=max_new_tokens)
        hyps = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for rec, hyp in zip(batch, hyps):
            results.append({**rec, "hypothesis": hyp.strip()})
        print(f"  transcribed {min(i + batch_size, len(records))}/{len(records)}")
    return results


def score(results: list[dict]) -> dict:
    refs = [r["draft_transcript"] for r in results]
    hyps = [r["hypothesis"] for r in results]
    return {
        "n": len(results),
        "wer": round(100 * jiwer.wer(refs, hyps), 2),
        "cer": round(100 * jiwer.cer(refs, hyps), 2),
    }

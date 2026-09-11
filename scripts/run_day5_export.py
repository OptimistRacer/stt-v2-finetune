"""Day 5: export the fine-tuned model for serving.

Two stages:
1. Merge the LoRA adapter into the base weights, producing a standalone Whisper model.
   A LoRA adapter isn't directly servable -- it needs PEFT plus the base model at
   runtime -- and merging costs nothing at inference since the deltas fold into the
   existing projection matrices.
2. Convert the merged model to CTranslate2, which is what faster-whisper serves. This
   repo already uses faster-whisper for draft transcription, so the serving stack is
   the one it's been running all along.

Merging happens in fp32 and quantization is applied at conversion time rather than
before it: merging low-rank deltas into fp16 weights compounds rounding error, and the
conversion step can quantize once from clean fp32 values instead.

Usage:
    python scripts/run_day5_export.py --config configs/day3_lora_augmented.yaml
    python scripts/run_day5_export.py --config configs/day3_lora_augmented.yaml --quantization int8_float16
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from peft import PeftModel
from transformers import WhisperForConditionalGeneration

from src.training.lora_setup import load_processor

torch.backends.cuda.matmul.allow_tf32 = False


def find_converter() -> str:
    """ct2-transformers-converter isn't always on PATH (pip installs it to a Scripts dir
    that Windows often leaves off PATH), so fall back to the known install location."""
    exe = shutil.which("ct2-transformers-converter")
    if exe:
        return exe
    for cand in Path(sys.executable).parent.glob("**/ct2-transformers-converter*"):
        return str(cand)
    import site
    for base in site.getusersitepackages(), *site.getsitepackages():
        for cand in Path(base).parent.glob("Scripts/ct2-transformers-converter*"):
            return str(cand)
    raise FileNotFoundError(
        "ct2-transformers-converter not found. It ships with ctranslate2; "
        "try `pip install ctranslate2` or invoke it via its full path."
    )


def run(config_path: str, adapter_dir: str | None, out_root: str | None, quantization: str, force: bool) -> None:
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    train_out = Path(cfg["training"]["output_dir"])
    adapter = Path(adapter_dir) if adapter_dir else train_out / "final_adapter"
    root = Path(out_root) if out_root else train_out / "export"
    merged_dir = root / "merged_hf"
    ct2_dir = root / "ct2"

    if not adapter.exists():
        raise FileNotFoundError(f"No adapter at {adapter}")

    print(f"[export] base      : {cfg['base_model']}")
    print(f"[export] adapter   : {adapter}")
    print(f"[export] output    : {root}")

    print("[export] merging LoRA into base weights (fp32)")
    base = WhisperForConditionalGeneration.from_pretrained(cfg["base_model"], dtype="float32")
    peft_model = PeftModel.from_pretrained(base, str(adapter))
    merged = peft_model.merge_and_unload()

    # Pin language/task so a served model defaults to Urdu transcription instead of
    # falling back to Whisper's per-call language auto-detection, which is what produced
    # the meaningless WER numbers during Day 3 training (see README).
    merged.generation_config.language = cfg["language"]
    merged.generation_config.task = cfg["task"]
    merged.generation_config.forced_decoder_ids = None

    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir))

    # Save the feature extractor and tokenizer separately rather than via
    # WhisperProcessor.save_pretrained: on transformers 5.x the combined processor
    # writes `processor_config.json`, but ct2-transformers-converter looks for the
    # older `preprocessor_config.json`, and conversion fails without it.
    processor = load_processor(cfg["base_model"], cfg["language"], cfg["task"])
    processor.feature_extractor.save_pretrained(str(merged_dir))
    processor.tokenizer.save_pretrained(str(merged_dir))
    print(f"[export] merged HF model -> {merged_dir}")

    del base, peft_model, merged
    torch.cuda.empty_cache()

    if ct2_dir.exists():
        if not force:
            raise FileExistsError(f"{ct2_dir} exists; pass --force to overwrite")
        shutil.rmtree(ct2_dir)

    print(f"[export] converting to CTranslate2 (quantization={quantization})")
    cmd = [
        find_converter(),
        "--model", str(merged_dir),
        "--output_dir", str(ct2_dir),
        "--quantization", quantization,
        "--copy_files", "tokenizer.json", "preprocessor_config.json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise RuntimeError(f"ct2 conversion failed (exit {proc.returncode})")

    print(f"[done] CTranslate2 model -> {ct2_dir}")
    print("\nServe with faster-whisper:")
    print("    from faster_whisper import WhisperModel")
    print(f'    model = WhisperModel(r"{ct2_dir}", device="cuda", compute_type="float16")')
    print('    segments, info = model.transcribe("clip.wav", language="ur")')


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/day3_lora_augmented.yaml")
    p.add_argument("--adapter_dir", default=None, help="Defaults to <training.output_dir>/final_adapter")
    p.add_argument("--out_root", default=None, help="Defaults to <training.output_dir>/export")
    p.add_argument("--quantization", default="float16",
                   help="CTranslate2 quantization: float16 (default), int8_float16, int8, float32")
    p.add_argument("--force", action="store_true", help="Overwrite an existing ct2 directory")
    a = p.parse_args()
    run(a.config, a.adapter_dir, a.out_root, a.quantization, a.force)

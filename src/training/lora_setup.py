"""Load Whisper large-v3 and wrap its attention projections with LoRA adapters."""
from peft import LoraConfig, get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def load_processor(base_model: str, language: str, task: str) -> WhisperProcessor:
    return WhisperProcessor.from_pretrained(base_model, language=language, task=task)


def load_lora_model(base_model: str, lora_cfg: dict):
    # Forced to fp32: the whisper-large-v3 checkpoint itself is stored in fp16, and
    # `from_pretrained` defaults to that native dtype rather than fp32. Loading at fp16
    # (or explicitly bf16) hits the same problem either way -- TrainingArguments
    # (bf16=True) autocasts train/eval forward passes uniformly, but model.generate()
    # during eval runs outside that autocast, so non-fp32 weights there raise a dtype
    # mismatch against the fp32 audio features.
    model = WhisperForConditionalGeneration.from_pretrained(base_model, dtype="float32")
    # Language/task are fixed per training run (Urdu/transcribe) via the processor's
    # tokenizer prompt instead of generation-time forced_decoder_ids -- required for
    # newer transformers versions, which deprecated forced_decoder_ids.
    model.generation_config.forced_decoder_ids = None

    peft_config = LoraConfig(
        r=lora_cfg.get("r", 32),
        lora_alpha=lora_cfg.get("alpha", 64),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model

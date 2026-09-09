"""Load Whisper large-v3 and wrap its attention projections with LoRA adapters.

Stability note (2026-09-09): this machine crashed outright (Windows event log shows
nvlddmkm driver faults followed by an unclean reboot) three separate times, every time
inside a batched model.generate() call -- twice in standalone eval, once inside
Seq2SeqTrainer's own predict_with_generate eval during training. Never once during
plain forward/backward. Root cause looks like fused cuBLASLt/SDPA attention kernels
being unstable on this RTX 5090 (Blackwell, sm_120) under this driver/torch version --
one crash log shows a `CUBLAS_STATUS_INTERNAL_ERROR` warning immediately before the
fatal error. `attn_implementation="eager"` forces the unfused attention path
everywhere a Whisper model is loaded in this repo, trading some speed for not taking
the whole machine down. See also: run_day3_train.py disables predict_with_generate
entirely (moves all generate()-based eval to the crash-resumable run_day4_eval.py),
and torch.backends.cuda.matmul.allow_tf32 is disabled at the same call sites.
"""
from peft import LoraConfig, get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def load_processor(base_model: str, language: str, task: str) -> WhisperProcessor:
    return WhisperProcessor.from_pretrained(base_model, language=language, task=task)


def load_lora_model(base_model: str, lora_cfg: dict):
    # bf16, default (fused/SDPA) attention -- NOT eager (2026-09-09 correction): all
    # 3 crashes on this machine happened inside model.generate() (twice standalone eval,
    # once Trainer's own predict_with_generate), never once during plain forward/
    # backward. Forcing eager attention onto training too -- reasonable-looking at the
    # time -- was solving a problem training never had, while creating a new one: eager
    # materializes the full O(seq_len^2) attention score matrix per head per layer
    # instead of a fused/memory-efficient kernel, and Whisper's encoder always
    # processes a fixed 30s/1500-token sequence regardless of actual clip length. That
    # pushed VRAM to ~98% and caused allocator-thrashing (GPU-reported 100%
    # "utilization" but power draw collapsed to ~115-130W) at both batch=8/fp32 and
    # batch=4/bf16 -- switching dtype didn't fix it because dtype was never the actual
    # lever, attn_implementation was. Training uses default attention here; eager stays
    # in src/evaluation/transcribe_eval.py, where the actual generate()-related crash
    # risk lives.
    model = WhisperForConditionalGeneration.from_pretrained(base_model, dtype="bfloat16")
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

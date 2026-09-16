"""First real test: synthesize new Urdu text in the confirmed host's voice, zero-shot,
using the F5-TTS-OpenBible-Urdu model. No fine-tuning -- just a reference clip + its
transcript, per the model card.
"""
import torch
from huggingface_hub import hf_hub_download
from hydra.utils import get_class
from omegaconf import OmegaConf
from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder, preprocess_ref_audio_text

repo_id = "multilingual-tts/F5-TTS-OpenBible-Urdu"
ckpt = "f5tts_model/model_last.pt"
vocab = "f5tts_model/vocab.txt"
config = "f5tts_model/F5TTS_v1_Base_Open_Bible_Urdu.yaml"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[test] device: {device}")

model_cfg = OmegaConf.load(config)
model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")

print("[test] loading vocoder...")
vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)

print("[test] loading F5-TTS Urdu model...")
model = load_model(
    model_cls, model_cfg.model.arch, ckpt,
    mel_spec_type="vocos", vocab_file=vocab, use_ema=True, device=device,
)

# Best reference clip found via diarization + speaker-ID confirmation: clean, 9.6s,
# confirmed to be the consented host (clip 048, "How Great Leaders Think Under Pressure").
ref_audio = "host_clips/How Great Leaders Think Under Press__048.wav"
ref_text = "پھر کومنٹ کر کے بتائیں کہ آپ کو اس اپیسوڈ کا کون سا پارٹ سب سے زیادہ پسند ہے۔ اور آخری ہمیں اگلے گیس کا سجیشن ضرور دیں۔"

# Held-out test sentence -- not from any training/reference audio, to see how the
# cloned voice generalizes to genuinely new content.
gen_text = "آج ہم ایک نئے موضوع پر بات کریں گے جو ہمارے سامعین کے لیے بہت اہم ہے۔ امید ہے آپ کو یہ اپیسوڈ پسند آئے گا۔"

print("[test] preprocessing reference...")
ref_audio_proc, ref_text_proc = preprocess_ref_audio_text(ref_audio, ref_text)

print("[test] synthesizing...")
wav, sr, _ = infer_process(
    ref_audio_proc, ref_text_proc, gen_text, model, vocoder,
    mel_spec_type="vocos", device=device,
)

import soundfile as sf
out_path = "test_output_01.wav"
sf.write(out_path, wav, sr)
print(f"[done] -> {out_path} ({len(wav)/sr:.1f}s @ {sr}Hz)")

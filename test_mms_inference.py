"""Zero-shot-no-more test: synthesize new Urdu text with the fine-tuned host-voice
MMS-TTS (VITS) model. Text is held out -- not present in the training transcripts.
"""
from transformers import pipeline
import scipy.io.wavfile

model_id = "D:/Audion-Data/TTS/vits_finetuned_host"
print("[test] loading fine-tuned model...")
synthesiser = pipeline("text-to-speech", model_id, device=0)

gen_text = "آج ہم ایک نئے موضوع پر بات کریں گے جو ہمارے سامعین کے لیے بہت اہم ہے۔ امید ہے آپ کو یہ اپیسوڈ پسند آئے گا۔"

print("[test] synthesizing...")
speech = synthesiser(gen_text)

audio = speech["audio"]
print("[test] audio shape:", audio.shape)
if audio.ndim > 1:
    audio = audio.squeeze()

out_path = "D:/Audion-Data/TTS/test_output_mms_finetuned.wav"
scipy.io.wavfile.write(out_path, rate=speech["sampling_rate"], data=audio)
print(f"[done] -> {out_path} ({len(audio) / speech['sampling_rate']:.1f}s @ {speech['sampling_rate']}Hz)")

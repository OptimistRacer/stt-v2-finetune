"""Diagnostic: feed the synthesized TTS audio back through our own working Urdu STT
model. If the STT can transcribe it back to something close to the intended text,
the TTS audio contains real recognizable speech even if it sounds rough to the ear.
If STT also fails, the TTS output itself is genuinely broken.
"""
import json
import sys

sys.path.insert(0, r"C:\Users\Sistems-AI\stt-v2-finetune")
from scripts.transcribe import transcribe, CODESWITCH_PROMPT
from src.postprocess.codeswitch import CodeSwitcher

INTENDED_TEXT = (
    "آج ہم ایک نئے موضوع پر بات کریں گے جو ہمارے سامعین کے لیے بہت اہم ہے۔ "
    "امید ہے آپ کو یہ اپیسوڈ پسند آئے گا۔"
)

FILES = [
    "D:/Audion-Data/TTS/test_output_mms_finetuned.wav",       # pass 1: 1800 steps
    "D:/Audion-Data/TTS/test_output_mms_finetuned_v2.wav",    # pass 2: 9000 steps
]

cs = CodeSwitcher()
results = []
for f in FILES:
    r = transcribe(
        f, language="ur", device="cuda", compute_type="float16",
        beam_size=5, vad=False, initial_prompt=CODESWITCH_PROMPT, prompt_chunk_s=0,
    )
    text = cs.apply(r["text"])
    results.append({
        "file": f,
        "stt_transcript": text,
        "duration_s": r["duration_s"],
        "n_segments": r["n_segments"],
    })

with open("D:/Audion-Data/TTS/stt_check_results.json", "w", encoding="utf-8") as out:
    json.dump({"intended_text": INTENDED_TEXT, "results": results}, out, ensure_ascii=False, indent=2)

print("[done] -> D:/Audion-Data/TTS/stt_check_results.json")

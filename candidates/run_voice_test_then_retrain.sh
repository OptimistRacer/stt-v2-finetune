#!/bin/bash
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export PATH="/c/Users/Sistems-AI/AppData/Local/Microsoft/WinGet/Links:$PATH"
C=/d/Audion-Data/TTS/candidates
cd $C
echo "=== voices: meta"; python gen_meta.py 2>&1 | grep -E "\[meta\]|Error|Traceback" | tail -3
for n in omnivoice voxcpm chatterbox; do
  echo "=== voices: $n"; $C/venv_$n/Scripts/python.exe gen_candidate.py $n 2>&1 | grep -E "\[$n\]|Error|Traceback|error" | tail -15
done
echo "=== scoring"; python score_voices.py 2>&1 | grep -E "\[score\]|Error|Traceback"
echo "=== VOICE TEST DONE, starting cleaned Arabic retrain"
cd /c/Users/Sistems-AI/stt-v2-finetune
python scripts/run_day3_train.py --config configs/day3_lora_ar_cv_clean.yaml > /d/Audion-Data/Arabic/train_ar_cv_clean.log 2>&1
echo "=== RETRAIN EXITED $?"

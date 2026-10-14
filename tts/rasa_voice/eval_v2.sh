#!/bin/bash
# Evaluate Parvaaz v2 checkpoints (male + female training) against v1 step 300, the voice in
# the app. Run after run_v2_pipeline.sh has finished training:
#     bash eval_v2.sh
# For each v2 checkpoint: the 6 shared + 20 held-out sentences, two takes (seeds 0 and 1),
# described as Parvaaz exactly as the app asks; then score_parvaaz.py adds them to
# eval_out/scores.json next to v1's s300 / s300_seed1 / s300_seed2.
set -e -o pipefail
cd /d/Audion-Data/TTS/rasa_voice
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
P=../candidates/venv_parler/Scripts/python.exe

powershell -NoProfile -ExecutionPolicy Bypass -File make_eval_models.ps1 -Run parvaaz_v2_train -Prefix v2_step
tags=()
for d in eval_models/v2_step*; do
  n=${d##*v2_step}
  for seed in 0 1; do
    tag=v2s${n}_seed$seed
    if [ "$(ls ../candidates/out/parvaaz_$tag/*.wav 2>/dev/null | wc -l)" -lt 6 ]; then  # skip if already made
      (cd ../candidates && TTS_SEED=$seed venv_parler/Scripts/python.exe gen_candidate.py parvaaz_$tag \
          D:/Audion-Data/TTS/rasa_voice/$d 2>&1 | { grep -E "done|Error|Traceback" || true; })
    fi
    TTS_SEED=$seed $P gen_heldout.py $tag $d 2>&1 | { grep -E "done|Error|Traceback" || true; }
    tags+=($tag)
  done
done
python score_parvaaz.py "${tags[@]}" 2>&1 | { grep -v -E "Warning|warn" || true; }

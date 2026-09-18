#!/usr/bin/env bash
# Wait for the Iqbal voice training run to finish, then sweep its checkpoints for
# clarity (WER) and voice match. Detached via nohup so it outlives tool timeouts.
cd /d/Audion-Data/TTS
LOG=finetune-hf-vits/train_iqbal_voice.log
until grep -q "Training / Inference Done" "$LOG" 2>/dev/null || grep -q "Traceback" "$LOG" 2>/dev/null; do
  sleep 20
done
if grep -q Traceback "$LOG"; then
  echo "[chain] TRAINING FAILED"
  exit 1
fi
echo "[chain] training done, starting sweep"
PATH="/d/ffmpeg-shared-dl/ffmpeg-9.0.1-full_build-shared/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  python sweep_checkpoints.py --run_dir D:/Audion-Data/TTS/vits_iqbal_voice \
    --out_dir D:/Audion-Data/TTS/sweep_iqbal_voice --n_samples 2 \
    --ref_speaker_dir D:/Audion-Data/TTS/iqbal_dataset/train
echo "[chain] sweep finished"

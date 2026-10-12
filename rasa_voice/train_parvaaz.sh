#!/bin/bash
# Fine-tune Indic Parler-TTS (Apache-2.0) on the Rasa Urdu male speaker (CC-BY-4.0).
# Recipe follows Parler-TTS's single-speaker fine-tuning notebook (Jenny), adapted:
#   - local datasets: rasa_male_hf (audio + text) joined by id to rasa_male_desc (descriptions)
#   - bfloat16 on the RTX 5090; no in-training eval (its WER judge is English Whisper --
#     scoring is done afterwards with our own Urdu STT)
#   - parler-tts/training/run_parler_tts_training.py carries a local patch for the DAC codec
#
#   bash train_parvaaz.sh smoke     # 10 steps on the 50-clip sample
#   bash train_parvaaz.sh full      # Parvaaz v1 (male only)
#   bash train_parvaaz.sh v2        # Parvaaz v2 (male + female)
set -e
MODE=${1:-smoke}
R=D:/Audion-Data/TTS/rasa_voice
PY=/d/Audion-Data/TTS/candidates/venv_parler/Scripts/python.exe
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 WANDB_MODE=disabled

if [ "$MODE" = smoke ]; then
  DATA=$R/smoke_hf; DESC=$R/smoke_desc; OUT=$R/smoke_train; CODES=$R/smoke_codes
  RUN=(--max_steps 10 --gradient_accumulation_steps 1 --warmup_steps 2 --logging_steps 1 --save_steps 10)
elif [ "$MODE" = v2 ]; then
  # Parvaaz v2: Rasa Urdu male (Parvaaz) + female ("Sana", training data only), ~2x the data.
  DATA=$R/rasa_mf_hf; DESC=$R/rasa_mf_desc; OUT=$R/parvaaz_v2_train; CODES=$R/parvaaz_v2_codes
  RUN=(--num_train_epochs ${EPOCHS:-2} --gradient_accumulation_steps 8 --warmup_steps 100 --logging_steps 10
       --save_steps 300 --save_total_limit 12)
else
  DATA=$R/rasa_male_hf; DESC=$R/rasa_male_desc; OUT=$R/parvaaz_train; CODES=$R/parvaaz_codes
  # Effective batch 4 x 8 = 32 clips; 2 epochs of 10,473 clips is ~655 updates. A checkpoint
  # every 150 lets earlier ones be compared by ear, in case later steps over-fit.
  RUN=(--num_train_epochs ${EPOCHS:-2} --gradient_accumulation_steps 8 --warmup_steps 100 --logging_steps 10
       --save_steps 150 --save_total_limit 12)
fi

cd /d/Audion-Data/TTS/parler-tts
$PY training/run_parler_tts_training.py \
  --model_name_or_path ai4bharat/indic-parler-tts --feature_extractor_name ai4bharat/indic-parler-tts \
  --description_tokenizer_name google/flan-t5-large --prompt_tokenizer_name ai4bharat/indic-parler-tts \
  --report_to none --overwrite_output_dir false \
  --train_dataset_name $DATA --train_metadata_dataset_name $DESC --train_dataset_config_name default --train_split_name train \
  --id_column_name id --target_audio_column_name audio --description_column_name description --prompt_column_name text \
  --max_duration_in_seconds 20 --min_duration_in_seconds 2.0 --max_text_length 400 --preprocessing_num_workers 1 \
  --do_train true --per_device_train_batch_size 4 --gradient_checkpointing true \
  --learning_rate 1e-4 --adam_beta1 0.9 --adam_beta2 0.99 --weight_decay 0.01 --lr_scheduler_type constant_with_warmup \
  --freeze_text_encoder true --audio_encoder_per_device_batch_size 4 --dtype bfloat16 --seed 456 \
  --output_dir $OUT --temporary_save_to_disk ${CODES}_tmp --save_to_disk $CODES \
  --dataloader_num_workers 0 "${RUN[@]}"

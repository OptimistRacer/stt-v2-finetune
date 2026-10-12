#!/bin/bash
# Parvaaz v2, unattended: wait for the Rasa Urdu female shards, then build the two-speaker
# dataset, measure it, write descriptions and train. Started detached by run_v2_pipeline.cmd
# (training alone outlasts any interactive session); progress in v2_pipeline.log, training
# detail in train_v2.log.
set -e -o pipefail
cd /d/Audion-Data/TTS/rasa_voice
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
DS=/d/Audion-Data/TTS/venv_dataspeech/Scripts/python.exe
log() { echo "[$(date '+%F %T')] $*"; }

log "download any missing Rasa Urdu shards (safe alongside another downloader: HF file locks)"
$DS -c "
import json, time
from pathlib import Path
from huggingface_hub import hf_hub_download, get_token
base = Path(r'D:\Audion-Data\TTS\rasa_urdu')
rows = json.loads((base / 'shard_index.json').read_text())
for r in rows:
    if not set(r['genders']) & {'Male', 'Female'} or (base / r['file']).exists():
        continue
    for attempt in range(8):
        try:
            hf_hub_download('ai4bharat/Rasa', r['file'], repo_type='dataset', local_dir=str(base), token=get_token())
            print('got', r['file'], flush=True); break
        except Exception as e:
            print('retry', r['file'], type(e).__name__, flush=True); time.sleep(15 * (attempt + 1))
    else:
        raise SystemExit('could not download ' + r['file'])
"

log "build dataset (male + female)"
$DS prepare_rasa_male.py --genders Male Female --out rasa_mf_hf | tail -40
log "measure pitch, noise, room sound"
$DS measure_rasa.py --src rasa_mf_hf --out rasa_mf_tags 2>&1 | { grep -v -E "it/s|s/it|examples/s|Warning|warn|Using default" || true; } | tail -20
log "write descriptions"
$DS describe_rasa.py --tags rasa_mf_tags --out rasa_mf_desc > desc_v2_log.txt 2>&1
head -c 1500 rasa_mf_desc/summary.json; echo
log "train v2 (log: train_v2.log)"
bash train_parvaaz.sh v2 >> train_v2.log 2>&1
log "v2 pipeline done"

#!/usr/bin/env bash
# 006_prep2: Tatoeba を加えて再分割。test は Wikipedia/青空文庫は前回と同じ記事、Tatoeba が増える。
set -uo pipefail
cd ~/horizon; . .venv/bin/activate
export TOKENIZERS_PARALLELISM=false JOB=006_prep2
mkdir -p runs/006_prep2; rm -f data/proc/READY
python punct/prep.py 2>&1 | tee runs/006_prep2/prep.log

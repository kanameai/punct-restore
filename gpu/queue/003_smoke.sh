#!/usr/bin/env bash
# 003_smoke: 学習コードの動作確認（200 ステップ、使い捨て）。002 が成功していることが前提。
set -uo pipefail
cd ~/horizon; . .venv/bin/activate
[ -f data/proc/READY ] || { echo "data/proc/READY が無い（002 失敗）"; exit 1; }
export TOKENIZERS_PARALLELISM=false JOB=003_smoke NAME=smoke MAX_STEPS=200 EVAL_EVERY=100
mkdir -p runs/003_smoke
python punct/train.py 2>&1 | tee runs/003_smoke/train.log

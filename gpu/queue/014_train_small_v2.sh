#!/usr/bin/env bash
# 014_train_small_v2: mmBERT-small（MIT, 約 140M）で同じ設定。軽量版。
set -uo pipefail
cd ~/horizon; . .venv/bin/activate
[ -f data/proc/READY ] || { echo "prep2 失敗"; exit 1; }
export TOKENIZERS_PARALLELISM=false JOB=014_train_small_v2 NAME=punct-small-v2 BASE=jhu-clsp/mmBERT-small EPOCHS=2 BS=64 LR=1e-4 AUG=0.25 EVAL_EVERY=4000 MAX_HOURS=6
mkdir -p runs/014_train_small_v2
python punct/train.py 2>&1 | tee runs/014_train_small_v2/train.log

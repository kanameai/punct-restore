#!/usr/bin/env bash
# 008_train_base_v1: Wikipedia＋青空文庫＋Tatoeba(CC0)、25% で断片切り出し。2 エポック。
set -uo pipefail
cd ~/horizon; . .venv/bin/activate
[ -f data/proc/READY ] || { echo "prep2 失敗"; exit 1; }
export TOKENIZERS_PARALLELISM=false JOB=008_train_base_v1 NAME=punct-base-v1 EPOCHS=2 BS=64 LR=5e-5 AUG=0.25 EVAL_EVERY=4000 MAX_HOURS=8
mkdir -p runs/008_train_base_v1
python punct/train.py 2>&1 | tee runs/008_train_base_v1/train.log

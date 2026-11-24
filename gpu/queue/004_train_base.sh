#!/usr/bin/env bash
# 004_train_base: mmBERT-base を 2 エポック学習（約 70 万段落、数時間の見込み）。smoke が通った場合のみ。
set -uo pipefail
cd ~/horizon; . .venv/bin/activate
[ -f runs/003_smoke/test_metrics.json ] || { echo "smoke が成功していない"; exit 1; }
export TOKENIZERS_PARALLELISM=false JOB=004_train_base NAME=punct-base-v0 EPOCHS=2 BS=64 LR=5e-5 MAX_HOURS=18
mkdir -p runs/004_train_base
python punct/train.py 2>&1 | tee runs/004_train_base/train.log

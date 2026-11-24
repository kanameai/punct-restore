#!/usr/bin/env bash
# 019: Wikipedia(CC-BY-SA) を学習に使わない保険版（青空文庫＋Tatoeba のみ）。test は同じ（Wikipedia test は未学習ドメインでの評価になる）。
set -uo pipefail
cd ~/horizon; . .venv/bin/activate
[ -f data/proc/train_nowiki.jsonl ] || { echo "no train_nowiki"; exit 1; }
export TOKENIZERS_PARALLELISM=false JOB=019_train_base_nowiki NAME=punct-base-nowiki TRAIN_FILE=data/proc/train_nowiki.jsonl EPOCHS=3 BS=64 LR=5e-5 AUG=0.25 EVAL_EVERY=4000 MAX_HOURS=6
mkdir -p runs/019_train_base_nowiki
python punct/train.py 2>&1 | tee runs/019_train_base_nowiki/train.log

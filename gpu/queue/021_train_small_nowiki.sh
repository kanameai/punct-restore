#!/usr/bin/env bash
# 021: small の Wikipedia 抜き版（base の nowiki と同じ条件）。
set -uo pipefail
cd ~/horizon; . .venv/bin/activate
[ -f data/proc/train_nowiki.jsonl ] || { echo "no train_nowiki"; exit 1; }
export TOKENIZERS_PARALLELISM=false JOB=021_train_small_nowiki NAME=punct-small-nowiki BASE=jhu-clsp/mmBERT-small TRAIN_FILE=data/proc/train_nowiki.jsonl EPOCHS=3 BS=64 LR=1e-4 AUG=0.25 EVAL_EVERY=4000 MAX_HOURS=5
mkdir -p runs/021_train_small_nowiki
python punct/train.py 2>&1 | tee runs/021_train_small_nowiki/train.log

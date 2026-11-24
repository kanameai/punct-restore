#!/usr/bin/env bash
# 007_eval_v0: v0（Wikipedia＋青空文庫のみで学習）を新しい test（Tatoeba 含む）で評価。会話調への一般化を見る。
set -uo pipefail
cd ~/horizon; . .venv/bin/activate
mkdir -p runs/007_eval_v0
python punct/eval_model.py models/punct-base-v0 runs/007_eval_v0/test_metrics.json 2>&1 | tee runs/007_eval_v0/eval.log

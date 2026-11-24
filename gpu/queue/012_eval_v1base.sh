#!/usr/bin/env bash
# v1(base) を新しい test（日本語など Tatoeba が増える）で評価。v2 との比較の基準。
set -uo pipefail
cd ~/horizon; . .venv/bin/activate
mkdir -p runs/012_eval_v1base
python punct/eval_model.py models/punct-base-v1 runs/012_eval_v1base/test_metrics.json 2>&1 | tee runs/012_eval_v1base/eval.log

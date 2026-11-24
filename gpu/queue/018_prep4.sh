#!/usr/bin/env bash
# 018: prep を再実行（test/dev は同一。Wikipedia を除いた train_nowiki.jsonl を追加で作る）。
set -uo pipefail
cd ~/horizon; . .venv/bin/activate
export TOKENIZERS_PARALLELISM=false JOB=018_prep4
mkdir -p runs/018_prep4; python punct/prep.py 2>&1 | tee runs/018_prep4/prep.log

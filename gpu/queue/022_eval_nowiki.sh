#!/usr/bin/env bash
# 022: nowiki 版 2 つを他と同じ条件（各 300 段落）で評価し、restore() のデモも取る。
set -uo pipefail
cd ~/horizon; . .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
mkdir -p runs/022_eval_nowiki
N=300 python punct/eval_model.py models/punct-base-nowiki runs/022_eval_nowiki/base_nowiki_n300.json 2>&1 | tail -2
N=300 python punct/eval_model.py models/punct-small-nowiki runs/022_eval_nowiki/small_nowiki_n300.json 2>&1 | tail -2
for m in punct-base-nowiki punct-small-nowiki; do python punct/demo.py models/$m runs/022_eval_nowiki/demo_$m.md 2>&1 | tail -3; done

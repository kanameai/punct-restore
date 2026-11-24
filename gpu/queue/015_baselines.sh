#!/usr/bin/env bash
# 015_baselines: 同じ test（各ファイル先頭 N 段落）で (a) こちらの v2 base、(b) oliverguhr、(c) punctuators pcs_47lang を測る。
# (c) は依存を壊さないよう別 venv（.venv-pcs）。失敗しても続行。
set -uo pipefail
cd ~/horizon; . .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
mkdir -p runs/015_baselines
N=300 python punct/eval_model.py models/punct-base-v2 runs/015_baselines/ours_base_v2_n300.json 2>&1 | tail -3
N=300 python punct/eval_model.py models/punct-small-v2 runs/015_baselines/ours_small_v2_n300.json 2>&1 | tail -3
N=300 OUTJSON=runs/015_baselines/oliverguhr.json python punct/baseline_oliverguhr.py 2>&1 | tail -5
deactivate
python3 -m venv .venv-pcs && . .venv-pcs/bin/activate && pip install -q --upgrade pip && pip install -q punctuators 2>&1 | tail -3
pip freeze > runs/015_baselines/pcs_freeze.txt
N=300 python punct/baseline_pcs.py 2>&1 | tail -30 | tee runs/015_baselines/pcs.log

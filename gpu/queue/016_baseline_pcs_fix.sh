#!/usr/bin/env bash
# 016: 015 の punctuators 評価は位置合わせのバグ（空白の扱い・<unk>）で過小評価だったため、位置合わせを直して再測定。
set -uo pipefail
cd ~/horizon; . .venv-pcs/bin/activate
mkdir -p runs/016_baseline_pcs_fix
N=300 OUTJSON=runs/016_baseline_pcs_fix/pcs47.json python punct/baseline_pcs.py 2>&1 | tail -30 | tee runs/016_baseline_pcs_fix/pcs.log

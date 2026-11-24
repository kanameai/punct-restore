#!/usr/bin/env bash
# 017: punctuators の Wikipedia 成績が低すぎる原因（長文の切り捨て・位置ずれ・評価の誤り）を確認する診断。
set -uo pipefail
cd ~/horizon; . .venv-pcs/bin/activate
mkdir -p runs/017_diag_pcs
python punct/diag_pcs.py 2>&1 | tail -20 | tee runs/017_diag_pcs/diag.log

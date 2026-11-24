#!/usr/bin/env bash
# 002_prep: 分割・ラベル生成の検証（トークナイザの天井）と、既存モデルの基準評価。
set -uo pipefail
cd ~/horizon; . .venv/bin/activate
mkdir -p runs/002_prep; export TOKENIZERS_PARALLELISM=false
date -u +%FT%TZ
python punct/prep.py 2>&1 | tee runs/002_prep/prep.log
python punct/baseline_oliverguhr.py 2>&1 | tee runs/002_prep/baseline.log || echo "baseline failed (non-fatal)"
date -u +%FT%TZ

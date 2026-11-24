#!/usr/bin/env bash
# 005_tatoeba: Tatoeba の CC0 文を取得して会話調データを作る（CC0 のみ。CC-BY 部分は使わない）。
set -uo pipefail
cd ~/horizon; . .venv/bin/activate
mkdir -p runs/005_tatoeba
python punct/tatoeba.py 2>&1 | tee runs/005_tatoeba/tatoeba.log

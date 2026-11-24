#!/usr/bin/env bash
# 020: 公開用 restore() の動作確認（手書き入力の出力、評価コードとの一致、速度、サイズ）。base v2 / small v2。
set -uo pipefail
cd ~/horizon; . .venv/bin/activate
mkdir -p runs/020_demo
for m in punct-base-v2 punct-small-v2; do python punct/demo.py models/$m runs/020_demo/$m.md 2>&1 | tail -30; done

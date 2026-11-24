#!/usr/bin/env bash
# 010_tatoeba_full: Tatoeba 全文エクスポート（CC BY 2.0 FR。出典表記が条件）。CC0 版では日本語・独・西等がほぼ空だったため。
set -uo pipefail
cd ~/horizon; . .venv/bin/activate
export JOB=010_tatoeba_full TATOEBA_URL=https://downloads.tatoeba.org/exports/sentences.tar.bz2
mkdir -p runs/010_tatoeba_full
python punct/tatoeba.py 2>&1 | tee runs/010_tatoeba_full/tatoeba.log

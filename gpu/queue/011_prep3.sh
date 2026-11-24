#!/usr/bin/env bash
set -uo pipefail
cd ~/horizon; . .venv/bin/activate
export TOKENIZERS_PARALLELISM=false JOB=011_prep3
mkdir -p runs/011_prep3; rm -f data/proc/READY
python punct/prep.py 2>&1 | tee runs/011_prep3/prep.log

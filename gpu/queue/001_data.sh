#!/usr/bin/env bash
# 001_data: 句読点・大文字小文字復元の元テキストを取得（公開・非gated のみ）。000 の venv が前提。
# 出典: wikimedia/wikipedia (CC-BY-SA-3.0 + GFDL), globis-university/aozorabunko-clean (CC-BY-4.0; 本文は著作権切れ)
set -uo pipefail
cd ~/horizon
. .venv/bin/activate
OUT=runs/001_data
mkdir -p "$OUT" data/raw
exec > >(tee "$OUT/data.log") 2>&1
date -u +%FT%TZ
python - <<'PY'
import json, re, itertools, os
from datasets import load_dataset
N_PER_LANG = 60000        # 1 言語あたりの段落数上限
LANGS = ["ja", "en", "de", "fr", "es", "it", "pt", "nl", "ru", "zh", "ko"]
stats = {}
def para_iter(text):
    for p in re.split(r"\n+", text):
        p = p.strip()
        if 40 <= len(p) <= 600: yield p
for lg in LANGS:
    path = f"data/raw/wiki_{lg}.jsonl"
    n = 0
    try:
        ds = load_dataset("wikimedia/wikipedia", f"20231101.{lg}", split="train", streaming=True)
        with open(path, "w") as f:
            for row in ds:
                for p in itertools.islice(para_iter(row["text"]), 3):
                    f.write(json.dumps({"lang": lg, "src": "wikipedia", "title": row["title"], "text": p}, ensure_ascii=False) + "\n")
                    n += 1
                if n >= N_PER_LANG: break
        stats[f"wiki_{lg}"] = n
    except Exception as e:
        stats[f"wiki_{lg}"] = f"ERR {e!r}"[:200]
try:
    ds = load_dataset("globis-university/aozorabunko-clean", split="train", streaming=True)
    n = 0
    with open("data/raw/aozora_ja.jsonl", "w") as f:
        for row in ds:
            for p in itertools.islice(para_iter(row["text"]), 20):
                f.write(json.dumps({"lang": "ja", "src": "aozora", "title": row["meta"]["作品名"] if "meta" in row and isinstance(row["meta"], dict) and "作品名" in row["meta"] else "", "text": p}, ensure_ascii=False) + "\n")
                n += 1
            if n >= 60000: break
    stats["aozora_ja"] = n
except Exception as e:
    stats["aozora_ja"] = f"ERR {e!r}"[:200]
os.makedirs("runs/001_data", exist_ok=True)
json.dump(stats, open("runs/001_data/stats.json", "w"), indent=2, ensure_ascii=False)
print(stats)
PY
# 小さなサンプルを戻す（各ファイル先頭 5 行）
for f in data/raw/*.jsonl; do head -n 5 "$f" > "$OUT/sample_$(basename "$f")"; done
du -sh data/raw > "$OUT/size.txt"
date -u +%FT%TZ

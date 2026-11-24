# punct-restore

Multilingual **punctuation and capitalization restoration** for unpunctuated, lower-case text (ASR post-processing).
11 languages: Japanese, Chinese, Korean, English, German, French, Spanish, Italian, Portuguese, Dutch, Russian.
Fine-tuned from [mmBERT](https://huggingface.co/jhu-clsp/mmBERT-base) (MIT). Code: MIT.

日本語: 句読点なし・小文字の文（音声認識の出力など）に、句読点と語頭の大文字を戻す小型モデルとその学習コードです。

## Models (Hugging Face, MIT)

| model | params | note |
|---|---|---|
| [kanameaijp/punct-restore-multilingual-base](https://huggingface.co/kanameaijp/punct-restore-multilingual-base) | 308M | |
| [kanameaijp/punct-restore-multilingual-small](https://huggingface.co/kanameaijp/punct-restore-multilingual-small) | 141M | lighter; about 1 point lower micro-F1 on in-domain sets |

Both were trained on Aozora Bunko and Tatoeba only (no Wikipedia). Evaluation tables, per-class scores and limitations are in the model cards.
**Evaluated only on our own test sets (held-out Aozora/Tatoeba, plus Wikipedia as out-of-domain); not on real ASR output yet.**

## Quick start

```bash
pip install torch transformers
python publish/punct_restore.py kanameaijp/punct-restore-multilingual-small "今日は朝から雨が降っていましたが午後には止んだので公園まで散歩に出かけました"
# 今日は朝から雨が降っていましたが、午後には止んだので、公園まで散歩に出かけました。
```

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer
from punct_restore import restore
name = "kanameaijp/punct-restore-multilingual-small"
tok = AutoTokenizer.from_pretrained(name); model = AutoModelForTokenClassification.from_pretrained(name).eval()
print(restore(model, tok, "what time does the train to berlin leave and do i need to change in hannover"))
```

## How it works

Each token gets one of 12 labels = {none, comma, period, question, exclamation, colon/semicolon} × {lower, upper-case word start}.
Training text has its punctuation removed and is lower-cased automatically (labels come from the original text; no model-generated labels).
Long inputs are split into 250-token windows. Details: `gpu/punct/common.py`.

## Reproducing the published models

The scripts are the job files we ran, in numeric order, on one RTX A6000 (48 GB). Copy `gpu/` to `~/horizon/` on a machine with an NVIDIA GPU (CUDA 12) and run:

```bash
cd ~/horizon
bash queue/000_env.sh            # venv + torch (cu126) + pinned libraries (requirements.txt)
bash queue/001_data.sh           # Wikipedia (test sets only for the published models) + Aozora Bunko
bash queue/010_tatoeba_full.sh   # Tatoeba sentences (CC BY 2.0 FR)
bash queue/018_prep4.sh          # split by article/work/sentence group; writes data/proc/train_nowiki.jsonl
bash queue/019_train_base_nowiki.sh    # ~1.4 h
bash queue/021_train_small_nowiki.sh   # ~45 min
bash queue/022_eval_nowiki.sh    # evaluation + demo check
```

The other jobs (002–017) are the experiments that led here: a first model with Wikipedia (`013`, `014`), baselines (`015`, `016`: oliverguhr/fullstop-punctuation-multilang-large and punctuators `pcs_47lang`), diagnostics.
The data and license record is in `NOTES/data_licenses.md`. Seeds are fixed but GPU non-determinism means numbers can differ slightly.

## Data and licenses

| source | license |
|---|---|
| Aozora Bunko, cleaned ([globis-university/aozorabunko-clean](https://huggingface.co/datasets/globis-university/aozorabunko-clean)) | CC BY 4.0 (texts are public domain) |
| [Tatoeba](https://tatoeba.org/en/downloads) sentences | CC BY 2.0 FR — © Tatoeba contributors |
| Wikipedia (`wikimedia/wikipedia`) — used for test sets and for the optional `-wiki` experiments only | CC BY-SA 3.0 / GFDL |

Base model mmBERT: MIT, © JHU-CLSP. This repository's code: MIT. See `NOTES/data_licenses.md` for the full record.

## Limitations

No evaluation on real speech transcripts; no quotes/brackets/ellipses; `!` and `:` are weak; acronyms get only the first letter capitalized; Japanese comma placement is subjective.
Comparison with other open models is in the model cards, with the caveat that our test sets favour models trained on matching data.

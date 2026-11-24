---
license: mit
language: [ja, zh, ko, en, de, fr, es, it, pt, nl, ru]
base_model: jhu-clsp/mmBERT-__BASE__
pipeline_tag: token-classification
library_name: transformers
tags: [punctuation, truecasing, asr, post-processing, multilingual, japanese]
---

# __NAME__

Restores **punctuation and word-initial capitalization** in unpunctuated, lower-case text such as speech-recognition (ASR) output.
Fine-tuned from [jhu-clsp/mmBERT-__BASE__](https://huggingface.co/jhu-clsp/mmBERT-__BASE__) (MIT) as a token classifier (__PARAMS__ parameters, __SIZE_MB__ MiB) for 11 languages:
Japanese, Chinese, Korean, English, German, French, Spanish, Italian, Portuguese, Dutch, Russian.

日本語の要約: 句読点なし・小文字の文（音声認識の出力など）に、句読点（、。？！ など）と語頭の大文字を戻すモデルです。日本語・中国語・韓国語・欧州 7 言語・ロシア語に対応。
評価は自分たちが作ったテスト（Wikipedia・青空文庫・Tatoeba）だけで、**実際の ASR 出力での評価はまだしていません**。

## Usage

```python
# pip install torch transformers   (checked with transformers 5.24 / torch 2.14)
from transformers import AutoModelForTokenClassification, AutoTokenizer
from punct_restore import restore          # punct_restore.py is in this repository's files

name = "kanameaijp/__NAME__"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForTokenClassification.from_pretrained(name).eval()
print(restore(model, tok, "今日は朝から雨が降っていましたが午後には止んだので公園まで散歩に出かけました"))
# 今日は朝から雨が降っていましたが、午後には止んだので、公園まで散歩に出かけました。
print(restore(model, tok, "what time does the train to berlin leave and do i need to change in hannover"))
```

The model predicts, for every token, the punctuation mark that follows it and whether the token starts with a capital letter (12 labels = 6 punctuation classes × 2).
Rendering (`、。？！` for Japanese, `，。？！` for Chinese, ASCII otherwise) is done by `restore()` according to the script of the input.
Texts longer than ~250 tokens are processed in separate windows (no overlap), so sentence context is lost at window borders.
Speed (10 short sentences, including first call): CPU 0.2–0.3 s, one RTX A6000 0.17 s.

## Labels

`O`, `COMMA`, `PERIOD`, `QUESTION`, `EXCLAIM`, `COLON` (colon and semicolon are merged) × `lc` / `uc`.
Not covered: quotation marks, brackets, hyphens, ellipses, Spanish inverted marks (`¿ ¡`), and capitalization other than the first letter of a word (e.g. `NASA`, `iPhone` → `Nasa`, `Iphone`).

## Training data

Text comes from the sources below; the labels are generated automatically by removing punctuation from the original text and lower-casing it (no model-generated or closed-model labels). The original punctuation is the target.

__DATA_TABLE__

25% of the training paragraphs are randomly cut into fragments so that the model also sees text that starts or ends mid-sentence.
Base model: mmBERT-__BASE__ (MIT). Training: __EPOCHS__ epochs, batch 64, lr __LR__, bf16, one RTX A6000 (__HOURS__).
__DATA_NOTE__

## Evaluation

Held-out test sets built the same way as the training data (split by article / work / sentence group, no overlap).
Metric: character-position micro-F1 over the five punctuation classes (a mark counts as correct only if class and position match); cap-F1 is the F1 of "word starts upper-case" (languages with case only).
**__EVAL_NOTE__.**

### Per test set

__MAIN__

### Per class (all test sets pooled, base)

__CLS__

Exclamation marks are rarely predicted correctly; colons/semicolons are mixed. Commas are the hardest class (their placement is partly a matter of style, especially in Japanese literary text).

### Comparison with other open models (same inputs, first 300 paragraphs per test set)

__CMP__

How to read this: the baselines were not trained on these sources, and the test sets follow our own punctuation conventions and our input format (lower-cased, punctuation removed).
`oliverguhr/fullstop-punctuation-multilang-large` (MIT) only supports en/de/fr/it/nl and does not predict `!` or capitalization; it was given the cased text (an advantage for it).
`punctuators` `pcs_47lang` (Apache-2.0) is a general 47-language punctuation / true-casing / sentence-splitting model; its output was aligned to the input automatically (checked on a sample: ≥97% of characters in sync; `<unk>` dropped), and it is not tuned to our conventions.
__CMP_NOTE__
We do not claim that this model is better in general; it was only evaluated here, on data that is closer to its own training data than to the baselines'.

## Limitations

- No evaluation on real ASR output (recognition errors, filler words, spoken Japanese without clear sentence ends, code-switching) yet.
- __JA_LIMIT__ Comma placement is subjective; the model follows the most frequent style of its training data.
- Chinese: simplified and traditional both occur in the data; no separate evaluation.
- Korean/Chinese/Japanese capitalization is not applicable; `?` / `!` are rare in the formal parts of the data.
- Names, acronyms and mixed-case words are capitalized at the first letter only.
- Do not use for legal/medical transcripts without human review.

## Attribution and license

__LICENSE_NOTE__
Code and training scripts: https://github.com/kanameai/punct-restore

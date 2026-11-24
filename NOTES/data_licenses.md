# データ・ベースモデルのライセンス記録
確認日 2026-11-24。方法: HF API（huggingface.co/api/{models,datasets}/...）の tags/cardData と gated フラグ。★は私が API で直接確認、☆はまだ要確認。

## ベースモデル
| 名前 | ライセンス | gated | 備考 |
|---|---|---|---|
| jhu-clsp/mmBERT-base ★ | MIT | false | 約 30 万 DL。Horizon Labs も同系を使用。多言語 ModernBERT。再配布・派生公開 OK。 |
| jhu-clsp/mmBERT-small ★ | MIT | false | 約 6.8 万 DL。small 版のベース（2026-11-24 確認）。 |

## 学習データ（句読点・大文字小文字復元の元テキスト）
| データ | ライセンス | gated | 用途・メモ |
|---|---|---|---|
| wikimedia/wikipedia (20231101.*) ★ | CC-BY-SA-3.0 ＋ GFDL | false | 11 言語の段落。出典表記必須。share-alike が重みに及ぶかは未決（hypotheses.md §4）。 |
| globis-university/aozorabunko-clean ★ | CC-BY-4.0（データセット整理版）。本文は著作権切れの青空文庫 | false | 日本語の文語・小説。現代口語は少ない。出典表記をモデルカードに。 |
| HuggingFaceFW/fineweb-2 ★ | ODC-By | false | **今回は使わない**（Common Crawl 由来で個々のページの権利が不明確なため）。 |
| Tatoeba「Sentences (CC0)」エクスポート ★ | CC0 1.0（tatoeba.org/en/downloads に「CC0 で提供される文を全て含む」と記載） | 公開 | https://downloads.tatoeba.org/exports/sentences_CC0.tar.bz2 （約 8MB）。会話調の短文。CC-BY 2.0 FR の全文エクスポートは使わない。 |
| Helsinki-NLP/tatoeba | CC-BY-2.0 | false | 使わない（上の CC0 版を直接使う）。 |
| PleIAs/common_corpus ☆ | cardData に記載なし（文書ごとに許諾） | false | 使うなら文書ごとのライセンス列で PD/CC0/CC-BY に絞る。今回は未使用。 |
| nvidia/Nemotron-PII ★ | CC-BY-4.0 | false | 英語のみ。PII 題材（見送り）用の参考。 |
| Helsinki-NLP/europarl ★ | unknown | false | 使わない（ライセンス不明）。 |
| unicamp-dnf/mmarco | API が 401（gated/要認証） | — | 使わない。 |

## 生成物について
- ラベルは元テキストの句読点・大文字小文字から機械的に作る。閉じたモデルの出力・合成データは今回使わない。

## 基準モデル（評価のみ・再配布なし）
- oliverguhr/fullstop-punctuation-multilang-large ★ MIT。評価用に重みを読み込むだけで、学習データには使っていない。

## 2026-11-24 追記: Tatoeba 全文エクスポート
- https://downloads.tatoeba.org/exports/sentences.tar.bz2 — tatoeba.org/en/downloads に「These files are released under CC BY 2.0 FR」。CC-BY は PROTOCOL で許可された種類。
  条件は出典表記（Tatoeba contributors）。モデルカードと GitHub README に「Tatoeba (CC BY 2.0 FR) の文で学習」と書く。
- CC0 エクスポートは en/fr/ru 以外ほぼ空だった（ja 2 文、de 46 文）ため、全文版を使う。一部の文は Tanaka Corpus 由来（Tatoeba の説明では public domain）。

## 日本語 ASR（2026-11-24 開始。確認済み/未確認を区別）
評価セット
- google/fleurs ★ CC-BY-4.0、gated=false。ja_jp の test/dev/train は parquet（parquet-data/ja_jp/）。**test は評価専用。学習に使わない。**
ベースモデル候補（HF API の tag、gated=false を確認 ★）
- openai/whisper-large-v3 ★ apache-2.0 ／ openai/whisper-large-v3-turbo ★ mit ／ kotoba-tech/kotoba-whisper-v2.0 ★ apache-2.0
- reazon-research/reazonspeech-nemo-v2 ★ apache-2.0（NeMo 要）／ nvidia/parakeet-tdt_ctc-0.6b-ja ★ cc-by-4.0（NeMo 要）
- いずれも「ベースの学習データの権利」までは確認していない（Whisper の学習データは非公開）。公開時は各ベースのライセンスに従う旨を書く。
学習用音声データ（☆ 未確認。次回以降に確認してから使う）
- Common Voice ja（CC0 のはずだが、配布先と gated 状況を確認） / FLEURS ja train（CC-BY-4.0） / JSUT・JVS（CC BY-SA 系、条件を確認） / ReazonSpeech（TV 由来。原則避ける）
- 話者の権利（音声は個人の声）にも注意。出典とライセンスを確かめるまで取得しない。

- japanese-asr/ja_asr.common_voice_8_0（評価専用・非 gated, 2026-11-24 確認 ★）: Common Voice 8.0 ja の test を kotoba 系が再配布。リポジトリにライセンス欄は無いが、元の Common Voice は CC0。**学習には使わない・再配布しない**。
- japanese-asr/ja_asr.jsut_basic5000 / ja_asr.reazonspeech_test: ライセンス欄なし（reazonspeech_test は TV 由来）。使わない。

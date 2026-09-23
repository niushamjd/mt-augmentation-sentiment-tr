# Data card

This project's data (PROJECT_INSTRUCTIONS.md Section 3.5). Raw downloads live under `data/raw/` (gitignored, not committed — see repo root `README.md` for how to regenerate). Everything below is committed since it's small and reproducibility-critical.

## Real Turkish sentiment data — `data/sentiment/`

**Source**: [`fthbrmnby/turkish_product_reviews`](https://huggingface.co/datasets/fthbrmnby/turkish_product_reviews) on Hugging Face — "Turkish Product Reviews," 235,165 reviews "collected online," curated by [Fatih Barmanbay](https://github.com/fthbrmnby/turkish-text-data).

**Not Trendyol.** The team initially tried the Kaggle dataset `sedayazici/trendyol-urun-yorumlari-duygu-analizi` (actual Trendyol reviews) and rejected it: its stopword removal was too aggressive, stripping enough words that many sentences lost their meaning — too noisy to use. `fthbrmnby/turkish_product_reviews` was chosen instead. Its own dataset card doesn't name a specific source site ("Source Data: [More Information Needed]"), so the project does **not** claim these are Trendyol reviews — report and README language should say "Turkish product reviews," not "Trendyol reviews."

**Licence**: the dataset card body states [CC-BY-SA-4.0](https://github.com/fthbrmnby/turkish-text-data/blob/master/LICENCE); note its own metadata header separately (and inconsistently) tags itself `license: unknown` — flagging both so nobody is surprised either way.

**Size / class distribution (raw pool)**: 235,165 reviews total — 220,284 positive / 14,881 negative (~15:1 imbalance). Already strictly binary at the source (`ClassLabel` with only `negative`/`positive`) — there is no neutral/3-star category to drop, unlike the rating-based English source below.

**Processing** (`src/get_data.py:prepare_turkish()`): drop blank/whitespace-only reviews (21 found in the raw pool), drop duplicate review text, then draw three disjoint, class-balanced samples with a fixed seed (42) — nothing else touches this pool. Frozen splits, tagged `real-data-v1`, read-only from here on (Section 2):

| Split | Size | Class balance |
|---|---|---|
| `real_train.tsv` | 2,000 | 1,000 / 1,000 (deliberately small — this is the low-resource setting the whole project tests) |
| `real_dev.tsv` | 1,000 | 500 / 500 |
| `real_test.tsv` | 2,000 | 1,000 / 1,000 |

Format: TSV, header row, columns `text`, `label` (`0`=negative, `1`=positive).

## English sentiment source — `data/mt/en_reviews_20k.tsv`

**Source**: [`fancyzhx/amazon_polarity`](https://huggingface.co/datasets/fancyzhx/amazon_polarity) on Hugging Face (Apache-2.0). Already binary at the source (1-2★→negative, 4-5★→positive, 3★ excluded by the dataset itself).

**Processing** (`src/get_data.py:prepare_english()`): drop duplicate review text, drop reviews over 50 tokens (long reviews translate badly with a small MT model), then sample 20,000 reviews class-balanced (10,000/10,000), seed 42. This is the pool that gets machine-translated into synthetic Turkish training data (Section 3.6).

## MT parallel corpus — `data/mt/opus_en_tr/`

**Source**: OPUS OpenSubtitles v2018 (en-tr), downloaded directly from OPUS's file server. See `src/prepare_mt_corpus.py` for the full filtering pipeline (empty lines, >60 tokens, length-ratio 0.5-2.0, duplicates, langid) and the repo-root `README.md`'s "MT pipeline" section for how to reproduce. 100k pairs total, split 96k train / 2k dev / 2k test, seed 42.

## Classifier tokenizer — `data/spm/`

`tr_sp8k.model`/`.vocab`: Turkish-only unigram SentencePiece, 8k vocab (16k wasn't reachable from only 2,000 training reviews), trained once on `real_train.tsv` only and frozen — shared by the LSTM and Transformer classifiers (BERT uses its own pretrained tokenizer instead). See `src/train_spm.py`.

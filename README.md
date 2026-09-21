# Backtranslation for Low-Resource Sentiment Classification (EN→TR)

Course project for "Introduction to Neural Networks and Sequence-to-Sequence 
Learning" (Heidelberg University, SoSe 2026).

## Team
- Ipek Uzun
- Buse Erkiraz
- Niyousha Mojoudi

## Goal
Test whether machine-translated (English→Turkish) labeled sentiment data 
improves Turkish sentiment classification, using JoeyNMT for translation 
and real Trendyol reviews as ground truth.

## Method
1. Translate labeled English sentiment reviews into Turkish using JoeyNMT
2. Train three classifier types (LSTM, Transformer, fine-tuned Turkish 
   BERT), each with and without the synthetic translated data
3. Evaluate all models on a real, human-labeled Turkish Trendyol review 
   test set
4. Ablations: clean vs. unclean synthetic data, effect of MT quality, 
   stability across seeds

## Setup

1. Clone this repository:
```bash
   git clone https://github.com/<your-username>/backtranslation-sentiment-tr.git
   cd backtranslation-sentiment-tr
```

2. Create and activate a dedicated conda environment:
```bash
   conda create -n joeynmt-project python=3.11
   conda activate joeynmt-project
```

3. Clone and install JoeyNMT (kept out of this repo's git history — see `.gitignore`):
```bash
   git clone https://github.com/joeynmt/joeynmt.git
   cd joeynmt
   pip install -e . --no-build-isolation
   cd ..
```

4. Install the exact dependency versions used by the team:
```bash
   pip install -r requirements.txt
```

5. Verify the install:
```bash
   cd joeynmt
   python -m unittest
   cd ..
```
   You should see `OK` at the end of the output (one skipped test is expected — that's fine).

**Note (Apple Silicon Macs, M1/M2/M3):** if you hit dependency errors, make sure you end up with `numpy<2`, `torch==2.1.2`, and `sentencepiece<0.2` — these exact versions are pinned in `requirements.txt` because newer versions of these three specifically break compatibility with JoeyNMT on this chip.

## MT pipeline (EN→TR)

The parallel corpus is [Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100) (en-tr pair, 1M/2K/2K train/dev/test), loaded directly via `dataset_type: "huggingface"` in `configs/en_tr_opus100.yaml` — no local copy of the corpus needed, `datasets` streams it from the HF cache.

1. Build the joint SentencePiece vocab (placeholder, trained on OPUS-100 itself — swap for Buse's shared review-domain tokenizer once it's ready):
```bash
   python src/build_mt_vocab.py
```
   This does **not** use `joeynmt/scripts/build_vocab.py --random-subset`: that script's subsampling path is broken for huggingface-backed datasets in the installed joeynmt==2.3.0 (calls a `sample_random_subset()` method that doesn't exist, and even without subsampling its `get_list()` does an O(n) `idx in self.indices` check per row — O(n^2) over 1M rows, effectively hangs). `src/build_mt_vocab.py` calls `sentencepiece` directly instead, replicating the same preprocessing (moses pretokenize + lowercase) and special-symbol IDs the config expects.

2. Train:
```bash
   cd joeynmt
   python -m joeynmt train ../configs/en_tr_opus100.yaml
```
   Training subsamples to 200k pairs (`sample_train_subset` in the config) for laptop-friendly runtime — this path is unaffected by the `build_vocab.py` bug above. `use_cuda: False` since JoeyNMT/torch 2.1.2 has no MPS support here, so this trains on CPU.
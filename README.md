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
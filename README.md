# Translation-Based Data Augmentation for Low-Resource Sentiment Classification (EN→TR)

Course project for "Introduction to Neural Networks and Sequence-to-Sequence 
Learning" (Heidelberg University, SoSe 2026).

## Team
- Ipek Uzun
- Buse Erkiraz
- Niyousha Mojoudi

## Goal
Test whether machine-translated (English→Turkish) labeled sentiment data 
improves Turkish sentiment classification, using JoeyNMT for translation 
and real Turkish product reviews as ground truth.

## Method
1. Translate labeled English sentiment reviews into Turkish using JoeyNMT
2. Train three classifier types (LSTM, Transformer, fine-tuned Turkish 
   BERT), each with and without the synthetic translated data
3. Evaluate all models on a real, human-labeled Turkish product review 
   test set
4. Ablations: clean vs. unclean synthetic data, effect of MT quality, 
   stability across seeds

## Setup

1. Clone this repository:
```bash
   git clone https://github.com/niushamjd/mt-augmentation-sentiment-tr.git
   cd mt-augmentation-sentiment-tr
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

## Hardware

What each of us is training on — matters for how ambitious MT/classifier training can be, and for explaining runtimes in the report.

| Person | Machine | GPU | Notes |
|---|---|---|---|
| Niyousha | Apple M1 Pro, 16GB RAM | none usable | JoeyNMT/torch==2.1.2 has no MPS (Apple GPU) support here — everything trains on CPU. |
| Ipek | *TODO* | *TODO* | |
| Buse | *TODO* | *TODO* | |

## MT pipeline (EN→TR)

Parallel corpus: OPUS OpenSubtitles v2018 (en-tr), downloaded directly from OPUS's own file server (not a HF mirror — `Helsinki-NLP/open_subtitles` on the HF Hub relies on a legacy loading script `datasets>=5.0` no longer supports). Chosen over SETIMES per the project spec's own domain-match reasoning: informal subtitle text is a closer register match for product reviews than SETIMES' news-domain text.

1. Download the raw corpus and prepare it (filters empty lines, >60-token pairs, length-ratio outliers, duplicates, and non-en/tr lines via langid; caps at 100k pairs; holds out 2k dev + 2k test):
```bash
   curl -o /tmp/opensubtitles_en-tr.zip "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-tr.txt.zip"
   unzip /tmp/opensubtitles_en-tr.zip -d /tmp/opensubtitles
   python src/prepare_mt_corpus.py \
     --en /tmp/opensubtitles/OpenSubtitles.en-tr.en \
     --tr /tmp/opensubtitles/OpenSubtitles.en-tr.tr \
     --max-raw-lines 1500000
```
   `--max-raw-lines` stream-subsamples before filtering, since the raw file is ~45M line pairs (~1.5GB) — far more than needed and too large to safely load whole into memory on a laptop. Writes `data/mt/opus_en_tr/{train,dev,test}.{en,tr}`.

2. Build the joint SentencePiece **BPE** vocab (8k merges, trained on the train split only, per the project spec):
```bash
   python src/build_mt_vocab.py
```
   This does **not** use `joeynmt/scripts/build_vocab.py`: that script's dataset-subsampling path is broken in the installed joeynmt==2.3.0 (calls a `sample_random_subset()` method that doesn't exist on `BaseDataset`, and its `get_list()` does an O(n) `idx in self.indices` check per row regardless — O(n²) over a large corpus, effectively hangs). `src/build_mt_vocab.py` calls `sentencepiece` directly instead.

3. Train:
```bash
   cd joeynmt
   python -m joeynmt train ../configs/en_tr_opus.yaml
```
   `use_cuda: False` since JoeyNMT/torch 2.1.2 has no MPS support here, so this trains on CPU. See the config's header comment for how to capture the `mt_early`/`mt_final` checkpoints the project needs for RQ4 — JoeyNMT only keeps checkpoints that beat the previous best, so this needs a deliberate manual step once training finishes.
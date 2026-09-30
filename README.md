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
| Niyousha | Apple M1 Pro, 16GB RAM | MPS (classifiers only) | JoeyNMT/torch==2.1.2 has no MPS (Apple GPU) support, so anything using JoeyNMT trains on CPU regardless of machine; BERT fine-tuning and the pretrained-MT translation (both plain `transformers`/PyTorch, not JoeyNMT) ran on MPS here. |
| Ipek | MacBook Pro, Intel | None| JoeyNMT MT  and LSTM classifier trained on CPU. |
| Buse | MacBook Air, Apple Silicon | MPS (classifier only) | MT training (`transformer_en_tr_v2.yaml`) and all translation ran on CPU, same JoeyNMT/MPS limitation as above. Transformer classifier trained on MPS. |

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

2. **Superseded, do not use:** `src/build_mt_vocab.py` was written for an earlier MT
   run and still does Moses pretokenisation + `.lower()` before training the
   SentencePiece vocab, to match that run's (also superseded) `lowercase: True`
   JoeyNMT config. If the *model* is later trained on raw cased text (as the
   current config does) while the *vocab* was built this way, every capital letter
   and apostrophe becomes `<unk>` -- this is exactly what happened in MT run 1 and
   is why its translations came out as mostly `⁇` characters. Use step 2 below
   instead, which trains the vocab on the same raw cased text the model actually
   sees.

2. Build the joint SentencePiece **BPE** vocab (8k merges, character_coverage 1.0,
   trained on the train split only, per the project spec) with the corrected script:
```bash
   python src/train_mt_spm.py
```
   This must end with `OK, wrote ...` -- it self-checks by encoding a cased test
   sentence and asserting zero `<unk>` pieces before declaring success. If it
   doesn't, stop and don't train; something about the training text changed.

3. Train (this is MT run 2, `transformer_en_tr_v2.yaml`; see the file's header
   comment for exactly how it differs from the superseded `transformer_en_tr.yaml`):
```bash
   python -m joeynmt train configs/transformer_en_tr_v2.yaml
```
   `use_cuda: True` falls back to CPU automatically when no GPU is found, so this
   trains on CPU on a laptop without MPS. JoeyNMT only saves a checkpoint when it
   beats the previous best (`keep_best_ckpts: -1` keeps every one of those), so
   after training finishes, check `models/transformer_en_tr_v2/validations.txt` for
   the step-by-step BLEU/chrF curve and copy out two checkpoints:
   - `mt_final`: the best checkpoint (highest dev BLEU/chrF).
   - `mt_early`: the *latest* checkpoint whose dev chrF is at least 6 points below
     `mt_final`'s, chosen *before* looking at any translations, so quality was not
     cherry-picked. For our actual run this was step 6000 vs step 20000 for
     `mt_final` -- don't assume those step numbers on a re-run with different data.
```bash
   cp models/transformer_en_tr_v2/<mt_final_step>.ckpt models/run2_mt_final.ckpt
   cp models/transformer_en_tr_v2/<mt_early_step>.ckpt models/run2_mt_early.ckpt
```

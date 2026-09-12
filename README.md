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

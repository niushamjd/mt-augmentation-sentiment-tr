import pandas as pd
import os

# Define the file paths for Seed 42 runs
PRED_DIR = "results/predictions"
LSTM_C1 = os.path.join(PRED_DIR, "lstm_C1_none_42_predictions.tsv")
LSTM_C3 = os.path.join(PRED_DIR, "lstm_C3_final_42_predictions.tsv")
TF_C1 = os.path.join(PRED_DIR, "transformer_C1_none_42_predictions.tsv")
BERT_C1 = os.path.join(PRED_DIR, "bert_C1_none_42_predictions.tsv")

def load_preds(path):
    if not os.path.exists(path):
        print(f"Warning: File not found {path}")
        return None
    return pd.read_csv(path, sep='\t')

def main():
    lstm_c1 = load_preds(LSTM_C1)
    lstm_c3 = load_preds(LSTM_C3)
    tf_c1 = load_preds(TF_C1)
    bert_c1 = load_preds(BERT_C1)

    os.makedirs("results/error_analysis", exist_ok=True)

    # 1. C1 vs C3 Comparison (using LSTM)
    if lstm_c1 is not None and lstm_c3 is not None:
        merged_lstm = lstm_c1.merge(
            lstm_c3, 
            on=['idx', 'text', 'gold'], 
            suffixes=('_c1', '_c3')
        )
        
        c1_wrong_c3_right = merged_lstm[(merged_lstm['correct_c1'] == 0) & (merged_lstm['correct_c3'] == 1)]
        c3_wrong_c1_right = merged_lstm[(merged_lstm['correct_c1'] == 1) & (merged_lstm['correct_c3'] == 0)]
        
        c1_wrong_c3_right.to_csv("results/error_analysis/fixed_by_synthetic_data.csv", index=False)
        c3_wrong_c1_right.to_csv("results/error_analysis/broken_by_synthetic_data.csv", index=False)
        
        print(f"LSTM: {len(c1_wrong_c3_right)} errors fixed by C3 synthetic data.")
        print(f"LSTM: {len(c3_wrong_c1_right)} correct examples broken by C3 synthetic data.")

    # 2. Cross-Model Agreement (All three fail)
    if lstm_c1 is not None and tf_c1 is not None and bert_c1 is not None:
        all_models = lstm_c1[['idx', 'text', 'gold', 'correct', 'pred']].rename(
            columns={'correct': 'lstm_correct', 'pred': 'lstm_pred'}
        )
        all_models = all_models.merge(
            tf_c1[['idx', 'correct', 'pred']].rename(columns={'correct': 'tf_correct', 'pred': 'tf_pred'}), 
            on='idx'
        )
        all_models = all_models.merge(
            bert_c1[['idx', 'correct', 'pred']].rename(columns={'correct': 'bert_correct', 'pred': 'bert_pred'}), 
            on='idx'
        )

        all_wrong = all_models[
            (all_models['lstm_correct'] == 0) & 
            (all_models['tf_correct'] == 0) & 
            (all_models['bert_correct'] == 0)
        ]
        
        all_wrong.to_csv("results/error_analysis/all_models_wrong.csv", index=False)
        print(f"\nAll Models Failed: {len(all_wrong)} common errors found in C1.")

        # 3. Trendyol-Specific Vocabulary Scan
        trendyol_keywords = ['kargo', 'kurye', 'satıcı', 'paket', 'teslimat', 'trendyol', 'orijinal', 'kutu', 'iade']
        
        def count_keywords(text):
            text_lower = str(text).lower()
            return sum(1 for kw in trendyol_keywords if kw in text_lower)

        all_wrong['keyword_count'] = all_wrong['text'].apply(count_keywords)
        trendyol_specific_errors = all_wrong[all_wrong['keyword_count'] > 0].sort_values(by='keyword_count', ascending=False)
        
        trendyol_specific_errors.to_csv("results/error_analysis/trendyol_specific_errors.csv", index=False)
        print(f"Trendyol-Specific Vocab: {len(trendyol_specific_errors)} of the common errors contain shipping/seller keywords.")

if __name__ == "__main__":
    main()
import os
import pandas as pd
import sentencepiece as spm

os.makedirs("data/spm", exist_ok=True)

df = pd.read_csv("data/sentiment/real_train.tsv", sep="\t")
df["text"].to_csv("data/spm/train_text.txt", index=False, header=False)

# train tokeniser
spm.SentencePieceTrainer.train(
    input="data/spm/train_text.txt",
    model_prefix="data/spm/tr_sp8k",
    vocab_size=8000,
    model_type="unigram",
    character_coverage=1.0,
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
)

sp = spm.SentencePieceProcessor(model_file="data/spm/tr_sp8k.model")
print(sp.get_piece_size())
print(sp.encode(df["text"][0], out_type=str))
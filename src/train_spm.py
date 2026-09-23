import os
import pandas as pd
import sentencepiece as spm

from text_utils import normalise_tr

os.makedirs("data/spm", exist_ok=True)

df = pd.read_csv("data/sentiment/real_train.tsv", sep="\t")

# normalise the text the same way the classifiers will (lowercase + fix the dotted i)
texts = [normalise_tr(t) for t in df["text"]]

# write one review per line
# before I used df["text"].to_csv(...) here, but pandas puts quotes around every
# review that has a comma in it (490 of 2000!), so the tokeniser learned fake " pieces.
# now I just write the lines myself
for t in texts:
    assert "\n" not in t and "\r" not in t, "review contains a line break"

with open("data/spm/train_text.txt", "w", encoding="utf-8") as f:
    for t in texts:
        f.write(t + "\n")

# check the file really has the same text as the tsv (no quotes added)
with open("data/spm/train_text.txt", encoding="utf-8") as f:
    lines = f.read().splitlines()
assert lines == texts, "train_text.txt is not the same as the normalised reviews"
print("train_text.txt ok:", len(lines), "lines")

# train tokeniser (same settings as before, only the input text changed)
spm.SentencePieceTrainer.train(
    input="data/spm/train_text.txt",
    model_prefix="data/spm/tr_sp8k",
    vocab_size=8000,
    model_type="unigram",
    character_coverage=1.0,
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
)

sp = spm.SentencePieceProcessor(model_file="data/spm/tr_sp8k.model")
print("vocab size:", sp.get_piece_size())
print("pad id:", sp.pad_id(), "unk id:", sp.unk_id())  # use sp.pad_id() in the models, not a hard-coded 0!
print(sp.encode(texts[0], out_type=str))

# check 1: no quote pieces made up by the old bug
quote_pieces = [sp.id_to_piece(i) for i in range(sp.get_piece_size()) if '"' in sp.id_to_piece(i)]
print("pieces containing a quote:", quote_pieces)

# check 2: <unk> rate on the dev set (the tokeniser never saw dev, so this is the honest test)
dev = pd.read_csv("data/sentiment/real_dev.tsv", sep="\t")
n_unk = 0
n_total = 0
for t in dev["text"]:
    ids = sp.encode(normalise_tr(t))
    n_total += len(ids)
    n_unk += ids.count(sp.unk_id())
unk_rate = n_unk / n_total
print(f"dev <unk> rate: {unk_rate:.5f} ({n_unk} of {n_total} pieces)")
assert unk_rate < 0.001, "too many <unk> on dev, something is wrong with the tokeniser"
print("tokeniser ok")

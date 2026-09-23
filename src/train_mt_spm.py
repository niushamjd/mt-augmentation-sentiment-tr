"""Train the joint EN-TR SentencePiece model for JoeyNMT on cased text.

The first MT model used a SentencePiece model trained on lowercased text, so
almost every capital letter and the apostrophe became <unk>. This script trains
a replacement on the cased training data and checks the result before anything
is trained with it.

Run from the repo root:  python src/train_mt_spm.py
"""
import sentencepiece as spm

DATA = "data/mt/opus_en_tr"
PREFIX = f"{DATA}/sp_cased"

spm.SentencePieceTrainer.train(
    input=f"{DATA}/train.en,{DATA}/train.tr",  # training split only, both languages
    model_prefix=PREFIX,
    vocab_size=8000,
    model_type="bpe",
    character_coverage=1.0,  # keep every character, incl. ç ğ ı İ ö ş ü and '
    input_sentence_size=2000000,
    shuffle_input_sentence=True,
    # same special ids as the JoeyNMT config: <unk>=0 <pad>=1 <s>=2 </s>=3
    unk_id=0, pad_id=1, bos_id=2, eos_id=3,
)

# JoeyNMT vocab file: one piece per line (the .vocab file also has a score column)
with open(f"{PREFIX}.vocab", encoding="utf-8") as f_in, \
        open(f"{PREFIX}.joey.vocab", "w", encoding="utf-8") as f_out:
    for line in f_in:
        f_out.write(line.split("\t")[0] + "\n")

# checks: no unknown pieces in a cased Turkish and English test sentence
sp = spm.SentencePieceProcessor(model_file=f"{PREFIX}.model")
test = "Bu Çok Güzel. İşte Tanrı'nın kitabı. This is GREAT, I love it!"
ids = sp.encode(test)
print(sp.encode(test, out_type=str))
n_unk = sum(1 for i in ids if i == sp.unk_id())
pieces = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
n_upper = sum(any(c.isupper() for c in p) for p in pieces)
print("unknown pieces in test sentence:", n_unk)
print("pieces containing uppercase:", n_upper)
assert n_unk == 0, "test sentence still has <unk> pieces"
assert n_upper > 100, "suspiciously few uppercase pieces"
print("OK, wrote", f"{PREFIX}.model", f"{PREFIX}.vocab", f"{PREFIX}.joey.vocab")

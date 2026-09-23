"""
Build the joint SentencePiece BPE vocab for the EN-TR MT model
(PROJECT_INSTRUCTIONS.md Section 3.2: "Joint SentencePiece BPE, 8k merges,
trained on the training portion only").

Sources sentences from data/mt/opus_en_tr/train.{en,tr} -- the already
filtered/capped/split OpenSubtitles corpus produced by
src/prepare_mt_corpus.py. Deliberately reads only the *train* split (never
dev/test), per the spec.

Calls sentencepiece directly (not joeynmt/scripts/build_vocab.py -- see
src/prepare_mt_corpus.py's docstring history / project memory for why that
script's subsampling path is unreliable for large corpora). Preprocesses
with moses pretokenize + lowercase first, matching the `tokenizer_cfg` the
MT JoeyNMT config applies at load time, so the vocab is fit to the same
text distribution the model will actually see.
"""
import argparse
from pathlib import Path

import sentencepiece as spm
from sacremoses import MosesTokenizer

VOCAB_SIZE = 8000
SPECIAL_SYMBOLS = {
    "unk_token": "<unk>", "unk_id": 0,
    "pad_token": "<pad>", "pad_id": 1,
    "bos_token": "<s>", "bos_id": 2,
    "eos_token": "</s>", "eos_id": 3,
}


def preprocessed_sents(path: Path, lang: str) -> list:
    tokenizer = MosesTokenizer(lang=lang)
    lines = path.read_text(encoding="utf-8").splitlines()
    return [tokenizer.tokenize(line, return_str=True).lower() for line in lines]


def main(train_dir: Path, out_dir: Path) -> None:
    print(f"Reading train split from {train_dir} ...")
    en_sents = preprocessed_sents(train_dir / "train.en", "en")
    tr_sents = preprocessed_sents(train_dir / "train.tr", "tr")
    assert len(en_sents) == len(tr_sents), (len(en_sents), len(tr_sents))
    pooled = en_sents + tr_sents
    print(f"Pooled {len(pooled)} sentences ({len(en_sents)} en + {len(tr_sents)} tr) for joint vocab training.")

    out_dir.mkdir(parents=True, exist_ok=True)
    input_file = out_dir / "_spm_train_input.txt"
    input_file.write_text("\n".join(pooled), encoding="utf-8")

    model_prefix = out_dir / "sp"
    print("Training SentencePiece BPE model...")
    spm.SentencePieceTrainer.Train(" ".join([
        f"--input={input_file}",
        f"--model_prefix={model_prefix}",
        "--model_type=bpe",
        f"--vocab_size={VOCAB_SIZE}",
        "--character_coverage=1.0",
        "--accept_language=en,tr",
        f"--unk_piece={SPECIAL_SYMBOLS['unk_token']}",
        f"--bos_piece={SPECIAL_SYMBOLS['bos_token']}",
        f"--eos_piece={SPECIAL_SYMBOLS['eos_token']}",
        f"--pad_piece={SPECIAL_SYMBOLS['pad_token']}",
        f"--unk_id={SPECIAL_SYMBOLS['unk_id']}",
        f"--bos_id={SPECIAL_SYMBOLS['bos_id']}",
        f"--eos_id={SPECIAL_SYMBOLS['eos_id']}",
        f"--pad_id={SPECIAL_SYMBOLS['pad_id']}",
        "--vocabulary_output_piece_score=false",
    ]))

    input_file.unlink()
    print(f"Wrote {model_prefix}.model and {model_prefix}.vocab")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-dir", type=str, default="data/mt/opus_en_tr")
    ap.add_argument("--out-dir", type=str, default="data/mt/opus_en_tr")
    args = ap.parse_args()
    main(Path(args.train_dir), Path(args.out_dir))

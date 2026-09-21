"""
Build the joint SentencePiece vocab for the EN-TR MT model, sourced from a
random subsample of Helsinki-NLP/opus-100 (en-tr).

Bypasses joeynmt's `scripts/build_vocab.py --random-subset`: for a
huggingface-backed dataset that script's subsampling path calls
`HuggingfaceTranslationDataset.sample_random_subset()`, which doesn't exist
in the installed joeynmt==2.3.0 (AttributeError), and even without
subsampling its `get_list()` filters via `idx in self.indices` -- an O(n)
membership check per example against what is effectively a list, so an O(n^2)
crawl over 1M rows that doesn't finish in a reasonable time. This script
calls sentencepiece directly instead, replicating what
`scripts/build_vocab.py`'s `train_spm()` would have produced (same special
symbols, same moses-pretokenize + lowercase preprocessing as
configs/en_tr_opus100.yaml), so the output is a drop-in sp.model/sp.vocab
for that config.

Placeholder tokenizer: trained on OPUS-100 itself, not the team's shared
review-domain corpus. Swap for Buse's tokenizer once it's ready.
"""
import argparse
from pathlib import Path

import sentencepiece as spm
from datasets import load_dataset
from sacremoses import MosesTokenizer

VOCAB_SIZE = 8000
SPECIAL_SYMBOLS = {
    "unk_token": "<unk>", "unk_id": 0,
    "pad_token": "<pad>", "pad_id": 1,
    "bos_token": "<s>", "bos_id": 2,
    "eos_token": "</s>", "eos_id": 3,
}


def preprocessed_sents(dataset, lang: str) -> list:
    tokenizer = MosesTokenizer(lang=lang)
    return [
        tokenizer.tokenize(ex["translation"][lang], return_str=True).lower()
        for ex in dataset
    ]


def main(n_pairs: int, seed: int, out_dir: Path) -> None:
    print(f"Loading Helsinki-NLP/opus-100 (en-tr) and sampling {n_pairs} pairs...")
    ds = load_dataset("Helsinki-NLP/opus-100", "en-tr", split="train")
    ds = ds.shuffle(seed=seed).select(range(n_pairs))

    print("Pre-processing (moses tokenize + lowercase) to match config...")
    en_sents = preprocessed_sents(ds, "en")
    tr_sents = preprocessed_sents(ds, "tr")
    pooled = en_sents + tr_sents
    print(f"Pooled {len(pooled)} sentences for joint vocab training.")

    out_dir.mkdir(parents=True, exist_ok=True)
    input_file = out_dir / "_spm_train_input.txt"
    input_file.write_text("\n".join(pooled), encoding="utf-8")

    model_prefix = out_dir / "sp"
    print("Training SentencePiece model...")
    spm.SentencePieceTrainer.Train(" ".join([
        f"--input={input_file}",
        f"--model_prefix={model_prefix}",
        "--model_type=unigram",
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
    ap.add_argument("--n-pairs", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="data/mt/opus100_en_tr")
    args = ap.parse_args()
    main(args.n_pairs, args.seed, Path(args.out_dir))

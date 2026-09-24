"""
Translate the 20k English reviews with a pretrained OPUS-MT model
(Helsinki-NLP/opus-mt-tc-big-en-tr), as a third MT-quality tier for RQ4 --
PROJECT_INSTRUCTIONS.md Section 6/8's optional extension "C": a much
stronger pretrained system alongside the team's own mt_early/mt_final,
approved as an optional extension only (never a replacement for the core
matrix). Lecturer approval for this scope addition confirmed 2026-09-24.

Decoding fixed at beam size 5 (matching Section 3.3's "beam size 5, length
penalty alpha 1.0 for every translation we ever produce" -- alpha isn't a
generate() parameter for MarianMT the same way JoeyNMT exposes it, so we
match beam width, which is the dominant cost/quality lever, and note this
in the report rather than silently deviating from the project's fixed
decoding without saying so).

This model is a genuinely strong translator (spot-checked: produces
correct, fluent output where the team's own model produces fluent but
unrelated Turkish -- see project memory for the comparison). Full 20k at
beam=5 takes ~4.3h on this machine's MPS device -- this script writes
translations incrementally (flushes after every batch) and can resume
from an interrupted run, since a 4+ hour unattended job is exactly the
kind of thing that gets interrupted by a laptop going to sleep.

Usage:
    python src/translate_pretrained.py
    # if interrupted, just re-run the same command -- it resumes
    # from data/mt/en_reviews_20k_pretrained.tr's existing line count.
"""
import argparse
import time
from pathlib import Path

import torch
from transformers import MarianMTModel, MarianTokenizer

MODEL_NAME = "Helsinki-NLP/opus-mt-tc-big-en-tr"
BATCH_SIZE = 64
NUM_BEAMS = 5
MAX_NEW_TOKENS = 128


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def already_done(out_path: Path) -> int:
    if not out_path.exists():
        return 0
    return len(out_path.read_text(encoding="utf-8").splitlines())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path("data/mt/en_reviews_20k.src"))
    ap.add_argument("--out", type=Path, default=Path("data/mt/en_reviews_20k_pretrained.tr"))
    ap.add_argument("--limit", type=int, default=None, help="only translate the first N lines (testing)")
    args = ap.parse_args()

    device = get_device()
    print(f"device: {device}")

    lines = args.src.read_text(encoding="utf-8").splitlines()
    if args.limit:
        lines = lines[:args.limit]
    n = len(lines)
    start = already_done(args.out)
    if start >= n:
        print(f"{args.out} already has {start} lines >= {n} source lines, nothing to do.")
        return
    if start > 0:
        print(f"resuming from line {start} ({args.out} already has {start} translations)")

    print(f"loading {MODEL_NAME} ...")
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    t0 = time.time()
    with args.out.open("a", encoding="utf-8") as out_f:
        for i in range(start, n, BATCH_SIZE):
            batch_lines = lines[i:i + BATCH_SIZE]
            enc = tokenizer(
                batch_lines, return_tensors="pt", padding=True,
                truncation=True, max_length=MAX_NEW_TOKENS,
            ).to(device)
            with torch.no_grad():
                out_ids = model.generate(
                    **enc, max_new_tokens=MAX_NEW_TOKENS, num_beams=NUM_BEAMS,
                )
            translations = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
            for t in translations:
                assert "\t" not in t and "\n" not in t, "translation contains a tab or line break"
                out_f.write(t + "\n")
            out_f.flush()

            done = i + len(batch_lines)
            elapsed = time.time() - t0
            rate = (done - start) / elapsed if elapsed > 0 else 0
            eta_min = (n - done) / rate / 60 if rate > 0 else float("inf")
            print(f"{done}/{n} ({rate:.2f} lines/s, ETA {eta_min:.1f} min)", flush=True)

    print(f"done. wrote {n - start} new translations to {args.out} ({n} total)")


if __name__ == "__main__":
    main()

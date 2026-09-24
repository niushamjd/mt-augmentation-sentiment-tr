"""
Filter machine-translated synthetic Turkish reviews
(PROJECT_INSTRUCTIONS.md Section 3.6).

An example is kept for synth_clean.tsv only if ALL of the following hold:
  1. Turkish output non-empty, at least 3 tokens.
  2. EN/TR token-length ratio between 0.5 and 2.0.
  3. No token repeated more than 4x consecutively; no trigram repeated
     more than twice (catches degenerate MT loops).
  4. langid says the output is Turkish.
  5. Less than 30% of output tokens also appear in the source (catches
     copy-through of untranslated English).
  6. Length-normalised log-probability above the 25th percentile of the
     batch's distribution -- only if --scores is given.

Thresholds are fixed here before looking at any downstream classification
result, per spec -- don't retune them based on what "looks better."

Reports, for each criterion independently, what fraction of examples it
alone would remove (the "why" table for the report), plus the final
N_clean after all criteria combined.

Usage:
    python src/filter_synth.py \
        --en data/mt/en_reviews_20k.src \
        --tr <translated Turkish output, one line per EN line> \
        --labels data/mt/en_reviews_20k.tsv \
        --mt-system final \
        [--scores <JoeyNMT `test --save-scores` .scores output file>] \
        [--out-dir data/sentiment]

NOTE on --scores (criterion 6): parsing this is best-effort and NOT yet
verified against a real scored run, because none exists in this repo yet
as of this writing. Reading JoeyNMT's source (joeynmt/prediction.py,
joeynmt/search.py): greedy decoding writes one line per sentence containing
a Python-list-literal of *per-token* scores; beam search (what this
project's decoding config actually uses, per Section 3.3's fixed beam
size 5) appears to write a single cumulative score per hypothesis instead,
and beam search already applies its own internal length penalty during
the search itself, which is a different formula from a plain
sum-of-log-probs / token-count average. This script handles both a bare
float and a list-literal per line, summing the list case, but the
resulting length normalization may not exactly match what "length-
normalised log-probability" means for a beam-search hypothesis whose
score was already penalty-adjusted mid-search. Before trusting criterion
6's results, sanity-check a handful of scores against the raw JoeyNMT
config/output by hand.
"""
import argparse
import ast
from collections import Counter
from pathlib import Path

import langid
import pandas as pd

langid.set_languages(["en", "tr"])  # ~60x faster than the unrestricted default

MIN_TOKENS = 3
MIN_RATIO, MAX_RATIO = 0.5, 2.0
MAX_CONSECUTIVE_REPEAT = 4
MAX_TRIGRAM_REPEAT = 2
MAX_COPY_THROUGH = 0.3
LOGPROB_PERCENTILE = 25


def read_lines(path: Path):
    return path.read_text(encoding="utf-8").splitlines()


def parse_score_line(line: str) -> float:
    """
    One JoeyNMT scores line is either a bare float (one cumulative score
    per hypothesis) or a Python-list-literal of per-token scores (greedy
    decoding's per-step breakdown) -- see the module docstring's NOTE on
    --scores. A list is summed to a single sequence log-prob.
    """
    line = line.strip()
    try:
        return float(line)
    except ValueError:
        pass
    parsed = ast.literal_eval(line)
    if isinstance(parsed, (list, tuple)):
        return float(sum(parsed))
    return float(parsed)


def check_length(tr_tokens) -> bool:
    return len(tr_tokens) >= MIN_TOKENS


def check_ratio(en_tokens, tr_tokens) -> bool:
    if not en_tokens or not tr_tokens:
        return False
    ratio = len(en_tokens) / len(tr_tokens)
    return MIN_RATIO <= ratio <= MAX_RATIO


def check_repetition(tr_tokens) -> bool:
    """True = OK (no degenerate repetition)."""
    run_len = 1
    for i in range(1, len(tr_tokens)):
        if tr_tokens[i] == tr_tokens[i - 1]:
            run_len += 1
            if run_len > MAX_CONSECUTIVE_REPEAT:
                return False
        else:
            run_len = 1
    if len(tr_tokens) >= 3:
        trigrams = Counter(tuple(tr_tokens[i:i + 3]) for i in range(len(tr_tokens) - 2))
        if any(c > MAX_TRIGRAM_REPEAT for c in trigrams.values()):
            return False
    return True


def check_langid(tr_text: str) -> bool:
    lang, _ = langid.classify(tr_text)
    return lang == "tr"


def check_copy_through(en_tokens, tr_tokens) -> bool:
    """True = OK (not mostly untranslated copy-through)."""
    if not tr_tokens:
        return False
    en_set = {t.lower() for t in en_tokens}
    n_copied = sum(1 for t in tr_tokens if t.lower() in en_set)
    return (n_copied / len(tr_tokens)) < MAX_COPY_THROUGH


def run_filters(en_lines, tr_lines, scores=None) -> dict:
    n = len(en_lines)
    en_toks = [line.split() for line in en_lines]
    tr_toks = [line.split() for line in tr_lines]

    results = {
        "length": [check_length(t) for t in tr_toks],
        "ratio": [check_ratio(e, t) for e, t in zip(en_toks, tr_toks)],
        "repetition": [check_repetition(t) for t in tr_toks],
        "langid": [check_langid(line) for line in tr_lines],
        "copy_through": [check_copy_through(e, t) for e, t in zip(en_toks, tr_toks)],
    }

    if scores is not None:
        norm_scores = [s / max(1, len(t)) for s, t in zip(scores, tr_toks)]
        sorted_scores = sorted(norm_scores)
        cut = sorted_scores[int(n * LOGPROB_PERCENTILE / 100)]
        results["logprob_25th_pct"] = [s >= cut for s in norm_scores]

    return results


def print_report(results: dict, n: int) -> None:
    print(f"filter report, N={n}:")
    for name, passed in results.items():
        n_fail = sum(1 for p in passed if not p)
        print(f"  {name}: {n_fail} removed alone ({n_fail / n * 100:.1f}%)")
    keep_mask = [all(results[c][i] for c in results) for i in range(n)]
    n_clean = sum(keep_mask)
    print(f"N_clean = {n_clean} ({n_clean / n * 100:.1f}% survive all criteria combined)")


def keep_mask(results: dict, n: int) -> list:
    return [all(results[c][i] for c in results) for i in range(n)]


def write_tsv(path: Path, texts, labels) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write("text\tlabel\n")
        for t, l in zip(texts, labels):
            assert "\t" not in t and "\n" not in t, "review contains a tab or line break"
            f.write(f"{t}\t{l}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--en", required=True, type=Path)
    ap.add_argument("--tr", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path, help="en_reviews_20k.tsv (id, text, label)")
    ap.add_argument("--mt-system", required=True, choices=["final", "early", "pretrained"])
    ap.add_argument("--scores", type=Path, default=None, help="JoeyNMT --save-scores output, one float/line")
    ap.add_argument("--out-dir", type=Path, default=Path("data/sentiment"))
    ap.add_argument("--no-write", action="store_true", help="report only, don't write synth_*.tsv")
    args = ap.parse_args()

    en_lines = read_lines(args.en)
    tr_lines = read_lines(args.tr)
    assert len(en_lines) == len(tr_lines), (len(en_lines), len(tr_lines))

    labels_df = pd.read_csv(args.labels, sep="\t")
    assert len(labels_df) == len(en_lines), (len(labels_df), len(en_lines))
    labels = labels_df["label"].tolist()

    scores = None
    if args.scores:
        scores = [parse_score_line(x) for x in read_lines(args.scores)]
        assert len(scores) == len(en_lines), (len(scores), len(en_lines))
    else:
        print("no --scores given: skipping criterion 6 (log-prob threshold)")

    n = len(en_lines)
    results = run_filters(en_lines, tr_lines, scores)
    print_report(results, n)

    if args.no_write:
        return

    suffix = "" if args.mt_system == "final" else f"_{args.mt_system}"
    write_tsv(args.out_dir / f"synth_all{suffix}.tsv", tr_lines, labels)
    print(f"wrote {args.out_dir / f'synth_all{suffix}.tsv'} ({n} rows, unfiltered)")

    mask = keep_mask(results, n)
    clean_texts = [t for t, m in zip(tr_lines, mask) if m]
    clean_labels = [l for l, m in zip(labels, mask) if m]
    write_tsv(args.out_dir / f"synth_clean{suffix}.tsv", clean_texts, clean_labels)
    print(f"wrote {args.out_dir / f'synth_clean{suffix}.tsv'} ({len(clean_texts)} rows, filtered)")


if __name__ == "__main__":
    main()

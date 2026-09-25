"""
Filter machine-translated synthetic Turkish reviews.

An example is kept for synth_clean.tsv only if all of the following hold:
  1. Turkish output non-empty, at least 3 tokens.
  2. EN/TR token-length ratio between 0.5 and 2.0. (The range is symmetric,
     so EN/TR and TR/EN give the same result.)
  3. No token repeated more than 4x consecutively; no trigram repeated
     more than twice (catches degenerate MT loops).
  4. langid says the output is Turkish. langid is restricted to en/tr for
     speed, so in practice this means "more Turkish than English": it catches
     untranslated English, not other kinds of garbage. State this in the report.
  5. Less than 30% of output tokens also appear in the source (catches
     copy-through of untranslated English). Raw lowercased tokens are
     compared, including numbers and punctuation, so number-heavy reviews
     ("75 lbs", "5-10%") can be flagged even when translated. Minor effect;
     one sentence in the report.
  6. Length-normalised log-probability above the 25th percentile of the
     batch's distribution -- only if --scores is given.

Thresholds are fixed here before looking at any downstream classification
result, per spec -- don't retune them based on what "looks better."

Before filtering, a leading "- " (subtitle dialogue style from the MT
training data) is stripped from every translation (STRIP_LEADING_DASH).
Filtering runs on the raw (cased) translation; everything written to
data/sentiment/ is passed through text_utils.normalise_tr, because the real
Turkish reviews are 100% lowercase and the classifier tokeniser has no
uppercase pieces.

Reports, for each criterion, how many examples it flags (regardless of the
other criteria) and how many it flags ONLY by itself, plus the final
N_clean after all criteria combined.

Usage:
    python src/filter_synth.py \
        --en data/mt/en_reviews_20k.src \
        --tr <translated Turkish output, one line per EN line> \
        --labels data/mt/en_reviews_20k.tsv \
        --mt-system final \
        [--scores <JoeyNMT `test --save-scores` .scores output file>] \
        [--out-dir data/sentiment]

Outputs (unless --no-write):
  mt-system final:  synth_all.tsv       C2  all non-empty translations
                    synth_clean.tsv     C3  translations passing all criteria
                    synth_matched.tsv   C2b random N_clean-sized sample of synth_all (seed 42)
                    synth_rejected.tsv  C3b random N_clean-sized sample of the rejected (seed 42)
  mt-system early:  synth_all_early.tsv     RQ4, unfiltered (no clean/matched/rejected)
Always written, next to the --tr file:
  <tr>.joined.tsv   id, English, raw and normalised Turkish, label, score and
                    one column per criterion (for error / sentiment-preservation analysis)
  <tr>.report.csv   the per-criterion table for the report

NOTE on --scores (criterion 6): checked against a real scored run and the
JoeyNMT 2.3.0 source. With beam search (Section 3.3: beam 5, alpha 1.0),
`test --save-scores` writes one value per line in brackets, e.g. "[-4.94]",
and that value is the sequence log-probability divided by the length penalty
((5 + length) / 6) ** alpha (joeynmt/search.py). So it already IS the
length-normalised log-probability and is used as it is; dividing it again by
the token count would count the length twice. The parser also accepts a bare
float, and sums a longer list (greedy decoding writes per-token scores).
"""
import argparse
import ast
import csv
import random
from collections import Counter
from pathlib import Path

import langid
import pandas as pd

from text_utils import normalise_tr

langid.set_languages(["en", "tr"])  # ~60x faster than the unrestricted default

MIN_TOKENS = 3
MIN_RATIO, MAX_RATIO = 0.5, 2.0
MAX_CONSECUTIVE_REPEAT = 4
MAX_TRIGRAM_REPEAT = 2
MAX_COPY_THROUGH = 0.3
LOGPROB_PERCENTILE = 25
STRIP_LEADING_DASH = True   # "- text" -> "text" before filtering
SUBSAMPLE_SEED = 42         # C2b / C3b subsamples, drawn once


def read_lines(path: Path):
    return path.read_text(encoding="utf-8").splitlines()


def parse_score_line(line: str) -> float:
    """
    One JoeyNMT scores line is either a bare float, a one-element list like
    "[-4.94]" (beam search, what this project uses), or a list of per-token
    scores (greedy decoding). A list is summed to a single value.
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


def clean_hyp(line: str) -> str:
    line = line.strip()
    if STRIP_LEADING_DASH and line.startswith("- "):
        line = line[2:].strip()
    return line


def is_empty(line: str) -> bool:
    # JoeyNMT writes "<unk>" for an empty hypothesis
    return line.strip() in ("", "<unk>")


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


def run_filters(en_lines, tr_lines, scores=None):
    n = len(en_lines)
    en_toks = [line.split() for line in en_lines]
    tr_toks = [line.split() for line in tr_lines]

    results = {
        "length": [check_length(t) and not is_empty(line) for t, line in zip(tr_toks, tr_lines)],
        "ratio": [check_ratio(e, t) for e, t in zip(en_toks, tr_toks)],
        "repetition": [check_repetition(t) for t in tr_toks],
        "langid": [check_langid(line) for line in tr_lines],
        "copy_through": [check_copy_through(e, t) for e, t in zip(en_toks, tr_toks)],
    }

    cut = None
    if scores is not None:
        # beam scores are already length-normalised (see module docstring)
        sorted_scores = sorted(scores)
        cut = sorted_scores[int(n * LOGPROB_PERCENTILE / 100)]
        results["logprob_25th_pct"] = [s >= cut for s in scores]

    return results, cut


def keep_mask(results: dict, n: int) -> list:
    return [all(results[c][i] for c in results) for i in range(n)]


def make_report(results: dict, n: int) -> pd.DataFrame:
    rows = []
    for name, passed in results.items():
        failed = [not p for p in passed]
        others = [c for c in results if c != name]
        only = [failed[i] and all(results[c][i] for c in others) for i in range(n)]
        rows.append({
            "criterion": name,
            "flagged": sum(failed),
            "pct_flagged": round(sum(failed) / n * 100, 1),
            "flagged_only_by_this": sum(only),
            "pct_only_by_this": round(sum(only) / n * 100, 1),
        })
    return pd.DataFrame(rows)


def print_report(report: pd.DataFrame, n: int, n_clean: int) -> None:
    print(f"filter report, N={n}:")
    print(report.to_string(index=False))
    print(f"N_clean = {n_clean} ({n_clean / n * 100:.1f}% survive all criteria combined)")


def write_tsv(path: Path, texts, labels) -> None:
    """Write text/label TSV with the same quoting convention as the real splits
    (pandas-style: a field containing a quote character is wrapped in quotes and
    inner quotes are doubled), so pd.read_csv(path, sep="\t") reads it back exactly.
    Writing raw lines breaks on translations that start with a quote: pandas then
    merges several lines into one field."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for t, l in zip(texts, labels):
        t = normalise_tr(t)  # same lowercase form as the real reviews
        assert "\t" not in t and "\n" not in t and "\r" not in t, "review contains a tab or line break"
        rows.append((t, l))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writerow(["text", "label"])
        w.writerows(rows)
    # read back and compare, so a quoting problem can never go unnoticed again
    back = pd.read_csv(path, sep="\t", keep_default_na=False)
    assert len(back) == len(rows), f"{path}: wrote {len(rows)} rows, read back {len(back)}"
    assert back["text"].tolist() == [t for t, _ in rows], f"{path}: text changed on read-back"


def label_counts(labels) -> dict:
    return dict(sorted(Counter(labels).items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--en", required=True, type=Path)
    ap.add_argument("--tr", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path, help="en_reviews_20k.tsv (id, text, label)")
    ap.add_argument("--mt-system", required=True, choices=["final", "early"])
    ap.add_argument("--scores", type=Path, default=None, help="JoeyNMT --save-scores output, one value/line")
    ap.add_argument("--out-dir", type=Path, default=Path("data/sentiment"))
    ap.add_argument("--no-write", action="store_true", help="report only, don't write synth_*.tsv")
    args = ap.parse_args()

    en_lines = read_lines(args.en)
    raw_tr_lines = read_lines(args.tr)
    assert len(en_lines) == len(raw_tr_lines), (len(en_lines), len(raw_tr_lines))

    labels_df = pd.read_csv(args.labels, sep="\t", keep_default_na=False)
    assert len(labels_df) == len(en_lines), (len(labels_df), len(en_lines))
    assert labels_df["text"].tolist() == en_lines, "--en and --labels are not in the same order"
    labels = labels_df["label"].tolist()
    ids = labels_df["id"].tolist() if "id" in labels_df.columns else list(range(len(labels_df)))

    tr_lines = [clean_hyp(x) for x in raw_tr_lines]
    n_dash = sum(1 for x in raw_tr_lines if x.strip().startswith("- "))

    scores = None
    if args.scores:
        scores = [parse_score_line(x) for x in read_lines(args.scores)]
        assert len(scores) == len(en_lines), (len(scores), len(en_lines))
    else:
        print("no --scores given: skipping criterion 6 (log-prob threshold)")

    n = len(en_lines)
    results, cut = run_filters(en_lines, tr_lines, scores)
    mask = keep_mask(results, n)
    n_clean = sum(mask)
    empty = [is_empty(t) for t in tr_lines]

    report = make_report(results, n)
    print(f"leading '- ' stripped: {n_dash}   empty translations (<unk>): {sum(empty)}")
    if cut is not None:
        print(f"criterion 6 cut-off (25th percentile of scores): {cut:.4f}")
    print_report(report, n, n_clean)
    print(f"labels, all: {label_counts(labels)}   "
          f"kept: {label_counts(l for l, m in zip(labels, mask) if m)}")

    # always: joined file and report next to the --tr file
    joined = pd.DataFrame({
        "id": ids,
        "text_en": en_lines,
        "text_tr_raw": raw_tr_lines,
        "text_tr": tr_lines,
        "text": [normalise_tr(t) for t in tr_lines],
        "label": labels,
        "score": scores if scores is not None else [None] * n,
        "empty": empty,
    })
    for name, passed in results.items():
        joined[f"ok_{name}"] = passed
    joined["kept"] = mask
    joined.to_csv(f"{args.tr}.joined.tsv", sep="\t", index=False)
    report.to_csv(f"{args.tr}.report.csv", index=False)
    print(f"wrote {args.tr}.joined.tsv and {args.tr}.report.csv")

    if args.no_write:
        return

    # C2: all translations except empty ones
    all_idx = [i for i in range(n) if not empty[i]]
    suffix = "" if args.mt_system == "final" else "_early"
    path = args.out_dir / f"synth_all{suffix}.tsv"
    write_tsv(path, [tr_lines[i] for i in all_idx], [labels[i] for i in all_idx])
    print(f"wrote {path} ({len(all_idx)} rows, unfiltered; {n - len(all_idx)} empty translations left out)")

    if args.mt_system == "early":
        return  # RQ4 uses unfiltered mt_early output only

    # C3: passing all criteria (empty ones always fail criterion 1)
    clean_idx = [i for i in range(n) if mask[i]]
    write_tsv(args.out_dir / "synth_clean.tsv", [tr_lines[i] for i in clean_idx], [labels[i] for i in clean_idx])
    print(f"wrote {args.out_dir / 'synth_clean.tsv'} ({len(clean_idx)} rows, filtered)")

    # C2b: random N_clean-sized sample of C2, drawn once with a fixed seed
    rng = random.Random(SUBSAMPLE_SEED)
    matched_idx = sorted(rng.sample(all_idx, len(clean_idx)))
    write_tsv(args.out_dir / "synth_matched.tsv", [tr_lines[i] for i in matched_idx], [labels[i] for i in matched_idx])
    print(f"wrote {args.out_dir / 'synth_matched.tsv'} ({len(matched_idx)} rows, "
          f"labels {label_counts(labels[i] for i in matched_idx)})")

    # C3b (optional): random N_clean-sized sample of the rejected, non-empty translations
    rejected_idx = [i for i in all_idx if not mask[i]]
    if len(rejected_idx) >= len(clean_idx):
        rej_sample = sorted(random.Random(SUBSAMPLE_SEED).sample(rejected_idx, len(clean_idx)))
        write_tsv(args.out_dir / "synth_rejected.tsv", [tr_lines[i] for i in rej_sample], [labels[i] for i in rej_sample])
        print(f"wrote {args.out_dir / 'synth_rejected.tsv'} ({len(rej_sample)} rows)")
    else:
        print(f"only {len(rejected_idx)} rejected translations, fewer than N_clean: synth_rejected.tsv not written")


if __name__ == "__main__":
    main()

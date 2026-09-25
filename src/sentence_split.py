"""Translate reviews sentence by sentence instead of as one long line.

Why: the MT model was trained on short subtitle lines. On whole reviews it
often translates only part of the text (61% of sample translations are less
than half as long as the source). Splitting a review into sentences, then
translating each sentence and joining the results, gives the model inputs
that look like its training data.

Step 1, split (before translation):
    python src/sentence_split.py split data/mt/sample2k.tsv data/mt/sample2k.sent
    -> data/mt/sample2k.sent.src   one sentence per line (JoeyNMT input)
    -> data/mt/sample2k.sent.map   review row index for every sentence line

Step 2, translate data/mt/sample2k.sent.src with JoeyNMT as usual.

Step 3, join (after translation):
    python src/sentence_split.py join data/mt/sample2k.sent data/mt/sample2k.sent.final
    reads  data/mt/sample2k.sent.final.test (+ .test.scores if present)
    -> data/mt/sample2k.sent.final.joined.test     one translated review per line
                                                   (a leading "- " is removed from every
                                                   sentence, same rule as filter_synth.py)
    -> data/mt/sample2k.sent.final.joined.scores   one score per review: mean of its
                                                   sentence scores, written as [x]
                                                   so filter_synth.py can read it
"""
import re
import sys
from pathlib import Path

import pandas as pd

# split after . ! ? (also repeated, e.g. "!!" or "...") when followed by a space,
# or directly followed by a capital letter ("belt.Because", a common typo in the data)
SPLIT = re.compile(r"(?<=[.!?])\s+|(?<=[a-z][.!?])(?=[A-Z])")


def split_review(text):
    parts = [p.strip() for p in SPLIT.split(str(text))]
    return [p for p in parts if p] or [str(text).strip()]


def do_split(tsv, prefix):
    df = pd.read_csv(tsv, sep="\t", keep_default_na=False)
    sents, owners = [], []
    for row, text in enumerate(df["text"]):
        for s in split_review(text):
            sents.append(s)
            owners.append(row)
    Path(f"{prefix}.src").write_text("\n".join(sents) + "\n", encoding="utf-8")
    Path(f"{prefix}.map").write_text("\n".join(map(str, owners)) + "\n", encoding="utf-8")
    n_per = pd.Series(owners).value_counts()
    print(f"{len(df)} reviews -> {len(sents)} sentences "
          f"(mean {n_per.mean():.1f}, max {n_per.max()} per review)")
    print(f"wrote {prefix}.src and {prefix}.map")


def do_join(prefix, out_prefix):
    owners = [int(x) for x in Path(f"{prefix}.map").read_text().splitlines()]
    hyps = Path(f"{out_prefix}.test").read_text(encoding="utf-8").splitlines()
    assert len(hyps) == len(owners), f"{len(hyps)} translations but {len(owners)} sentences"
    score_path = Path(f"{out_prefix}.test.scores")
    scores = None
    if score_path.exists():
        scores = [float(l.strip().strip("[]")) for l in score_path.read_text().splitlines()]
        assert len(scores) == len(owners)

    n_reviews = max(owners) + 1
    joined = [[] for _ in range(n_reviews)]
    joined_scores = [[] for _ in range(n_reviews)]
    n_dash = 0
    for i, (row, hyp) in enumerate(zip(owners, hyps)):
        hyp = hyp.strip()
        if hyp.startswith("- "):  # subtitle dialogue style, per sentence
            hyp = hyp[2:].strip()
            n_dash += 1
        if hyp and hyp != "<unk>":
            joined[row].append(hyp)
        if scores is not None:
            joined_scores[row].append(scores[i])

    lines = [" ".join(parts) if parts else "<unk>" for parts in joined]
    Path(f"{out_prefix}.joined.test").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_prefix}.joined.test ({len(lines)} reviews; leading '- ' removed from {n_dash} sentences)")
    if scores is not None:
        means = [sum(s) / len(s) for s in joined_scores]
        Path(f"{out_prefix}.joined.scores").write_text(
            "\n".join(f"[{m}]" for m in means) + "\n", encoding="utf-8")
        print(f"wrote {out_prefix}.joined.scores")


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in ("split", "join"):
        sys.exit(__doc__)
    if sys.argv[1] == "split":
        do_split(sys.argv[2], sys.argv[3])
    else:
        do_join(sys.argv[2], sys.argv[3])

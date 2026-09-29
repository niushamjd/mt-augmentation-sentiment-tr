"""
Build the nested RQ3 subsamples (PROJECT_INSTRUCTIONS.md Section 4, RQ3
extension): synthetic data at 1x and 5x the real training set size (2,000
and 10,000 examples). The 10x point is C2 itself (all 20,000, already
used) -- no new file needed for it.

Drawn once, seed 42, shared across all three classifiers (same principle
as synth_matched.tsv for C2b: "drawn once... not re-drawn per model or
per seed"), so the 1x set is a strict subset of the 5x set, which is a
strict subset of synth_all.tsv (C2).

Usage: python src/build_rq3_subsamples.py
Writes data/sentiment/synth_1x.tsv (2,000 rows) and synth_5x.tsv (10,000 rows).
"""
import random
from pathlib import Path

import pandas as pd

SEED = 42
SIZES = {"1x": 2000, "5x": 10000}
SRC = Path("data/sentiment/synth_all.tsv")
OUT_DIR = Path("data/sentiment")


def main() -> None:
    df = pd.read_csv(SRC, sep="\t", keep_default_na=False)
    n = len(df)
    rng = random.Random(SEED)
    order = list(range(n))
    rng.shuffle(order)

    for name, size in SIZES.items():
        assert size <= n, (name, size, n)
        idx = sorted(order[:size])  # nested: 1x's indices are a subset of 5x's
        subset = df.iloc[idx]
        out_path = OUT_DIR / f"synth_{name}.tsv"
        subset.to_csv(out_path, sep="\t", index=False)
        print(f"wrote {out_path}: {len(subset)} rows, labels {dict(subset['label'].value_counts())}")

    # sanity: nested containment holds
    one_x = set(pd.read_csv(OUT_DIR / "synth_1x.tsv", sep="\t", keep_default_na=False)["text"])
    five_x = set(pd.read_csv(OUT_DIR / "synth_5x.tsv", sep="\t", keep_default_na=False)["text"])
    assert one_x <= five_x, "1x is not a subset of 5x"
    print("sanity check passed: 1x subset-of 5x")


if __name__ == "__main__":
    main()

"""Sanity checks before the data freeze (tag data-v1). Run from the repo root:

    python src/check_freeze.py

Checks every file in data/sentiment/ that exists. It stops with an error on
anything that would make the experiment wrong, and prints warnings for
things a human should look at. Finally it prints 20 random synthetic
examples to read aloud together (Section 3.7).

The checks exist because of problems we actually had: a tokeniser that did
not match the text it was used on (MT run 1), and quotes silently added by
pandas (train_spm.py). Both would have been caught by an <unk> check.
"""
import random
from pathlib import Path

import pandas as pd
import sentencepiece as spm

from text_utils import normalise_tr

DATA = Path("data/sentiment")
SPM_MODEL = "data/spm/tr_sp8k.model"
REAL = {"real_train.tsv": 2000, "real_dev.tsv": 1000, "real_test.tsv": 2000}
SYNTH = ["synth_all.tsv", "synth_clean.tsv", "synth_matched.tsv",
         "synth_rejected.tsv", "synth_all_early.tsv"]
MAX_SYNTH_UNK_RATE = 0.005   # 0.5% of pieces; real dev is ~0

errors, warnings = [], []
sp = spm.SentencePieceProcessor(model_file=SPM_MODEL)


def load(name):
    path = DATA / name
    raw = path.read_text(encoding="utf-8")
    df = pd.read_csv(path, sep="\t", keep_default_na=False)
    if list(df.columns) != ["text", "label"]:
        errors.append(f"{name}: columns are {list(df.columns)}, expected ['text', 'label']")
    if not raw.endswith("\n"):
        warnings.append(f"{name}: no newline at end of file")
    return df


def basic_checks(name, df):
    if not set(df["label"].unique()) <= {0, 1}:
        errors.append(f"{name}: labels other than 0/1: {sorted(df['label'].unique())}")
    n_empty = (df["text"].astype(str).str.strip() == "").sum()
    if n_empty:
        errors.append(f"{name}: {n_empty} empty texts")
    n_break = df["text"].astype(str).str.contains("\t|\n|\r").sum()
    if n_break:
        errors.append(f"{name}: {n_break} texts contain a tab or line break")
    n_dup = df["text"].duplicated().sum()
    if n_dup:
        warnings.append(f"{name}: {n_dup} duplicate texts")
    counts = df["label"].value_counts().to_dict()
    share_pos = counts.get(1, 0) / len(df) if len(df) else 0
    print(f"  rows {len(df)}, labels {counts}")
    if not 0.4 <= share_pos <= 0.6:
        warnings.append(f"{name}: class balance {share_pos:.2f} positive")


def unk_rate(texts):
    n_unk = n_total = 0
    for t in texts:
        ids = sp.encode(normalise_tr(t))
        n_total += len(ids)
        n_unk += ids.count(sp.unk_id())
    return n_unk / max(n_total, 1)


# ---------- real data ----------
real = {}
for name, expected_rows in REAL.items():
    print(name)
    df = load(name)
    real[name] = df
    basic_checks(name, df)
    if len(df) != expected_rows:
        errors.append(f"{name}: {len(df)} rows, expected {expected_rows}")
    n_changed = sum(normalise_tr(t) != t for t in df["text"])
    print(f"  rows changed by normalise_tr: {n_changed} (fine: the classifiers normalise at load time)")
    print(f"  <unk> rate: {unk_rate(df['text']):.5f}")

norm_sets = {n: set(map(normalise_tr, df["text"])) for n, df in real.items()}
names = list(real)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        overlap = len(norm_sets[names[i]] & norm_sets[names[j]])
        if overlap:
            errors.append(f"{names[i]} and {names[j]} share {overlap} texts")

# ---------- synthetic data ----------
synth = {}
for name in SYNTH:
    if not (DATA / name).exists():
        print(f"{name}: not there (skipped)")
        continue
    print(name)
    df = load(name)
    synth[name] = df
    basic_checks(name, df)

    n_not_norm = sum(normalise_tr(t) != t for t in df["text"])
    if n_not_norm:
        errors.append(f"{name}: {n_not_norm} texts not normalised (capitals or dotted i left)")
    n_upper = sum(any(c.isupper() for c in t) for t in df["text"])
    print(f"  texts with uppercase: {n_upper}")

    n_unk_lit = df["text"].str.contains("<unk>", regex=False).sum()
    if n_unk_lit:
        msg = f"{name}: {n_unk_lit} texts contain a literal <unk> (empty or broken MT output)"
        (errors if name == "synth_clean.tsv" else warnings).append(msg)

    rate = unk_rate(df["text"])
    print(f"  <unk> rate: {rate:.5f}")
    if rate > MAX_SYNTH_UNK_RATE:
        errors.append(f"{name}: <unk> rate {rate:.4f} above {MAX_SYNTH_UNK_RATE}")

    s = set(map(normalise_tr, df["text"]))
    for real_name in ["real_dev.tsv", "real_test.tsv"]:
        overlap = len(s & norm_sets[real_name])
        if overlap:
            errors.append(f"{name}: {overlap} texts also in {real_name}")

if "synth_clean.tsv" in synth and "synth_matched.tsv" in synth:
    if len(synth["synth_clean.tsv"]) != len(synth["synth_matched.tsv"]):
        errors.append("synth_matched.tsv must have the same size as synth_clean.tsv")
if "synth_clean.tsv" in synth and "synth_all.tsv" in synth:
    extra = set(synth["synth_clean.tsv"]["text"]) - set(synth["synth_all.tsv"]["text"])
    if extra:
        errors.append(f"synth_clean.tsv has {len(extra)} texts that are not in synth_all.tsv")

# ---------- read-aloud sample ----------
if "synth_all.tsv" in synth:
    print("\n20 random synthetic examples (seed 7), read these aloud together:")
    sample = synth["synth_all.tsv"].sample(min(20, len(synth["synth_all.tsv"])), random_state=7)
    for _, row in sample.iterrows():
        print(f"  [{row['label']}] {row['text']}")

# ---------- summary ----------
print()
for w in warnings:
    print("WARNING:", w)
for e in errors:
    print("ERROR:  ", e)
if errors:
    raise SystemExit(f"{len(errors)} error(s): do NOT freeze the data yet")
print("all checks passed" + (f" ({len(warnings)} warning(s) to look at)" if warnings else ""))

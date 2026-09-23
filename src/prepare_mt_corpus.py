"""
Prepare the EN-TR MT parallel corpus (PROJECT_INSTRUCTIONS.md Section 3.2).

Source: OPUS OpenSubtitles v2018 (en-tr), downloaded directly from OPUS's
own file server (object.pouta.csc.fi), not a Hugging Face mirror --
Helsinki-NLP/open_subtitles on the HF Hub relies on a legacy loading script
that datasets>=5.0 no longer supports. OpenSubtitles is chosen over SETIMES
per the spec's own domain-match reasoning: informal, colloquial subtitle
text is a much closer register match for product reviews than SETIMES'
news-domain text, even though it's noisier -- hence the filtering below.

Pipeline: drop empty lines, drop pairs >60 tokens (either side), drop pairs
whose token-length ratio falls outside 0.5-2.0, drop duplicate pairs, then
(after subsampling for langid's sake, see --sample-before-langid) drop
pairs where langid doesn't confirm en/tr on each side respectively. Cap at
100k pairs total, then hold out 2k for MT dev + 2k for MT test out of that
100k, leaving the rest for MT train.
"""
import argparse
import random
import subprocess
from pathlib import Path

import langid

langid.set_languages(["en", "tr"])  # restrict candidates: ~60x faster than the full ~97-language default

MAX_TOKENS = 60
MIN_RATIO, MAX_RATIO = 0.5, 2.0
CAP = 100_000
N_DEV = 2_000
N_TEST = 2_000
SEED = 42


def count_lines(path: Path) -> int:
    return int(subprocess.check_output(["wc", "-l", str(path)]).split()[0])


def read_pairs(en_path: Path, tr_path: Path, max_raw_lines: int = None, seed: int = SEED):
    """
    Reads aligned line pairs. For a corpus far bigger than we need (e.g. the
    ~45M-line OpenSubtitles dump vs. our 100k cap), pass `max_raw_lines` to
    stream a random subsample by line index instead of materializing the
    whole file in memory -- this repo's OpenSubtitles files are ~1.5GB each,
    too much to safely double-load as Python string lists.
    """
    if max_raw_lines is None:
        en_lines = en_path.read_text(encoding="utf-8").splitlines()
        tr_lines = tr_path.read_text(encoding="utf-8").splitlines()
        assert len(en_lines) == len(tr_lines), (len(en_lines), len(tr_lines))
        return list(zip(en_lines, tr_lines))

    total = count_lines(en_path)
    assert total == count_lines(tr_path), "en/tr line counts differ"
    rnd = random.Random(seed)
    keep = set(rnd.sample(range(total), min(max_raw_lines, total)))
    pairs = []
    with en_path.open(encoding="utf-8") as ef, tr_path.open(encoding="utf-8") as tf:
        for i, (e, t) in enumerate(zip(ef, tf)):
            if i in keep:
                pairs.append((e.rstrip("\n"), t.rstrip("\n")))
    return pairs


def filter_pairs(pairs, sample_before_langid=500_000, seed=SEED):
    stats = {"input": len(pairs)}

    pairs = [(e, t) for e, t in pairs if e.strip() and t.strip()]
    stats["after_empty"] = len(pairs)

    kept = []
    for e, t in pairs:
        e_toks, t_toks = e.split(), t.split()
        if not e_toks or not t_toks:
            continue
        if len(e_toks) > MAX_TOKENS or len(t_toks) > MAX_TOKENS:
            continue
        ratio = len(e_toks) / len(t_toks)
        if not (MIN_RATIO <= ratio <= MAX_RATIO):
            continue
        kept.append((e, t))
    pairs = kept
    stats["after_length_ratio"] = len(pairs)

    seen = set()
    deduped = []
    for e, t in pairs:
        key = (e, t)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((e, t))
    pairs = deduped
    stats["after_dedup"] = len(pairs)

    if sample_before_langid and len(pairs) > sample_before_langid:
        rnd = random.Random(seed)
        pairs = rnd.sample(pairs, sample_before_langid)
    stats["before_langid"] = len(pairs)

    kept = []
    for e, t in pairs:
        if langid.classify(e)[0] != "en":
            continue
        if langid.classify(t)[0] != "tr":
            continue
        kept.append((e, t))
    pairs = kept
    stats["after_langid"] = len(pairs)

    return pairs, stats


def split_and_cap(pairs, seed=SEED, cap=CAP, n_dev=N_DEV, n_test=N_TEST):
    rnd = random.Random(seed)
    pairs = list(pairs)
    rnd.shuffle(pairs)
    pairs = pairs[:cap]
    test = pairs[:n_test]
    dev = pairs[n_test:n_test + n_dev]
    train = pairs[n_test + n_dev:]
    return train, dev, test


def write_moses(pairs, out_dir: Path, split: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{split}.en").write_text("\n".join(e for e, _ in pairs), encoding="utf-8")
    (out_dir / f"{split}.tr").write_text("\n".join(t for _, t in pairs), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--en", required=True, type=Path)
    ap.add_argument("--tr", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("data/mt/opus_en_tr"))
    ap.add_argument(
        "--sample-before-langid", type=int, default=500_000,
        help="subsample to this many pairs before the slow langid pass (0 to disable)",
    )
    ap.add_argument(
        "--max-raw-lines", type=int, default=None,
        help="stream-subsample the input to this many lines before filtering "
             "(memory-safe for corpora much bigger than the 100k cap, e.g. OpenSubtitles)",
    )
    args = ap.parse_args()

    pairs = read_pairs(args.en, args.tr, max_raw_lines=args.max_raw_lines)
    print(f"read {len(pairs)} raw pairs")
    filtered, stats = filter_pairs(
        pairs, sample_before_langid=args.sample_before_langid or None
    )
    print("filter stats:")
    for k, v in stats.items():
        removed = ""
        print(f"  {k}: {v}{removed}")

    if len(filtered) < CAP:
        print(f"WARNING: only {len(filtered)} pairs survived filtering, below the {CAP} cap.")

    train, dev, test = split_and_cap(filtered)
    print(f"train={len(train)} dev={len(dev)} test={len(test)}")

    write_moses(train, args.out_dir, "train")
    write_moses(dev, args.out_dir, "dev")
    write_moses(test, args.out_dir, "test")
    print(f"wrote to {args.out_dir}")


if __name__ == "__main__":
    main()

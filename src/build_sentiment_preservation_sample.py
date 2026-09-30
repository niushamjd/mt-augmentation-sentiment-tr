"""
Build the sentiment-preservation annotation sample (Section 5): did sentiment
survive MT, for the mt_final tier used in the core C2/C3 conditions.

Sampled from the *unfiltered* 20k translations (data/mt/full.sent.final.joined.test.joined.tsv),
not the C3-filtered subset, since the point is to characterise failure modes
(negation lost, intensifier dropped, mistranslated evaluative word, incoherent
output) -- the automatic C3 filters already remove some of the worst cases,
which would undercount exactly what this analysis is trying to measure.

Design: 90 examples, 30 assigned to each of the three annotators for
single annotation, plus a separate 30-example block that all three annotate
independently (identical copies, no cross-visibility) for inter-annotator
agreement. Total 120 distinct examples, drawn once with a fixed seed so the
sample is reproducible and non-overlapping between the four blocks.

Usage: python src/build_sentiment_preservation_sample.py
Writes data/annotation/{niyousha,ipek,buse}_unique.tsv (30 rows each) and
data/annotation/shared_iaa_{niyousha,ipek,buse}.tsv (30 identical rows each,
separate files so nobody sees a teammate's answers while annotating).
"""
import csv
import random
from pathlib import Path

SEED = 42
SRC = Path("data/mt/full.sent.final.joined.test.joined.tsv")
OUT_DIR = Path("data/annotation")
ANNOTATORS = ["niyousha", "ipek", "buse"]
N_UNIQUE_EACH = 30
N_SHARED = 30

ANNOTATION_COLUMNS = ["sentiment_preserved", "failure_category", "notes"]
# failure_category, when sentiment_preserved == "no", one of:
#   negation_lost | intensifier_dropped | word_mistranslated | incoherent | other


def main() -> None:
    with SRC.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    n = len(rows)
    need = N_UNIQUE_EACH * len(ANNOTATORS) + N_SHARED
    assert n >= need, (n, need)

    rng = random.Random(SEED)
    order = list(range(n))
    rng.shuffle(order)

    blocks = {}
    i = 0
    for name in ANNOTATORS:
        idx = order[i : i + N_UNIQUE_EACH]
        blocks[f"{name}_unique"] = idx
        i += N_UNIQUE_EACH
    shared_idx = order[i : i + N_SHARED]
    i += N_SHARED

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "text_en", "text_tr", "label"] + ANNOTATION_COLUMNS

    def write_block(out_name: str, indices: list) -> None:
        out_path = OUT_DIR / f"{out_name}.tsv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for idx in indices:
                r = rows[idx]
                writer.writerow(
                    {
                        "id": r["id"],
                        "text_en": r["text_en"],
                        "text_tr": r["text_tr"],
                        "label": r["label"],
                        "sentiment_preserved": "",
                        "failure_category": "",
                        "notes": "",
                    }
                )
        print(f"wrote {out_path}: {len(indices)} rows")

    for name in ANNOTATORS:
        write_block(f"{name}_unique", blocks[f"{name}_unique"])

    # identical shared block, one file per annotator so nobody sees the
    # others' judgments while annotating independently (blind IAA)
    for name in ANNOTATORS:
        write_block(f"shared_iaa_{name}", shared_idx)

    all_ids = set()
    for name in ANNOTATORS:
        all_ids.update(blocks[f"{name}_unique"])
    all_ids.update(shared_idx)
    assert len(all_ids) == need, "overlap detected between blocks"
    print(f"\nsanity check passed: {need} distinct examples, no overlap between blocks")
    print(f"each annotator's total workload: {N_UNIQUE_EACH} unique + {N_SHARED} shared = {N_UNIQUE_EACH + N_SHARED} rows")


if __name__ == "__main__":
    main()

"""
Inter-annotator agreement for the sentiment-preservation analysis (Section 5).

Reads the three shared_iaa_{niyousha,ipek,buse}.tsv files -- identical 30
examples, annotated independently and blindly by each of the three of you --
and reports agreement on `sentiment_preserved` (Fleiss' kappa across all
three, plus pairwise Cohen's kappa) and, as a secondary check, raw agreement
on `failure_category` restricted to rows all three marked as not-preserved.

If any annotator hasn't finished yet, prints how many rows are still blank
per file and stops there rather than computing a report on partial data.

Usage: python src/compute_iaa.py
Writes results/sentiment_iaa_report.txt
"""
import csv
from itertools import combinations
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

ANNOTATORS = ["niyousha", "ipek", "buse"]
IN_DIR = Path("data/annotation")
OUT_PATH = Path("results/sentiment_iaa_report.txt")


def load(name: str) -> dict:
    path = IN_DIR / f"shared_iaa_{name}.tsv"
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return {r["id"]: r for r in rows}


def fleiss_kappa(labels_by_item: list) -> float:
    """labels_by_item: list of lists, each the N raters' labels for one item."""
    n_items = len(labels_by_item)
    n_raters = len(labels_by_item[0])
    categories = sorted({lab for item in labels_by_item for lab in item})

    p_j = {c: 0 for c in categories}
    p_i_list = []
    for item in labels_by_item:
        counts = {c: item.count(c) for c in categories}
        for c in categories:
            p_j[c] += counts[c]
        p_i = (sum(v * v for v in counts.values()) - n_raters) / (n_raters * (n_raters - 1))
        p_i_list.append(p_i)

    p_bar = sum(p_i_list) / n_items
    for c in categories:
        p_j[c] /= (n_items * n_raters)
    p_e = sum(v * v for v in p_j.values())

    if p_e == 1.0:
        return 1.0  # every rater picked the same single category everywhere
    return (p_bar - p_e) / (1 - p_e)


def main() -> None:
    data = {name: load(name) for name in ANNOTATORS}

    ids_per_annotator = {name: set(d.keys()) for name, d in data.items()}
    ref_ids = ids_per_annotator[ANNOTATORS[0]]
    for name in ANNOTATORS[1:]:
        assert ids_per_annotator[name] == ref_ids, (
            f"{name}'s shared_iaa file covers different ids than {ANNOTATORS[0]}'s -- "
            "these must be the exact same 30 examples for IAA to be meaningful"
        )
    ids = sorted(ref_ids, key=int)

    incomplete = False
    for name in ANNOTATORS:
        blank = [i for i in ids if data[name][i]["sentiment_preserved"] == ""]
        if blank:
            incomplete = True
            print(f"{name}: {len(blank)}/{len(ids)} rows still blank -- ids {blank}")
    if incomplete:
        print("\nnot all three annotators have finished -- run this again once they have.")
        return

    sp = {name: [int(data[name][i]["sentiment_preserved"]) for i in ids] for name in ANNOTATORS}

    lines = []
    lines.append("=== sentiment_preserved agreement (n=30, 3 raters) ===\n")

    labels_by_item = [[sp[name][idx] for name in ANNOTATORS] for idx in range(len(ids))]
    kappa_all = fleiss_kappa(labels_by_item)
    lines.append(f"Fleiss' kappa (all three): {kappa_all:.3f}\n")

    all_agree = sum(1 for item in labels_by_item if len(set(item)) == 1)
    lines.append(f"unanimous agreement: {all_agree}/{len(ids)} ({100 * all_agree / len(ids):.1f}%)\n")

    lines.append("\npairwise:\n")
    for a, b in combinations(ANNOTATORS, 2):
        pct_agree = sum(1 for x, y in zip(sp[a], sp[b]) if x == y) / len(ids)
        kappa_pair = cohen_kappa_score(sp[a], sp[b])
        lines.append(f"  {a} vs {b}: {100 * pct_agree:.1f}% raw agreement, Cohen's kappa = {kappa_pair:.3f}\n")

    disagreement_ids = [i for idx, i in enumerate(ids) if len(set(labels_by_item[idx])) > 1]
    lines.append(f"\ndisagreement examples (ids, for illustrative examples in the report): {disagreement_ids}\n")

    lines.append("\n=== failure_category agreement (rows all three marked not-preserved) ===\n")
    unanimous_not_preserved = [
        i for idx, i in enumerate(ids)
        if all(sp[name][idx] == 0 for name in ANNOTATORS)
    ]
    if unanimous_not_preserved:
        cat_matches = 0
        for i in unanimous_not_preserved:
            cats = {data[name][i]["failure_category"] for name in ANNOTATORS}
            if len(cats) == 1:
                cat_matches += 1
        lines.append(
            f"of {len(unanimous_not_preserved)} rows all three flagged as not-preserved, "
            f"{cat_matches} have identical failure_category across all three "
            f"({100 * cat_matches / len(unanimous_not_preserved):.1f}%)\n"
        )
    else:
        lines.append("no rows where all three marked not-preserved\n")

    report = "".join(lines)
    print(report)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(report, encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

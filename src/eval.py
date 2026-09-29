"""
Shared evaluation script (PROJECT_INSTRUCTIONS.md Section 1.7/1.8).

One script, used by all three classifiers. Nobody writes their own metric
code. Call `evaluate_and_log()` from your training script after restoring
the best checkpoint and running it on the (frozen) test set, or run this
file directly on a pair of saved prediction/gold-label files.
"""
import argparse
import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

RESULTS_CSV_HEADER = [
    "run_id", "model", "condition", "synth_ratio", "mt_system", "seed",
    "n_real", "n_synth", "epochs_run", "best_epoch", "lr", "batch_size",
    "dev_macro_f1", "test_accuracy", "test_macro_f1", "test_f1_neg",
    "test_f1_pos", "train_time_s", "commit_hash", "timestamp", "notes",
]
LABELS = [0, 1]  # 0 = negative, 1 = positive


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    """Argmax predictions + gold labels (0/1) in, sklearn metrics out."""
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=LABELS, zero_division=0)
    precision_per_class = precision_score(y_true, y_pred, average=None, labels=LABELS, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, labels=LABELS, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0),
        "f1_neg": f1_per_class[0],
        "f1_pos": f1_per_class[1],
        "precision_neg": precision_per_class[0],
        "precision_pos": precision_per_class[1],
        "recall_neg": recall_per_class[0],
        "recall_pos": recall_per_class[1],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS),
    }


def print_report(metrics: dict, run_id: str) -> None:
    cm = metrics["confusion_matrix"]
    print(f"--- eval report: {run_id} ---")
    print(f"accuracy:      {metrics['accuracy']:.4f}")
    print(f"macro-F1:      {metrics['macro_f1']:.4f}")
    print(f"F1   (neg/pos): {metrics['f1_neg']:.4f} / {metrics['f1_pos']:.4f}")
    print(f"prec (neg/pos): {metrics['precision_neg']:.4f} / {metrics['precision_pos']:.4f}")
    print(f"rec  (neg/pos): {metrics['recall_neg']:.4f} / {metrics['recall_pos']:.4f}")
    print("confusion matrix (rows=gold, cols=pred, order=[neg,pos]):")
    print(cm)


def write_confusion_matrix(cm, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gold_pred", "neg", "pos"])
        writer.writerow(["neg", cm[0][0], cm[0][1]])
        writer.writerow(["pos", cm[1][0], cm[1][1]])


def write_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    out_path: Path,
    texts: Optional[Sequence[str]] = None,
) -> None:
    """
    One row per test example, in the same order the test set was passed in.
    `idx` is that position -- use it to line up predictions across models
    for the cross-model error analysis in the final phase.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        header = ["idx", "gold", "pred", "correct"]
        if texts is not None:
            header.insert(1, "text")
        writer.writerow(header)
        for i, (gold, pred) in enumerate(zip(y_true, y_pred)):
            row = [i, gold, pred, int(gold == pred)]
            if texts is not None:
                row.insert(1, texts[i])
            writer.writerow(row)


def get_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def append_results_row(row: dict, results_csv: Path) -> None:
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_csv.exists()
    with results_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_CSV_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def evaluate_and_log(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    run_id: str,
    model: str,
    condition: str,
    seed: int,
    n_real: int,
    results_csv: str,
    n_synth: int = 0,
    synth_ratio: float = 0.0,
    mt_system: str = "none",
    epochs_run: Optional[int] = None,
    best_epoch: Optional[int] = None,
    lr: Optional[float] = None,
    batch_size: Optional[int] = None,
    dev_macro_f1: Optional[float] = None,
    train_time_s: Optional[float] = None,
    texts: Optional[Sequence[str]] = None,
    notes: str = "",
) -> dict:
    """
    Score predictions against gold labels, print the report, write the
    confusion matrix + per-example predictions to results/predictions/, and
    append one row to `results_csv`. Call this once per run, on the frozen
    test set, after restoring the best checkpoint.

    `results_csv` has no default -- pass your own `results/results_{name}.csv`
    explicitly every time, so you never accidentally write into a
    teammate's file. run_id should be `{model}_{condition}_{mt_system}_{seed}`.
    """
    assert model in {"lstm", "transformer", "bert"}, model
    assert condition in {"C1", "C2", "C3", "C2b"}, condition
    assert set(y_true) <= set(LABELS) and set(y_pred) <= set(LABELS), "labels must be 0/1"
    if "," in notes:
        raise ValueError("notes must not contain commas (Section 1.8)")

    metrics = compute_metrics(y_true, y_pred)
    print_report(metrics, run_id)

    pred_dir = Path("results/predictions")
    write_confusion_matrix(metrics["confusion_matrix"], pred_dir / f"{run_id}_confusion_matrix.csv")
    write_predictions(y_true, y_pred, pred_dir / f"{run_id}_predictions.tsv", texts=texts)

    row = {
        "run_id": run_id,
        "model": model,
        "condition": condition,
        "synth_ratio": synth_ratio,
        "mt_system": mt_system,
        "seed": seed,
        "n_real": n_real,
        "n_synth": n_synth,
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "lr": lr,
        "batch_size": batch_size,
        "dev_macro_f1": dev_macro_f1,
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "test_f1_neg": metrics["f1_neg"],
        "test_f1_pos": metrics["f1_pos"],
        "train_time_s": train_time_s,
        "commit_hash": get_commit_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": notes,
    }
    append_results_row(row, Path(results_csv))
    return metrics


def _read_labels(path: Path) -> List[int]:
    return [int(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip() != ""]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", required=True, help="file with one predicted label (0/1) per line")
    ap.add_argument("--gold", required=True, help="file with one gold label (0/1) per line")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model", required=True, choices=["lstm", "transformer", "bert"])
    ap.add_argument("--condition", required=True, choices=["C1", "C2", "C3", "C2b"])
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--n-real", required=True, type=int)
    ap.add_argument("--results-csv", required=True, help="your own results/results_{name}.csv")
    ap.add_argument("--n-synth", type=int, default=0)
    ap.add_argument("--synth-ratio", type=float, default=0.0)
    ap.add_argument("--mt-system", default="none", choices=["none", "final", "early", "pretrained"])
    ap.add_argument("--epochs-run", type=int, default=None)
    ap.add_argument("--best-epoch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--dev-macro-f1", type=float, default=None)
    ap.add_argument("--train-time-s", type=float, default=None)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    y_pred = _read_labels(Path(args.preds))
    y_true = _read_labels(Path(args.gold))
    assert len(y_pred) == len(y_true), (len(y_pred), len(y_true))

    evaluate_and_log(
        y_true, y_pred,
        run_id=args.run_id, model=args.model, condition=args.condition,
        seed=args.seed, n_real=args.n_real, results_csv=args.results_csv,
        n_synth=args.n_synth, synth_ratio=args.synth_ratio, mt_system=args.mt_system,
        epochs_run=args.epochs_run, best_epoch=args.best_epoch, lr=args.lr,
        batch_size=args.batch_size, dev_macro_f1=args.dev_macro_f1,
        train_time_s=args.train_time_s, notes=args.notes,
    )


if __name__ == "__main__":
    main()

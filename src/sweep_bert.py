"""
Hyperparameter sweep for the BERT classifier, C1 only, seed 42 fixed
(PROJECT_INSTRUCTIONS.md Section 1.5: "If you tune, tune on C1 and then
use the same settings for C2, C2b and C3"). Compares candidates on dev
macro-F1 only -- never the test set (Section 1.2: "No hyperparameter
decision may be made on the test set").

Writes results/bert_hparam_sweep.csv for the report's "hyperparameters
we tried" table. This is separate from results/results_niyousha.csv,
which is reserved for official per-condition/per-seed runs.

Usage: python src/sweep_bert.py
"""
import csv
import subprocess
import sys
from pathlib import Path

GRID = [
    {"lr": 2e-5, "batch_size": 16},
    {"lr": 3e-5, "batch_size": 16},
    {"lr": 5e-5, "batch_size": 16},
    {"lr": 3e-5, "batch_size": 32},
]

OUT_CSV = Path("results/bert_hparam_sweep.csv")


def run_trial(lr: float, batch_size: int) -> dict:
    cmd = [
        sys.executable, "src/train_bert.py",
        "--condition", "C1", "--seed", "42",
        "--lr", str(lr), "--batch-size", str(batch_size),
        "--dev-only",
    ]
    print(f"--- running lr={lr} batch_size={batch_size} ---", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout[-1200:])
    if proc.returncode != 0:
        print(proc.stderr[-2000:])
        raise RuntimeError(f"trial failed: lr={lr} batch_size={batch_size}")
    for line in proc.stdout.splitlines():
        if line.startswith("DEV_RESULT"):
            return dict(kv.split("=") for kv in line.split()[1:])
    raise RuntimeError(f"no DEV_RESULT line in output for lr={lr} batch_size={batch_size}")


def main() -> None:
    rows = [run_trial(c["lr"], c["batch_size"]) for c in GRID]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== sweep summary (sorted by dev macro-F1) ===")
    for r in sorted(rows, key=lambda r: -float(r["best_dev_f1"])):
        print(r)

    best = max(rows, key=lambda r: float(r["best_dev_f1"]))
    print(f"\nbest: lr={best['lr']} batch_size={best['batch_size']} dev_macro_f1={best['best_dev_f1']}")
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()

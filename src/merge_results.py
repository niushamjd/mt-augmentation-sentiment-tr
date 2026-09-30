"""Merge results/results_{buse,ipek,niyousha}.csv into results/results.csv
and write the headline table (Section 5): mean +- std per model and condition.

    python src/merge_results.py

Stops with an error if a file has the wrong columns or if a run_id appears twice

Writes:
  results/results.csv          all rows, exact schema from Section 1.8
  results/summary_table.csv    core matrix: mean, std, n seeds per model x condition
  results/summary_table.md     the same, ready to paste into the report
"""
import sys
from pathlib import Path

import pandas as pd

HEADER = ("run_id,model,condition,synth_ratio,mt_system,seed,n_real,n_synth,epochs_run,"
          "best_epoch,lr,batch_size,dev_macro_f1,test_accuracy,test_macro_f1,test_f1_neg,"
          "test_f1_pos,train_time_s,commit_hash,timestamp,notes").split(",")
FILES = ["results/results_buse.csv", "results/results_ipek.csv", "results/results_niyousha.csv"]
MODELS = ["lstm", "transformer", "bert"]
CONDITIONS = ["C1", "C2", "C3", "C2b"]


def is_core(run_id):
    # core runs have exactly {model}_{condition}_{mt_system}_{seed}; RQ3 runs carry a suffix
    return run_id.count("_") == 3


def main():
    frames = []
    for f in FILES:
        if not Path(f).exists():
            print(f"missing: {f} (skipped)")
            continue
        df = pd.read_csv(f, keep_default_na=False)
        if list(df.columns) != HEADER:
            sys.exit(f"{f}: columns differ from the Section 1.8 schema:\n{list(df.columns)}")
        print(f"{f}: {len(df)} rows")
        frames.append(df)
    allres = pd.concat(frames, ignore_index=True)

    dup = allres[allres["run_id"].duplicated(keep=False)]
    if len(dup):
        sys.exit("duplicate run_id values (remove the old rows first):\n"
                 + dup[["run_id", "timestamp", "notes"]].to_string(index=False))

    allres.to_csv("results/results.csv", index=False)
    print(f"wrote results/results.csv ({len(allres)} rows)")

    core = allres[allres["run_id"].map(is_core)
                  & allres["mt_system"].isin(["none", "final"])
                  & allres["condition"].isin(CONDITIONS)]
    rows = []
    for m in MODELS:
        for c in CONDITIONS:
            v = core[(core["model"] == m) & (core["condition"] == c)]["test_macro_f1"].astype(float)
            rows.append({"model": m, "condition": c, "n_seeds": len(v),
                         "mean": round(v.mean(), 4) if len(v) else None,
                         "std": round(v.std(ddof=1), 4) if len(v) > 1 else None,
                         "seeds": " ".join(str(s) for s in core[(core["model"] == m)
                                                               & (core["condition"] == c)]["seed"])})
    table = pd.DataFrame(rows)
    table.to_csv("results/summary_table.csv", index=False)

    lines = ["| model | C1 real only | C2 + all synthetic | C3 + filtered | C2b size-matched |",
             "|---|---|---|---|---|"]
    for m in MODELS:
        cells = []
        for c in CONDITIONS:
            r = table[(table["model"] == m) & (table["condition"] == c)].iloc[0]
            if r["n_seeds"] == 0:
                cells.append("n/a")
            elif r["n_seeds"] == 1:
                cells.append(f"{r['mean']:.3f} (1 seed)")
            else:
                cells.append(f"{r['mean']:.3f} ± {r['std']:.3f}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    Path("results/summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    missing = table[table["n_seeds"] < 2]
    if len(missing):
        print("\nWARNING: fewer than 2 seeds for:\n" + missing[["model", "condition", "n_seeds"]].to_string(index=False))


if __name__ == "__main__":
    main()

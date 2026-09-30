"""The three figures for the report

    python src/merge_results.py     # first
    python src/make_figures.py

Reads results/results.csv, writes results/figures/*.png and *.pdf:
  fig_conditions.*  test macro-F1 by condition, grouped by model, mean +- std over seeds
  fig_rq3.*         test macro-F1 vs amount of synthetic data (seed 42)
  fig_rq4.*         test macro-F1 with data from mt_early / mt_final (/ pretrained), seed 42
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ["lstm", "transformer", "bert"]
MODEL_NAMES = {"lstm": "LSTM", "transformer": "Transformer", "bert": "BERT"}
CONDITIONS = ["C1", "C2", "C3", "C2b"]
COND_NAMES = {"C1": "real only", "C2": "+ all synthetic", "C3": "+ filtered", "C2b": "+ size-matched"}
GREYS = ["0.15", "0.45", "0.7", "0.9"]
HATCHES = ["", "//", "..", "xx"]
OUT = Path("results/figures")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=200)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"wrote {OUT / name}.png/.pdf")


def main():
    df = pd.read_csv("results/results.csv", keep_default_na=False)
    df["test_macro_f1"] = df["test_macro_f1"].astype(float)
    core = df[(df["run_id"].str.count("_") == 3) & df["mt_system"].isin(["none", "final"])]

    # (a) conditions by model
    fig, ax = plt.subplots(figsize=(7, 3.6))
    width = 0.2
    x = np.arange(len(MODELS))
    for i, c in enumerate(CONDITIONS):
        means, stds = [], []
        for m in MODELS:
            v = core[(core["model"] == m) & (core["condition"] == c)]["test_macro_f1"]
            means.append(v.mean() if len(v) else np.nan)
            stds.append(v.std(ddof=1) if len(v) > 1 else 0)
        ax.bar(x + (i - 1.5) * width, means, width, yerr=stds, capsize=3,
               color=GREYS[i], hatch=HATCHES[i], edgecolor="black", linewidth=0.6,
               label=COND_NAMES[c])
    ax.set_xticks(x, [MODEL_NAMES[m] for m in MODELS])
    ax.set_ylabel("test macro-F1")
    lo = core["test_macro_f1"].min()
    ax.set_ylim(max(0, lo - 0.05), min(1, core["test_macro_f1"].max() + 0.03))
    ax.legend(ncol=4, fontsize=8, loc="upper left", frameon=False)
    ax.set_title("Effect of synthetic data (mean ± std over seeds)", fontsize=10)
    save(fig, "fig_conditions")

    # (b) RQ3: 0x = C1, 1x/5x = RQ3 runs, 10x = C2 (all synthetic), seed 42
    fig, ax = plt.subplots(figsize=(5, 3.4))
    markers = {"lstm": "o", "transformer": "s", "bert": "^"}
    styles = {"lstm": "-", "transformer": "--", "bert": ":"}
    for m in MODELS:
        d = df[(df["model"] == m) & (df["seed"].astype(int) == 42)]
        pts = {}
        c1 = d[(d["condition"] == "C1") & (d["run_id"].str.count("_") == 3)]
        c2 = d[(d["condition"] == "C2") & (d["mt_system"] == "final")]
        if len(c1):
            pts[0] = c1["test_macro_f1"].iloc[0]
        for _, r in c2.iterrows():
            ratio = round(float(r["synth_ratio"]))
            if ratio in (1, 5) and r["run_id"].count("_") == 4:
                pts[ratio] = r["test_macro_f1"]
            if ratio == 10 and r["run_id"].count("_") == 3:
                pts[10] = r["test_macro_f1"]
        if pts:
            xs = sorted(pts)
            ax.plot(xs, [pts[k] for k in xs], color="black", linestyle=styles[m],
                    marker=markers[m], label=MODEL_NAMES[m])
    ax.set_xticks([0, 1, 5, 10], ["0x", "1x", "5x", "10x"])
    ax.set_xlabel("synthetic data (multiple of the 2,000 real reviews)")
    ax.set_ylabel("test macro-F1")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("How much synthetic data? (seed 42)", fontsize=10)
    save(fig, "fig_rq3")

    # (c) RQ4: MT quality, seed 42, all synthetic data from each MT system
    systems = [s for s in ["early", "final", "pretrained"]
               if len(df[(df["condition"] == "C2") & (df["mt_system"] == s)])]
    names = {"early": "mt_early", "final": "mt_final", "pretrained": "pretrained"}
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    width = 0.8 / len(systems)
    for i, s in enumerate(systems):
        vals = []
        for m in MODELS:
            v = df[(df["model"] == m) & (df["condition"] == "C2") & (df["mt_system"] == s)
                   & (df["seed"].astype(int) == 42) & (df["run_id"].str.count("_") == 3)]["test_macro_f1"]
            vals.append(v.iloc[0] if len(v) else np.nan)
        ax.bar(np.arange(len(MODELS)) + (i - (len(systems) - 1) / 2) * width, vals, width,
               color=GREYS[i], hatch=HATCHES[i], edgecolor="black", linewidth=0.6, label=names[s])
    ax.set_xticks(np.arange(len(MODELS)), [MODEL_NAMES[m] for m in MODELS])
    ax.set_ylabel("test macro-F1")
    sub = df[(df["condition"] == "C2") & (df["mt_system"].isin(systems))]["test_macro_f1"]
    ax.set_ylim(max(0, sub.min() - 0.05), min(1, sub.max() + 0.03))
    ax.legend(fontsize=8, frameon=False, ncol=len(systems), loc="upper left")
    ax.set_title("Effect of MT quality (seed 42)", fontsize=10)
    save(fig, "fig_rq4")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Redraw the two result figures from the CSV tables in results/ (no data or GPU needed).

  python scripts/make_figures.py --results results
Reads: summary_locked_test.csv, summary_cv_pooled.csv, per_class_f1_locked_test.csv
Writes: results/figures/macro_f1_test_vs_cv.png, results/figures/per_class_f1_test.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COL = {"CNN": "#4C78A8", "SVM": "#9A9A9A", "ResNet18": "#E07B39"}   # colour-blind-safe trio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    r = Path(ap.parse_args().results)
    fig_dir = r / "figures"; fig_dir.mkdir(exist_ok=True)
    test = pd.read_csv(r / "summary_locked_test.csv").set_index("model")
    cv = pd.read_csv(r / "summary_cv_pooled.csv").set_index("model")
    per = pd.read_csv(r / "per_class_f1_locked_test.csv").set_index("class")
    models = ["CNN", "SVM", "ResNet18"]

    # 1) macro-F1: CV vs locked test (with test CI)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x, w = np.arange(3), 0.36
    ax.bar(x - w / 2, [cv.loc[m, "macro_f1"] for m in models], w, label="5-fold CV (pooled)",
           color=[COL[m] for m in models], alpha=0.45, edgecolor="black", linewidth=0.6)
    t = [test.loc[m, "macro_f1"] for m in models]
    err = [[t[i] - test.loc[m, "macro_f1_ci_low"] for i, m in enumerate(models)],
           [test.loc[m, "macro_f1_ci_high"] - t[i] for i, m in enumerate(models)]]
    ax.bar(x + w / 2, t, w, yerr=err, capsize=4, label="Locked test (95% group-bootstrap CI)",
           color=[COL[m] for m in models], edgecolor="black", linewidth=0.6, hatch="//")
    for xi, m in zip(x, models):
        ax.text(xi - w / 2, cv.loc[m, "macro_f1"] + 0.006, f"{cv.loc[m,'macro_f1']:.3f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, test.loc[m, "macro_f1_ci_high"] + 0.006, f"{test.loc[m,'macro_f1']:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, models); ax.set_ylim(0.6, 1.0); ax.set_ylabel("Macro-F1")
    ax.set_title("ResNet18 leads in CV, but the locked test is a near tie with the CNN")
    ax.legend(loc="upper left", fontsize=8, frameon=False); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "macro_f1_test_vs_cv.png", dpi=200); plt.close(fig)

    # 2) per-class F1 on the locked test
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x, w = np.arange(len(per)), 0.26
    for i, m in enumerate(models):
        ax.bar(x + (i - 1) * w, per[m], w, label=m, color=COL[m], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x, [c.replace("_", " ") for c in per.index], rotation=15)
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("F1 (locked test)")
    ax.set_title("Per-class F1: PCO and Normal are easy, Teratoma and Serous are hard")
    ax.legend(frameon=False, fontsize=8, ncol=3); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "per_class_f1_test.png", dpi=200); plt.close(fig)
    print("wrote figures to", fig_dir)


if __name__ == "__main__":
    main()

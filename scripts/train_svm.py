#!/usr/bin/env python
"""Step 3 - SVM baseline on hand-crafted features.

  python scripts/train_svm.py --splits data/splits --out runs/svm            # features + CV grid + final fit
  python scripts/train_svm.py --splits data/splits --out runs/svm --predict-test

The reported setting is C=30, gamma=0.001 (RBF), standardised features, no class weights.
The grid below includes it; the grid is a plain 2-D search scored by pooled out-of-fold macro-F1
on the same group-aware folds used for the CNNs. The test set is touched only with --predict-test.
"""
import argparse
import itertools
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ovarian.config import SVM_CONFIG  # noqa: E402
from ovarian.data import load_split  # noqa: E402
from ovarian.evaluate import report_table  # noqa: E402
from ovarian.features import extract_many  # noqa: E402


def cached_features(df, path):
    if Path(path).exists():
        z = np.load(path, allow_pickle=True)
        if list(z["paths"]) == df["image_path"].tolist():
            return z["X"]
    X = extract_many(df["image_path"].tolist())
    np.savez_compressed(path, X=X, paths=np.array(df["image_path"].tolist(), dtype=object))
    return X


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--out", default="runs/svm")
    ap.add_argument("--predict-test", action="store_true")
    ap.add_argument("--C", type=float, nargs="*", default=[1, 3, 10, 30, 100])
    ap.add_argument("--gamma", type=float, nargs="*", default=[0.0001, 0.0003, 0.001, 0.003])
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev, test = load_split(a.splits)

    if a.predict_test:
        model = joblib.load(out / "svm_final.joblib")
        X = cached_features(test, out / "features_test.npz")
        pred = test[["image_path", "class_name", "label", "group"]].copy()
        pred["pred"] = model.predict(X)
        pred.to_csv(out / "test_predictions.csv", index=False)
        print(report_table(pred["label"].values, pred["pred"].values).round(3))
        return

    X = cached_features(dev, out / "features_dev.npz")
    y, folds = dev["label"].values, dev["fold"].values
    rows = []
    for C, g in itertools.product(a.C, a.gamma):
        oof = np.zeros_like(y)
        for f in sorted(set(folds)):
            tr, va = folds != f, folds == f
            m = make_pipeline(StandardScaler(), SVC(C=C, gamma=g, kernel="rbf")).fit(X[tr], y[tr])
            oof[va] = m.predict(X[va])
        rows.append(dict(C=C, gamma=g, oof_macro_f1=f1_score(y, oof, average="macro")))
        print(rows[-1])
    grid = pd.DataFrame(rows).sort_values("oof_macro_f1", ascending=False)
    grid.to_csv(out / "svm_cv_grid.csv", index=False)
    best = grid.iloc[0]
    print(f"grid best: C={best.C} gamma={best.gamma} (reported setting: {SVM_CONFIG})")

    # final model uses the REPORTED setting for reproducibility of the paper numbers
    final = make_pipeline(StandardScaler(), SVC(**SVM_CONFIG)).fit(X, y)
    joblib.dump(final, out / "svm_final.joblib")
    print("saved", out / "svm_final.joblib")


if __name__ == "__main__":
    main()

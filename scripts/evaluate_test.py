#!/usr/bin/env python
"""Step 5 - score every model ONCE on the locked test set and compare them.

  python scripts/evaluate_test.py \
      --pred CNN=runs/cnn/test_predictions.csv SVM=runs/svm/test_predictions.csv \
             ResNet18=runs/resnet18/test_predictions.csv --out results

Each prediction CSV needs: label, pred, group (written by train_cv.py / train_svm.py).
Outputs: per_class_precision_recall_f1_support.csv (long), model_summary_metrics.csv,
paired_comparisons.csv. Confidence intervals resample near-duplicate GROUPS, 2000 times.
"""
import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ovarian.evaluate import group_bootstrap, paired_bootstrap, report_table  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pred", nargs="+", required=True, help="NAME=path/to/test_predictions.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    preds = {kv.split("=", 1)[0]: pd.read_csv(kv.split("=", 1)[1]) for kv in a.pred}

    long_rows, summ = [], []
    for name, d in preds.items():
        rep = report_table(d["label"], d["pred"])
        for cls, r in rep.iterrows():
            long_rows.append(dict(model=name, class_or_average=cls, precision=r["precision"],
                                  recall=r["recall"], f1=r["f1-score"], support=int(r["support"])))
        b = group_bootstrap(d["label"], d["pred"], d["group"], n_boot=a.n_boot)
        summ.append(dict(model=name, n=len(d), macro_f1=b["macro_f1"],
                         macro_f1_ci_low=b["macro_f1_ci"][0], macro_f1_ci_high=b["macro_f1_ci"][1],
                         accuracy=b["accuracy"], accuracy_ci_low=b["accuracy_ci"][0],
                         accuracy_ci_high=b["accuracy_ci"][1], correct=int(round(b["accuracy"] * len(d)))))
    pd.DataFrame(long_rows).round(4).to_csv(out / "per_class_precision_recall_f1_support.csv", index=False)
    pd.DataFrame(summ).round(4).to_csv(out / "model_summary_metrics.csv", index=False)

    pair_rows = []
    for (na, da), (nb, db) in itertools.combinations(preds.items(), 2):
        if not (da["image_path"].values == db["image_path"].values).all():
            sys.exit(f"{na} and {nb} predictions are not on the same images/order")
        r = paired_bootstrap(da["label"], da["pred"], db["pred"], da["group"], n_boot=a.n_boot)
        pair_rows.append(dict(comparison=f"{na} - {nb}", macro_f1_diff=r["diff"],
                              ci_low=r["ci"][0], ci_high=r["ci"][1],
                              ci_excludes_zero=r["ci_excludes_zero"]))
    pd.DataFrame(pair_rows).round(4).to_csv(out / "paired_comparisons.csv", index=False)
    print(pd.DataFrame(summ).round(3).to_string(index=False))
    print(pd.DataFrame(pair_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

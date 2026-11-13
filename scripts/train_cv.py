#!/usr/bin/env python
"""Step 2/4 - 5-fold CV training for the CNN or ResNet18, then the one-shot test prediction.

  python scripts/train_cv.py --model cnn      --splits data/splits --out runs/cnn
  python scripts/train_cv.py --model resnet18 --splits data/splits --out runs/resnet18
  python scripts/train_cv.py --model resnet18 --splits data/splits --out runs/resnet18 --predict-test

Training is resumable at fold level (a fold with summary.json is skipped).
`--predict-test` averages the 5 fold models' softmax outputs on the LOCKED test set
and writes <out>/test_predictions.csv. Run it once, after all decisions are frozen.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ovarian.config import CLASS_NAMES, CNN_CONFIG, RESNET_CONFIG  # noqa: E402
from ovarian.data import load_split  # noqa: E402
from ovarian.evaluate import report_table  # noqa: E402
from ovarian.train import ensemble_predict, train_fold  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", choices=["cnn", "resnet18"], required=True)
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=None, help="override max epochs (smoke tests)")
    ap.add_argument("--config-json", help="JSON file overriding the default hyper-parameters")
    ap.add_argument("--predict-test", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    cfg = dict(CNN_CONFIG if a.model == "cnn" else RESNET_CONFIG)
    if a.config_json:
        cfg.update(json.loads(Path(a.config_json).read_text()))
    dev, test = load_split(a.splits)

    if a.predict_test:
        prob = ensemble_predict(a.model, cfg, test, a.out)
        pred = test[["image_path", "class_name", "label", "group"]].copy()
        pred["pred"] = prob.argmax(1)
        for i, c in enumerate(CLASS_NAMES):
            pred[f"prob_{c}"] = prob[:, i]
        pred.to_csv(Path(a.out) / "test_predictions.csv", index=False)
        print(report_table(pred["label"].values, pred["pred"].values).round(3))
        return

    scores = []
    for f in a.folds:
        s = train_fold(a.model, cfg, dev, f, a.out, seed=a.seed, max_epochs=a.epochs,
                       pretrained=True)
        scores.append(s["best_val_macro_f1"])
        print(f"fold {f}: best val macro-F1 {s['best_val_macro_f1']:.4f} @ epoch {s['best_epoch']}")
    print(f"CV macro-F1 (mean of best-epoch fold scores, slightly optimistic): "
          f"{np.mean(scores):.4f} +/- {np.std(scores):.4f}")


if __name__ == "__main__":
    main()

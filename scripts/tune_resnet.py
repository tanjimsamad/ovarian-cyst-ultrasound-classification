#!/usr/bin/env python
"""Optional - Optuna search for the ResNet18 recipe (resumable: study lives in SQLite).

  python scripts/tune_resnet.py --splits data/splits --out runs/resnet_optuna --trials 20

Each trial trains folds 0-2 and returns their mean best validation macro-F1 (folds 3-4 are not
used for tuning). A finished trial persists; an interrupted trial is lost and re-run.
The search ranges below are a reconstruction; the best trial that was actually reported is stored
in src/ovarian/config.py::RESNET_CONFIG (lr 2.87e-4, wd 6.2e-3, batch 32, moderate aug, LS 0.05).
"""
import argparse
import sys
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ovarian.config import RESNET_CONFIG  # noqa: E402
from ovarian.data import load_split  # noqa: E402
from ovarian.train import train_fold  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--out", default="runs/resnet_optuna")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--tune-folds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev, _ = load_split(a.splits)

    def objective(trial):
        cfg = dict(
            learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True),
            batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
            augmentation=trial.suggest_categorical("augmentation", ["none", "mild", "moderate", "strong"]),
            label_smoothing=trial.suggest_categorical("label_smoothing", [0.0, 0.02, 0.05, 0.1]),
        )
        scores = [train_fold("resnet18", cfg, dev, f, out / f"trial{trial.number}",
                             max_epochs=a.epochs)["best_val_macro_f1"] for f in a.tune_folds]
        return sum(scores) / len(scores)

    study = optuna.create_study(direction="maximize", study_name="resnet18",
                                storage=f"sqlite:///{out / 'optuna_study.db'}", load_if_exists=True)
    remaining = a.trials - len([t for t in study.trials if t.state.name == "COMPLETE"])
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    study.trials_dataframe().to_csv(out / "optuna_trials.csv", index=False)
    print("best value", study.best_value, "\nbest params", study.best_params,
          "\n(repo default for comparison)", RESNET_CONFIG)


if __name__ == "__main__":
    main()

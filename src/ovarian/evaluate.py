"""Metrics, group-bootstrap confidence intervals and paired model comparison."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score)

from .config import CLASS_NAMES


def report_table(y, p, class_names=CLASS_NAMES):
    """Per-class precision/recall/F1/support + macro/weighted + accuracy (= micro average)."""
    rep = classification_report(y, p, labels=list(range(len(class_names))),
                                target_names=class_names, output_dict=True, zero_division=0)
    df = pd.DataFrame(rep).T
    df.loc["accuracy", ["precision", "recall", "f1-score"]] = accuracy_score(y, p)
    df.loc["accuracy", "support"] = len(y)
    return df


def _group_confusions(y, p, groups, k):
    """One KxK confusion matrix per group -> array (n_groups, K, K). Lets the bootstrap
    resample whole groups by simply summing matrices (fast; no re-prediction)."""
    uniq, inv = np.unique(groups, return_inverse=True)
    cms = np.zeros((len(uniq), k, k), dtype=np.int64)
    np.add.at(cms, (inv, y, p), 1)
    return cms


def _macro_f1_from_cm(cm):
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return f1.mean()


def group_bootstrap(y, p, groups, n_boot=2000, seed=0, k=len(CLASS_NAMES)):
    """Percentile 95% CI of macro-F1 and accuracy, resampling groups (not images)."""
    cms = _group_confusions(np.asarray(y), np.asarray(p), np.asarray(groups), k)
    rng = np.random.default_rng(seed)
    n = len(cms)
    f1s, accs = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        cm = cms[rng.integers(0, n, n)].sum(0)
        f1s[b], accs[b] = _macro_f1_from_cm(cm), np.trace(cm) / max(cm.sum(), 1)
    ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return {"macro_f1": float(f1_score(y, p, average="macro")), "macro_f1_ci": ci(f1s),
            "accuracy": float(accuracy_score(y, p)), "accuracy_ci": ci(accs)}


def paired_bootstrap(y, p_a, p_b, groups, n_boot=2000, seed=0, k=len(CLASS_NAMES)):
    """Macro-F1(A) - Macro-F1(B) with a CI from the SAME resampled groups for both models."""
    y, p_a, p_b, groups = map(np.asarray, (y, p_a, p_b, groups))
    ca, cb = _group_confusions(y, p_a, groups, k), _group_confusions(y, p_b, groups, k)
    rng = np.random.default_rng(seed)
    n = len(ca)
    d = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        d[b] = _macro_f1_from_cm(ca[idx].sum(0)) - _macro_f1_from_cm(cb[idx].sum(0))
    point = f1_score(y, p_a, average="macro") - f1_score(y, p_b, average="macro")
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": float(point), "ci": (float(lo), float(hi)),
            "ci_excludes_zero": bool(lo > 0 or hi < 0)}


def confusion(y, p, k=len(CLASS_NAMES)):
    return confusion_matrix(y, p, labels=list(range(k)))

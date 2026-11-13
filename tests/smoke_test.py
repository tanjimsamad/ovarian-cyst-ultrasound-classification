"""End-to-end smoke test on SYNTHETIC images (no real data, CPU, ~1-2 min).

    python tests/smoke_test.py

Checks: grouping + split has no group leakage, SVM features, 1-epoch CNN/ResNet18 fold training,
fold-ensemble prediction, bootstrap CIs. It validates the code path, not model quality.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ovarian.config import CLASS_NAMES, CNN_CONFIG, RESNET_CONFIG  # noqa: E402
from ovarian.evaluate import group_bootstrap, paired_bootstrap, report_table  # noqa: E402
from ovarian.features import feature_dim, extract_many  # noqa: E402
from ovarian.splits import make_splits  # noqa: E402
from ovarian.train import ensemble_predict, train_fold  # noqa: E402


def synth(root, per_class=24, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for ci, c in enumerate(CLASS_NAMES):
        (root / c).mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            base = rng.integers(0, 255, (48, 64), dtype=np.uint8)
            base[:, : 8 + 10 * ci] = 255                                 # class-dependent stripe
            if i % 6 == 1:                                               # near-duplicate of previous
                base = prev
            prev = base
            p = root / c / f"{i}.png"
            Image.fromarray(base).resize((120, 90)).save(p)
            rows.append(dict(image_path=str(p), class_name=c, source="synthetic"))
    return pd.DataFrame(rows)


def main():
    tmp = Path(tempfile.mkdtemp())
    man = synth(tmp / "img")
    dev, test = make_splits(man, CLASS_NAMES, test_fraction=0.2, n_folds=3)
    assert not set(dev.group) & set(test.group)
    print("split ok:", len(dev), "dev /", len(test), "test; groups:", man.shape[0], "->", dev.group.nunique() + test.group.nunique())

    X = extract_many(dev.image_path.tolist()[:3])
    print("SVM feature dim:", X.shape[1], "== feature_dim()", feature_dim())

    for name, cfg in (("cnn", CNN_CONFIG), ("resnet18", RESNET_CONFIG)):
        run = tmp / name
        for f in (0, 1):
            train_fold(name, dict(cfg, batch_size=8), dev, f, run, max_epochs=1, pretrained=False)
        prob = ensemble_predict(name, dict(cfg, batch_size=8), test, run)
        assert prob.shape == (len(test), 5) and np.allclose(prob.sum(1), 1, atol=1e-4)
        print(name, "ensemble ok")
    y, p = test.label.values, prob.argmax(1)
    print(report_table(y, p).round(2).head(3))
    print(group_bootstrap(y, p, test.group.values, n_boot=200))
    print(paired_bootstrap(y, p, (p + 1) % 5, test.group.values, n_boot=200))
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()

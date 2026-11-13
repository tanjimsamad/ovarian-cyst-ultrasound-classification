#!/usr/bin/env python
"""Step 1 - build the group-aware locked test set and 5 CV folds.

Input (choose one):
  --manifest manifest.csv   columns: image_path, class_name [, source]
  --data-root DIR           layout DIR/<class_name>/*.{png,jpg,jpeg,bmp}  (source = "unknown")
Output: <out>/split_dev.csv (with `group`, `fold`), <out>/split_test.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ovarian.config import CLASS_NAMES  # noqa: E402
from ovarian.splits import make_splits  # noqa: E402

EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def scan(root):
    rows = [dict(image_path=str(p), class_name=p.parent.name, source="unknown")
            for c in CLASS_NAMES for p in sorted((Path(root) / c).glob("*")) if p.suffix.lower() in EXT]
    if not rows:
        sys.exit(f"no images found under {root}/<class_name>/ for classes {CLASS_NAMES}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--manifest")
    ap.add_argument("--data-root")
    ap.add_argument("--out", default="data/splits")
    ap.add_argument("--test-fraction", type=float, default=0.13)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if bool(a.manifest) == bool(a.data_root):
        ap.error("give exactly one of --manifest / --data-root")
    manifest = pd.read_csv(a.manifest) if a.manifest else scan(a.data_root)
    dev, test = make_splits(manifest, CLASS_NAMES, test_fraction=a.test_fraction, seed=a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev.to_csv(out / "split_dev.csv", index=False); test.to_csv(out / "split_test.csv", index=False)
    print(f"dev {len(dev)} images / {dev.group.nunique()} groups | test {len(test)} images / "
          f"{test.group.nunique()} groups\nclass counts (dev):\n{dev.class_name.value_counts()}"
          f"\nclass counts (test):\n{test.class_name.value_counts()}")


if __name__ == "__main__":
    main()

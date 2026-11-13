"""Leakage-aware data splitting.

The datasets contain many near-duplicate images (often the same patient/scan) and carry no
patient IDs. Random splits therefore leak. We approximate "same patient" by clustering
near-duplicates, then split by cluster:

1. dHash prefilter: images whose 64-bit difference hashes differ in <= 8 bits are candidates.
2. Confirm: z-scored 64x64 grayscale pixel correlation >= 0.90.
3. Union-find over confirmed pairs -> `group` id.
4. Locked test set: StratifiedGroupKFold, one fold held out. Groups with >= 10 images are
   forced into the development set (a single huge group must not dominate the test set).
5. Development set -> 5 StratifiedGroupKFold folds (`fold` column).

This reduces, but cannot rule out, leakage: true patient IDs do not exist.
"""
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold


def _gray(path, size):
    with Image.open(path) as im:
        return np.asarray(im.convert("L").resize(size, Image.Resampling.BILINEAR), dtype=np.float32)


def dhash(path):
    g = _gray(path, (9, 8))
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def small_vector(path, size=64):
    g = _gray(path, (size, size)).flatten()
    return (g - g.mean()) / (g.std() + 1e-6)


def _popcount(x):
    return bin(int(x)).count("1")


class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def assign_groups(paths, hash_dist=8, corr_thr=0.90):
    """Return an int group id per path (near-duplicate clustering)."""
    n = len(paths)
    hashes = [dhash(p) for p in paths]
    vecs = {}
    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if _popcount(hashes[i] ^ hashes[j]) <= hash_dist:
                for k in (i, j):
                    if k not in vecs:
                        vecs[k] = small_vector(paths[k])
                if float(np.dot(vecs[i], vecs[j]) / len(vecs[i])) >= corr_thr:
                    uf.union(i, j)
    roots = [uf.find(i) for i in range(n)]
    remap = {r: g for g, r in enumerate(dict.fromkeys(roots))}
    return np.array([remap[r] for r in roots])


def make_splits(manifest: pd.DataFrame, class_names, test_fraction=0.13, n_folds=5,
                big_group=10, seed=42):
    """manifest needs: image_path, class_name (and optionally source, filename).
    Returns (dev_df, test_df); dev_df has a `fold` column."""
    df = manifest.copy().reset_index(drop=True)
    df["label"] = df["class_name"].map({c: i for i, c in enumerate(class_names)})
    df["group"] = assign_groups(df["image_path"].tolist())

    sizes = df.groupby("group")["group"].transform("size")
    forced_dev = sizes >= big_group
    pool = df[~forced_dev]

    k = max(2, round(1 / test_fraction))
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    _, test_idx = next(sgkf.split(pool, pool["label"], pool["group"]))
    test_rows = pool.index[test_idx]
    test_df = df.loc[test_rows].reset_index(drop=True)
    dev_df = df.drop(index=test_rows).reset_index(drop=True)

    dev_df["fold"] = -1
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for f, (_, val_idx) in enumerate(cv.split(dev_df, dev_df["label"], dev_df["group"])):
        dev_df.loc[val_idx, "fold"] = f

    assert not set(dev_df["group"]) & set(test_df["group"]), "group leaked into test"
    return dev_df, test_df

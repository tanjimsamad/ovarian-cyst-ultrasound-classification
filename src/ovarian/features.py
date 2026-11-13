"""Hand-crafted texture/shape features for the SVM baseline.

Input: the same letterboxed 224x224 image as the CNNs, converted to 8-bit grayscale.
Feature groups: HOG (shape/edges), uniform LBP histogram (micro-texture), GLCM statistics
(second-order texture), grey-level histogram (intensity).

NOTE: this is a clean re-implementation. The original run used a 1352-dimensional vector whose
extractor file (`svm_feature_extractor.py`) stayed in the author's local output folder; this
version prints its own dimension (see `feature_dim()`), so re-run the SVM before comparing numbers.
"""
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern

from .data import Letterbox

_GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")


def load_gray(path, size=224):
    with Image.open(path) as im:
        rgb = Letterbox(size)(im.convert("RGB"))
    return np.asarray(rgb.convert("L"), dtype=np.uint8)


def extract_features(gray: np.ndarray) -> np.ndarray:
    # 1) HOG: 9 orientations, 32x32 cells, 2x2 blocks -> 36 blocks x 36 = 1296
    f_hog = hog(gray, orientations=9, pixels_per_cell=(32, 32), cells_per_block=(2, 2),
                block_norm="L2-Hys", feature_vector=True)
    # 2) uniform LBP (P=8, R=1) -> 10-bin histogram
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    f_lbp = np.bincount(lbp.astype(int).ravel(), minlength=10)[:10].astype(np.float32)
    f_lbp /= f_lbp.sum() + 1e-9
    # 3) GLCM (32 grey levels, 2 distances x 4 angles, averaged over angles) -> 6 props x 2
    q = (gray // 8).astype(np.uint8)
    glcm = graycomatrix(q, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=32, symmetric=True, normed=True)
    f_glcm = np.concatenate([graycoprops(glcm, p).mean(axis=1) for p in _GLCM_PROPS])
    # 4) intensity histogram, 32 bins
    f_hist = np.histogram(gray, bins=32, range=(0, 256))[0].astype(np.float32)
    f_hist /= f_hist.sum() + 1e-9
    return np.concatenate([f_hog, f_lbp, f_glcm, f_hist]).astype(np.float32)


def feature_dim():
    return extract_features(np.zeros((224, 224), np.uint8)).shape[0]


def extract_many(paths):
    return np.stack([extract_features(load_gray(p)) for p in paths])

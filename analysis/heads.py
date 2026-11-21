"""
analysis/heads.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

LOGREG_C        = 1.0
LOGREG_MAX_ITER = 1000
SEED            = 0


CAST_CHUNK_ROWS = 100_000


def to_float64(X: np.ndarray, idx: Optional[np.ndarray] = None) -> np.ndarray:
    n = int(X.shape[0]) if idx is None else int(len(idx))
    out = np.empty((n, X.shape[1]), dtype=np.float64)
    for i in range(0, n, CAST_CHUNK_ROWS):
        j = min(i + CAST_CHUNK_ROWS, n)
        out[i:j] = X[i:j] if idx is None else X[idx[i:j]]
    return out


def fit_standardizer(Xd: np.ndarray):
    sc = StandardScaler().fit(Xd)
    return sc, sc.transform(Xd, copy=False)


def scaled_block(sc: StandardScaler, X: np.ndarray,
                 idx: Optional[np.ndarray] = None) -> np.ndarray:
    return sc.transform(to_float64(X, idx), copy=False)


def split_masks(manifest: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = manifest["split"].astype(str).str.lower() if "split" in manifest.columns \
        else pd.Series(["train"] * len(manifest))
    val = s.isin(["val", "valid", "validation"])
    test = s.isin(["test"])
    train = ~(val | test)
    return train.values, val.values, test.values


def fit_head(head_type: str, X: np.ndarray, y: np.ndarray, standardized=None):
    if standardized is None:
        scaler, Xs = fit_standardizer(to_float64(X))
    else:
        scaler, Xs = standardized
        if Xs.shape != X.shape:
            raise ValueError(f"[heads] standardized block {Xs.shape} does not "
                             f"match X {X.shape}; it must come from this X.")
    if head_type == "linear":
        model = LogisticRegression(C=LOGREG_C, max_iter=LOGREG_MAX_ITER,
                                   random_state=SEED, class_weight="balanced")
    else:
        raise ValueError(f"[heads] unknown head_type '{head_type}'")
    model.fit(Xs, y)
    return model, scaler


def score_head(model, scaler, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(scaled_block(scaler, X))[:, 1]


def threshold_at_sensitivity(y_true: np.ndarray, y_score: np.ndarray,
                             target: float = 0.80) -> float:
    y_true = np.asarray(y_true, float); y_score = np.asarray(y_score, float)
    pos = y_score[y_true == 1.0]
    if pos.size == 0:
        return float(np.median(y_score)) if y_score.size else 0.5
    return float(np.quantile(pos, max(0.0, 1.0 - target)))

"""
analysis/heads.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

LOGREG_C        = 1.0
LOGREG_MAX_ITER = 1000
MLP_HIDDEN      = (256,)
MLP_ALPHA       = 1e-4
MLP_MAX_ITER    = 200
SEED            = 0

MAX_TRAIN: Optional[int] = None
MAX_TEST:  Optional[int] = None

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
    elif head_type == "mlp":
        model = MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, alpha=MLP_ALPHA,
                              max_iter=MLP_MAX_ITER, early_stopping=True,
                              random_state=SEED)
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


def threshold_youden(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_curve
    y_true = np.asarray(y_true, float); y_score = np.asarray(y_score, float)
    if len(np.unique(y_true)) < 2:
        return float(np.median(y_score)) if y_score.size else 0.5
    fpr, tpr, thr = roc_curve(y_true, y_score)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def run_head_per_finding(
    X: np.ndarray,
    manifest: pd.DataFrame,
    findings: List[str],
    head_type: str = "linear",
    target_sensitivity: float = 0.80,
    min_train: int = 50,
    min_test: int = 20,
    max_train: Optional[int] = None,
    max_test: Optional[int] = None,
) -> Dict[str, dict]:
    max_train = MAX_TRAIN if max_train is None else max_train
    max_test = MAX_TEST if max_test is None else max_test
    train_m, val_m, test_m = split_masks(manifest)
    finite = np.isfinite(X).all(axis=1)
    rng = np.random.RandomState(SEED)
    out: Dict[str, dict] = {}

    pbar = tqdm(findings, desc="[heads] fitting", unit="finding", leave=False)
    for f in pbar:
        if f not in manifest.columns:
            continue
        col = pd.to_numeric(manifest[f], errors="coerce").values
        labeled = np.isfinite(col)
        valid = labeled & finite

        tr = np.where(train_m & valid)[0]
        va = np.where(val_m & valid)[0]
        te = np.where(test_m & valid)[0]
        if va.size == 0 and tr.size > 0:
            n_hold = max(1, int(0.1 * tr.size))
            tr_shuffled = rng.permutation(tr)
            va = np.sort(tr_shuffled[:n_hold])
            tr = np.sort(tr_shuffled[n_hold:])

        if tr.size < min_train or te.size < min_test:
            continue
        if len(np.unique(col[tr])) < 2 or len(np.unique(col[te])) < 2:
            continue

        if max_train and tr.size > max_train:
            tr = np.sort(rng.choice(tr, max_train, replace=False))
        if max_test and te.size > max_test:
            te = np.sort(rng.choice(te, max_test, replace=False))
        pbar.set_postfix(finding=f, train=len(tr), test=len(te))

        model, scaler = fit_head(head_type, X[tr], col[tr])
        if va.size > 0 and len(np.unique(col[va])) >= 1:
            va_scores = score_head(model, scaler, X[va])
            thr_sens = threshold_at_sensitivity(col[va], va_scores, target_sensitivity)
            thr_youd = threshold_youden(col[va], va_scores)
        else:
            te_scores_tmp = score_head(model, scaler, X[te])
            thr_sens = threshold_at_sensitivity(col[te], te_scores_tmp, target_sensitivity)
            thr_youd = threshold_youden(col[te], te_scores_tmp)

        te_scores = score_head(model, scaler, X[te])
        out[f] = {
            "y_true": col[te].astype(float),
            "y_score": te_scores.astype(float),
            "threshold": float(thr_sens),
            "threshold_youden": float(thr_youd),
            "test_idx": te,
        }
    return out

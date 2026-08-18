"""
mitigation/preprocessing.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from analysis.heads import fit_standardizer, scaled_block, to_float64


def reweigh_weights(y: np.ndarray, g: np.ndarray) -> np.ndarray:
    y = np.asarray(y); g = np.asarray([str(v) for v in g])
    n = len(y)
    w = np.ones(n, dtype=float)
    for gv in np.unique(g):
        for yv in np.unique(y):
            cell = (g == gv) & (y == yv)
            n_cell = cell.sum()
            if n_cell == 0:
                continue
            p_g = (g == gv).mean()
            p_y = (y == yv).mean()
            w[cell] = (p_g * p_y) / (n_cell / n)
    return w


def _fit_logistic_block(Xd, y_tr, sample_weight, seed):
    sc, Xs = fit_standardizer(Xd)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    clf.fit(Xs, y_tr, sample_weight=sample_weight)
    return sc, clf


def reweighted(X_tr, y_tr, g_tr, X_va, y_va, g_va, X_te, g_te, seed=0):
    w = reweigh_weights(y_tr, g_tr)
    sc, clf = _fit_logistic_block(to_float64(X_tr), y_tr, w, seed)
    va = clf.predict_proba(scaled_block(sc, X_va))[:, 1]
    te = clf.predict_proba(scaled_block(sc, X_te))[:, 1]
    return va, te


def resampled(X_tr, y_tr, g_tr, X_va, y_va, g_va, X_te, g_te, seed=0):
    y = np.asarray(y_tr); g = np.asarray([str(v) for v in g_tr])
    rng = np.random.RandomState(seed)
    cells = [np.where((g == gv) & (y == yv))[0]
             for gv in np.unique(g) for yv in np.unique(y)]
    cells = [c for c in cells if len(c) > 0]
    target = max(len(c) for c in cells)
    idx = np.concatenate([rng.choice(c, target, replace=True) for c in cells])
    sc, clf = _fit_logistic_block(to_float64(X_tr, idx), y[idx], None, seed)
    del idx
    va = clf.predict_proba(scaled_block(sc, X_va))[:, 1]
    te = clf.predict_proba(scaled_block(sc, X_te))[:, 1]
    return va, te

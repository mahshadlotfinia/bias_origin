"""
mitigation/postprocessing.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


def _groups(g):
    return np.asarray([str(v) for v in g])


def per_group_platt(base_va, y_va, g_va, base_te, g_te, target_sens=0.80, seed=0):
    g_va = _groups(g_va); g_te = _groups(g_te)
    y_va = np.asarray(y_va, float)
    va = np.array(base_va, float).copy()
    te = np.array(base_te, float).copy()
    for lv in np.unique(g_va):
        m = g_va == lv
        if m.sum() < 10 or len(np.unique(y_va[m])) < 2:
            continue
        cal = LogisticRegression(C=1e6, max_iter=1000, random_state=seed)
        cal.fit(base_va[m].reshape(-1, 1), y_va[m])
        va[m] = cal.predict_proba(base_va[m].reshape(-1, 1))[:, 1]
        mt = g_te == lv
        if mt.any():
            te[mt] = cal.predict_proba(np.asarray(base_te)[mt].reshape(-1, 1))[:, 1]
    return va, te


def per_group_threshold_shift(base_va, y_va, g_va, base_te, g_te,
                              target_sens=0.80, seed=0):
    g_va = _groups(g_va); g_te = _groups(g_te)
    y_va = np.asarray(y_va, float)
    base_va = np.asarray(base_va, float); base_te = np.asarray(base_te, float)
    va = base_va.copy(); te = base_te.copy()
    for lv in np.unique(g_va):
        m = g_va == lv
        pos = base_va[m][y_va[m] == 1.0]
        thr = float(np.quantile(pos, max(0.0, 1.0 - target_sens))) if pos.size else 0.5
        va[m] = base_va[m] - thr + 0.5
        mt = g_te == lv
        if mt.any():
            te[mt] = base_te[mt] - thr + 0.5
    return np.clip(va, 0, 1), np.clip(te, 0, 1)

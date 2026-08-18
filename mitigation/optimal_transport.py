"""
mitigation/optimal_transport.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import numpy as np


def _shrunk_cov(X: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    cov = np.cov(X, rowvar=False)
    cov = np.atleast_2d(cov)
    d = cov.shape[0]
    mu = np.trace(cov) / max(d, 1)
    return (1 - alpha) * cov + alpha * mu * np.eye(d)


def _sym_eig(M):
    M = (M + M.T) / 2.0
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 1e-8, None)
    return w, V


def _sqrtm(M):
    w, V = _sym_eig(M)
    return (V * np.sqrt(w)) @ V.T


def _invsqrtm(M):
    w, V = _sym_eig(M)
    return (V * (1.0 / np.sqrt(w))) @ V.T


def _gaussian_ot_map(mu_s, Sig_s, mu_t, Sig_t):
    Ss = _sqrtm(Sig_s)
    Ss_inv = _invsqrtm(Sig_s)
    middle = _sqrtm(Ss @ Sig_t @ Ss)
    A = Ss_inv @ middle @ Ss_inv
    return mu_t, A, mu_s


def fairclip_ot_with_groups(X_tr, g_tr, X_te, g_te, seed=0, shrinkage=0.1):
    g_tr = np.asarray([str(v) for v in g_tr])
    Xtr = X_tr.astype(np.float64)
    mu_g = Xtr.mean(0)
    Sig_g = _shrunk_cov(Xtr, shrinkage)
    maps = {}
    for lv in np.unique(g_tr):
        m = g_tr == lv
        if m.sum() < 5:
            continue
        maps[lv] = _gaussian_ot_map(Xtr[m].mean(0), _shrunk_cov(Xtr[m], shrinkage), mu_g, Sig_g)

    def _apply(X, g):
        X = np.asarray(X, np.float64); g = np.asarray([str(v) for v in g])
        out = X.copy()
        for lv, (mu_t, A, mu_s) in maps.items():
            m = g == lv
            if m.any():
                out[m] = (X[m] - mu_s) @ A.T + mu_t
        return out.astype(np.float32)

    return _apply(Xtr, g_tr), _apply(X_te, g_te)

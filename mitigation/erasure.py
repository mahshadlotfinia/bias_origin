"""
mitigation/erasure.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from analysis.heads import fit_standardizer, scaled_block, to_float64


def _onehot(g):
    g = np.asarray(g, dtype=object).astype(str)
    levels = np.unique(g)
    ids = np.searchsorted(levels, g)
    Z = np.zeros((len(g), len(levels)), dtype=np.float32)
    Z[np.arange(len(g)), ids] = 1.0
    return Z, ids


def leace(X_tr, g_tr, X_te, seed=0, X_more=None):
    import torch
    from concept_erasure import LeaceEraser
    Z, _ = _onehot(g_tr)
    Xtr = torch.tensor(X_tr, dtype=torch.float32)
    Ztr = torch.tensor(Z, dtype=torch.float32)
    eraser = LeaceEraser.fit(Xtr, Ztr)
    Xtr_e = eraser(Xtr).numpy().astype(np.float32)
    del Xtr, Ztr
    Xte_e = eraser(torch.tensor(X_te, dtype=torch.float32)).numpy().astype(np.float32)
    if X_more is None:
        return Xtr_e, Xte_e
    Xmore_e = eraser(torch.tensor(X_more, dtype=torch.float32)).numpy().astype(np.float32)
    return Xtr_e, Xte_e, Xmore_e


def inlp(X_tr, g_tr, X_te, seed=0, n_iter=10, tol_acc=None, X_more=None):
    _, gid = _onehot(g_tr)
    n_groups = len(np.unique(gid))
    chance = 1.0 / n_groups if n_groups > 0 else 0.5
    tol_acc = tol_acc if tol_acc is not None else chance + 0.02
    d = X_tr.shape[1]
    P = np.eye(d, dtype=np.float64)
    Xtr = to_float64(X_tr)
    for _ in range(n_iter):
        if len(np.unique(gid)) < 2:
            break
        clf = LogisticRegression(C=1.0, max_iter=500, random_state=seed)
        XP = Xtr @ P
        clf.fit(XP, gid)
        acc = clf.score(XP, gid)
        del XP
        W = clf.coef_
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        basis = Vt[S > 1e-10]
        if basis.shape[0] == 0:
            break
        P = P @ (np.eye(d) - basis.T @ basis)
        if acc <= tol_acc:
            break
    Xtr_e = (Xtr @ P).astype(np.float32)
    del Xtr
    Xte_e = (to_float64(X_te) @ P).astype(np.float32)
    if X_more is None:
        return Xtr_e, Xte_e
    Xmore_e = (to_float64(X_more) @ P).astype(np.float32)
    return Xtr_e, Xte_e, Xmore_e


def rlace(X_tr, g_tr, X_te, seed=0, rank=1, n_iter=4000, X_more=None):
    try:
        from rlace import solve_adv_game
    except (ImportError, ModuleNotFoundError) as e:
        raise RuntimeError("[rlace] requires the rlace package "
                           "(pip install git+https://github.com/shauli-ravfogel/rlace.git). "
                           f"Import error: {e}") from e
    import torch
    _, gid = _onehot(g_tr)
    out = solve_adv_game(torch.tensor(X_tr, dtype=torch.float32),
                         torch.tensor(gid), rank=rank, device="cpu",
                         out_iters=n_iter)
    P = out["P"].numpy().astype(np.float64)
    Xtr_e = (to_float64(X_tr) @ P).astype(np.float32)
    Xte_e = (to_float64(X_te) @ P).astype(np.float32)
    if X_more is None:
        return Xtr_e, Xte_e
    return Xtr_e, Xte_e, (to_float64(X_more) @ P).astype(np.float32)


def _remap_test_labels(g_tr, g_te):
    _, gtr = _onehot(g_tr)
    levels = np.unique(np.asarray(g_tr, dtype=object).astype(str))
    gs_te = np.asarray(g_te, dtype=object).astype(str)
    pos = np.minimum(np.searchsorted(levels, gs_te), len(levels) - 1)
    gte = np.where(levels[pos] == gs_te, pos, -1)
    return gtr, gte, (gte >= 0), list(levels)


def _macro_ovr_auroc(clf, X_keep, y_keep) -> float:
    try:
        proba = clf.predict_proba(X_keep)
        if proba.shape[1] == 2:
            return float(roc_auc_score((y_keep == 1).astype(int), proba[:, 1]))
        present = np.unique(y_keep)
        cols = [list(clf.classes_).index(c) for c in present]
        return float(roc_auc_score(y_keep, proba[:, cols], multi_class="ovr",
                                   average="macro", labels=present))
    except ValueError:
        return float("nan")


def decodability(X_tr, g_tr, X_te, g_te, seed=0) -> float:
    gtr, gte, keep, _ = _remap_test_labels(g_tr, g_te)
    if keep.sum() < 5 or len(np.unique(gtr)) < 2:
        return float("nan")
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    clf.fit(X_tr, gtr)
    return _macro_ovr_auroc(clf, X_te[keep], gte[keep])


def decodability_nonlinear(X_tr, g_tr, X_te, g_te, seed=0, hidden=256,
                           max_iter=300) -> float:
    from sklearn.neural_network import MLPClassifier
    gtr, gte, keep, _ = _remap_test_labels(g_tr, g_te)
    if keep.sum() < 5 or len(np.unique(gtr)) < 2:
        return float("nan")
    sc, Xs = fit_standardizer(to_float64(X_tr))
    clf = MLPClassifier(hidden_layer_sizes=(hidden,), activation="relu",
                        alpha=1e-3, max_iter=max_iter, random_state=seed,
                        early_stopping=True, n_iter_no_change=10)
    clf.fit(Xs, gtr)
    del Xs
    keep_idx = np.where(keep)[0]
    return _macro_ovr_auroc(clf, scaled_block(sc, X_te, keep_idx), gte[keep])

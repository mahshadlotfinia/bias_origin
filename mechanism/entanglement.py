"""
mechanism/entanglement.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from Inference.stats_utils import subspace_overlap
from mitigation.erasure import leace, decodability, decodability_nonlinear
from analysis.heads import (split_masks, fit_head, score_head, fit_standardizer,
                            to_float64)
from analysis.fairness_metrics import auroc


def _coef(Xs, y, seed):
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    clf.fit(Xs, y)
    return clf.coef_


def geometric_overlap(W_sensitive: np.ndarray, W_disease: np.ndarray) -> float:
    return subspace_overlap(W_sensitive.T, W_disease.T)


def leace_collateral(X_tr, g_tr, y_tr, X_te, y_te, seed=0,
                     standardized=None) -> Dict[str, float]:
    clf, sc = fit_head("linear", X_tr, y_tr, standardized=standardized)
    before = auroc(y_te, score_head(clf, sc, X_te))
    try:
        Xtr_e, Xte_e = leace(X_tr, g_tr, X_te, seed)
        clf2, sc2 = fit_head("linear", Xtr_e, y_tr)
        after = auroc(y_te, score_head(clf2, sc2, Xte_e))
    except Exception:
        after = float("nan")
    collat = (before - after) if (np.isfinite(before) and np.isfinite(after)) else float("nan")
    return {"before": before, "after": after, "collateral": collat}


class PoolPrep:

    __slots__ = ("finite", "g", "g_ok", "tr_m", "va_m", "te_m")

    def __init__(self, finite: np.ndarray, g: np.ndarray, g_ok: np.ndarray,
                 tr_m: np.ndarray, va_m: np.ndarray, te_m: np.ndarray):
        self.finite = finite
        self.g = g
        self.g_ok = g_ok
        self.tr_m = tr_m
        self.va_m = va_m
        self.te_m = te_m


def prepare_pool(X: np.ndarray, man: pd.DataFrame, attr: str) -> PoolPrep:
    g = man[attr].astype(str).values if attr in man.columns \
        else np.array(["nan"] * len(man))
    g_ok = pd.Series(g).astype(str).str.lower().values != "nan"
    tr_m, va_m, te_m = split_masks(man)
    return PoolPrep(np.isfinite(X).all(axis=1), g, g_ok, tr_m, va_m, te_m)


def _split_for(X, man, finding, attr, prep: Optional[PoolPrep] = None):
    if finding not in man.columns:
        return None
    p = prep if prep is not None else prepare_pool(X, man, attr)
    y = pd.to_numeric(man[finding], errors="coerce").values
    has = np.isfinite(y) & p.finite & p.g_ok
    tr = np.where((p.tr_m | p.va_m) & has)[0]
    te = np.where(p.te_m & has)[0]
    if min(tr.size, te.size) < 30 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        return None
    if len(np.unique(p.g[tr])) < 2:
        return None
    return y, p.g, tr, te


def entanglement_for(X, man, finding, attr, seed=0, prep: Optional[PoolPrep] = None,
                     with_decodability: bool = True,
                     timings: Optional[Dict[str, float]] = None
                     ) -> Optional[Dict[str, float]]:
    sp = _split_for(X, man, finding, attr, prep)
    if sp is None:
        return None
    y, g, tr, te = sp
    Xtr, Xte = X[tr], X[te]
    ytr, yte, gtr, gte = y[tr], y[te], g[tr], g[te]

    def timed(name, fn):
        if timings is None:
            return fn()
        t0 = time.time()
        out = fn()
        timings[name] = timings.get(name, 0.0) + (time.time() - t0)
        return out

    scaler, Xtr_s = timed("standardize", lambda: fit_standardizer(to_float64(Xtr)))
    W_s = timed("coef_sensitive", lambda: _coef(Xtr_s, gtr, seed))
    W_d = timed("coef_disease", lambda: _coef(Xtr_s, ytr, seed))
    overlap = timed("overlap", lambda: geometric_overlap(W_s, W_d))
    if with_decodability:
        decod = timed("decodability", lambda: decodability(Xtr, gtr, Xte, gte, seed))
        decod_nl = timed("decodability_nl",
                         lambda: decodability_nonlinear(Xtr, gtr, Xte, gte, seed))
    else:
        decod = decod_nl = float("nan")
    collat = timed("leace_collateral",
                   lambda: leace_collateral(Xtr, gtr, ytr, Xte, yte, seed,
                                            standardized=(scaler, Xtr_s)))
    return {"decodability": decod, "decodability_nonlinear": decod_nl,
            "geometric_overlap": overlap,
            "leace_collateral": collat["collateral"], "disease_auroc": collat["before"],
            "n_test": int(len(te))}


def format_timings(timings: Dict[str, float]) -> str:
    items = sorted(timings.items(), key=lambda kv: -kv[1])
    return f"{sum(timings.values()):.0f}s total: " + \
           ", ".join(f"{k} {v:.0f}s" for k, v in items)


def _clean(F, y):
    F = np.asarray(F, float); y = np.asarray(y, float)
    m = np.isfinite(y) & np.isfinite(F).all(axis=1)
    return F[m], y[m], m


def cross_fit_r2(F: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 0) -> float:
    F, y, _ = _clean(F, y)
    if len(y) < max(2 * n_splits, 6):
        return float("nan")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.zeros_like(y)
    for tr, te in kf.split(F):
        sc = StandardScaler().fit(F[tr])
        m = Ridge(alpha=1.0).fit(sc.transform(F[tr]), y[tr])
        pred[te] = m.predict(sc.transform(F[te]))
    return float(r2_score(y, pred))


def logo_r2(F: np.ndarray, y: np.ndarray, groups, seed: int = 0) -> float:
    F, y, mask = _clean(F, y)
    groups = np.asarray(groups)[mask]
    if len(np.unique(groups)) < 3:
        return float("nan")
    logo = LeaveOneGroupOut()
    pred = np.zeros_like(y)
    for tr, te in logo.split(F, y, groups):
        sc = StandardScaler().fit(F[tr])
        m = Ridge(alpha=1.0).fit(sc.transform(F[tr]), y[tr])
        pred[te] = m.predict(sc.transform(F[te]))
    return float(r2_score(y, pred))


def permutation_r2_null(F, y, n_perm=1000, n_splits=5, seed=0) -> Tuple[float, float]:
    obs = cross_fit_r2(F, y, n_splits, seed)
    if not np.isfinite(obs):
        return obs, float("nan")
    F2, y2, _ = _clean(F, y)
    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(n_perm):
        if cross_fit_r2(F2, rng.permutation(y2), n_splits, seed) >= obs:
            count += 1
    return obs, (count + 1) / (n_perm + 1)


def transfer_r2(F_train, y_train, F_test, y_test) -> float:
    Ftr, ytr, _ = _clean(F_train, y_train)
    Fte, yte, _ = _clean(F_test, y_test)
    if len(ytr) < 4 or len(yte) < 2:
        return float("nan")
    sc = StandardScaler().fit(Ftr)
    m = Ridge(alpha=1.0).fit(sc.transform(Ftr), ytr)
    return float(r2_score(yte, m.predict(sc.transform(Fte))))

"""
analysis/fairness_metrics.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

from typing import Dict, List, Optional

import numpy as np
from scipy.stats import rankdata


def _finite_pair(y_true, y_score):
    y_true = np.asarray(y_true, float)
    y_score = np.asarray(y_score, float)
    m = np.isfinite(y_true) & np.isfinite(y_score)
    return y_true[m], y_score[m]


def _auroc_from_ranks(y_true_bin: np.ndarray, ranks: np.ndarray) -> float:
    n_pos = np.sum(y_true_bin == 1.0)
    n_neg = np.sum(y_true_bin == 0.0)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = ranks[y_true_bin == 1.0].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auroc(y_true, y_score) -> float:
    y_true, y_score = _finite_pair(y_true, y_score)
    if y_true.size < 2 or len(np.unique(y_true)) < 2:
        return float("nan")
    return _auroc_from_ranks(y_true, rankdata(y_score))


def _subgroups(group) -> List[str]:
    vals = []
    for g in group:
        if g is None:
            continue
        s = str(g)
        if s and s.lower() not in ("nan", "none"):
            vals.append(s)
    return sorted(set(vals))


class GroupCodes:

    __slots__ = ("codes", "names")

    def __init__(self, codes: np.ndarray, names: List[str]):
        self.codes = np.asarray(codes)
        self.names = list(names)

    def take(self, rows: np.ndarray) -> "GroupCodes":
        return GroupCodes(self.codes[rows], self.names)


def as_group_strings(group) -> np.ndarray:
    try:
        return np.asarray(group, dtype=object).astype(str)
    except Exception:
        return np.asarray([str(g) for g in group], dtype=object).astype(str)


def make_group_codes(group) -> GroupCodes:
    gs = as_group_strings(group)
    uniq = np.unique(gs)
    names = sorted(v for v in uniq.tolist()
                   if v and v.lower() not in ("nan", "none"))
    if not names:
        return GroupCodes(np.full(len(gs), -1, dtype=np.int32), [])
    names_arr = np.asarray(names)
    pos = np.searchsorted(names_arr, gs)
    pos = np.minimum(pos, len(names) - 1)
    codes = np.where(names_arr[pos] == gs, pos, -1).astype(np.int32)
    return GroupCodes(codes, names)


def panel_point_fn(names: List[str], threshold: Optional[float], n_bins: int = 15):
    def _f(label, score, codes):
        return compute_fairness_point(label, score, GroupCodes(codes, names),
                                      threshold, n_bins)
    return _f


def subgroup_auroc(y_true, y_score, group) -> Dict[str, float]:
    y_true = np.asarray(y_true, float)
    y_score = np.asarray(y_score, float)
    group = np.asarray([str(g) for g in group], dtype=object)
    out = {}
    for sub in _subgroups(group):
        m = group == sub
        out[sub] = auroc(y_true[m], y_score[m])
    return out


def es_auc(y_true, y_score, group) -> float:
    overall = auroc(y_true, y_score)
    if not np.isfinite(overall):
        return float("nan")
    subs = subgroup_auroc(y_true, y_score, group)
    devs = [abs(overall - a) for a in subs.values() if np.isfinite(a)]
    if not devs:
        return float("nan")
    return float(overall / (1.0 + sum(devs)))


def _rates_at_threshold(y_true, y_score, threshold):
    y_true, y_score = _finite_pair(y_true, y_score)
    if y_true.size == 0:
        return float("nan"), float("nan"), float("nan")
    pred = (y_score >= threshold).astype(float)
    pos = y_true == 1.0
    neg = y_true == 0.0
    tpr = float(pred[pos].mean()) if pos.any() else float("nan")
    fpr = float(pred[neg].mean()) if neg.any() else float("nan")
    fnr = (1.0 - tpr) if np.isfinite(tpr) else float("nan")
    return tpr, fpr, fnr


def group_rates(y_true, y_score, group, threshold) -> Dict[str, Dict[str, float]]:
    y_true = np.asarray(y_true, float)
    y_score = np.asarray(y_score, float)
    group = np.asarray([str(g) for g in group], dtype=object)
    out = {}
    for sub in _subgroups(group):
        m = group == sub
        tpr, fpr, fnr = _rates_at_threshold(y_true[m], y_score[m], threshold)
        out[sub] = {"tpr": tpr, "fpr": fpr, "fnr": fnr}
    return out


def ece(y_true, y_score, n_bins: int = 15) -> float:
    y_true, y_score = _finite_pair(y_true, y_score)
    if y_true.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_score, bins) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf = float(y_score[m].mean())
        acc = float(y_true[m].mean())
        total += (m.mean()) * abs(acc - conf)
    return float(total)


def group_ece(y_true, y_score, group, n_bins: int = 15) -> Dict[str, float]:
    y_true = np.asarray(y_true, float)
    y_score = np.asarray(y_score, float)
    group = np.asarray([str(g) for g in group], dtype=object)
    return {sub: ece(y_true[group == sub], y_score[group == sub], n_bins)
            for sub in _subgroups(group)}


def _gap(values) -> float:
    v = [x for x in values if np.isfinite(x)]
    return float(max(v) - min(v)) if len(v) >= 2 else float("nan")


def _worst(values) -> float:
    v = [x for x in values if np.isfinite(x)]
    return float(min(v)) if v else float("nan")


def compute_fairness_point(
    y_true,
    y_score,
    group,
    threshold: Optional[float] = None,
    n_bins: int = 15,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    finite = np.isfinite(y_true) & np.isfinite(y_score)
    if isinstance(group, GroupCodes):
        subs = group.names
        codes = group.codes
        masks = {s: (codes == i) & finite for i, s in enumerate(subs)}
    else:
        group_str = np.asarray([str(g) for g in group], dtype=object)
        subs = _subgroups(group_str)
        masks = {s: (group_str == s) & finite for s in subs}

    out: Dict[str, float] = {}

    yt_f, ys_f = y_true[finite], y_score[finite]
    if yt_f.size >= 2 and len(np.unique(yt_f)) >= 2:
        overall = _auroc_from_ranks(yt_f, rankdata(ys_f))
    else:
        overall = float("nan")
    out["auroc_overall"] = overall

    sub_auc: Dict[str, float] = {}
    for s in subs:
        m = masks[s]
        yt_s, ys_s = y_true[m], y_score[m]
        if yt_s.size < 2 or len(np.unique(yt_s)) < 2:
            sub_auc[s] = float("nan")
            continue
        sub_auc[s] = _auroc_from_ranks(yt_s, rankdata(ys_s))
    for s, a in sub_auc.items():
        out[f"auroc::{s}"] = a
    out["auroc_gap"] = _gap(sub_auc.values())
    out["auroc_worst"] = _worst(sub_auc.values())

    if np.isfinite(overall):
        devs = [abs(overall - a) for a in sub_auc.values() if np.isfinite(a)]
        out["es_auc"] = float(overall / (1.0 + sum(devs))) if devs else float("nan")
    else:
        out["es_auc"] = float("nan")

    if threshold is not None:
        pred = (y_score >= threshold).astype(np.float64)
        rates: Dict[str, Dict[str, float]] = {}
        for s in subs:
            m = masks[s]
            yt_s, pr_s = y_true[m], pred[m]
            pos = yt_s == 1.0
            neg = yt_s == 0.0
            tpr = float(pr_s[pos].mean()) if pos.any() else float("nan")
            fpr = float(pr_s[neg].mean()) if neg.any() else float("nan")
            fnr = (1.0 - tpr) if np.isfinite(tpr) else float("nan")
            rates[s] = {"tpr": tpr, "fpr": fpr, "fnr": fnr}
            out[f"tpr::{s}"] = tpr
            out[f"fpr::{s}"] = fpr
            out[f"fnr::{s}"] = fnr
        out["tpr_gap"] = _gap(r["tpr"] for r in rates.values())
        out["fpr_gap"] = _gap(r["fpr"] for r in rates.values())
        out["fnr_gap"] = _gap(r["fnr"] for r in rates.values())

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx_all = np.clip(np.digitize(y_score, bins) - 1, 0, n_bins - 1)
    g_ece: Dict[str, float] = {}
    for s in subs:
        m = masks[s]
        if not m.any():
            g_ece[s] = float("nan")
            continue
        bidx, yt_s, ys_s = bin_idx_all[m], y_true[m], y_score[m]
        total = 0.0
        for b in range(n_bins):
            bm = bidx == b
            if not bm.any():
                continue
            conf = float(ys_s[bm].mean())
            acc = float(yt_s[bm].mean())
            total += bm.mean() * abs(acc - conf)
        g_ece[s] = float(total)
    for s, e in g_ece.items():
        out[f"ece::{s}"] = e
    out["ece_gap"] = _gap(g_ece.values())
    return out

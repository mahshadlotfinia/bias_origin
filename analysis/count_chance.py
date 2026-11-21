"""
analysis/count_chance.py
Created on November 20, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

from typing import List, Tuple

import numpy as np


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    if not np.isfinite(auc) or n_pos < 1 or n_neg < 1:
        return float("nan")
    a = float(min(max(auc, 1e-6), 1 - 1e-6))
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    var = (a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) \
        / (float(n_pos) * float(n_neg))
    return float(np.sqrt(var)) if var > 0 else 0.0


def _sim_subgroup_hanley(auc, n_pos, n_neg, n_sim, rng) -> np.ndarray:
    se = hanley_mcneil_se(auc, n_pos, n_neg)
    if not np.isfinite(se):
        return np.full(n_sim, np.nan)
    return np.clip(rng.normal(auc, se, n_sim), 0.0, 1.0)


def _sim_subgroup_rate(rate, n_den, _unused, n_sim, rng) -> np.ndarray:
    if not np.isfinite(rate) or n_den < 1:
        return np.full(n_sim, np.nan)
    r = float(min(max(rate, 0.0), 1.0))
    return rng.binomial(int(n_den), r, n_sim) / float(n_den)


def simulate_null_gap(value: float, counts: List[Tuple[int, int]], n_sim: int,
                      rng, method: str = "hanley") -> np.ndarray:
    if method in ("rate_tpr", "rate_fpr"):
        k = 0 if method == "rate_tpr" else 1
        draws = np.vstack([_sim_subgroup_rate(value, c[k], None, n_sim, rng)
                           for c in counts])
    else:
        sim = _sim_subgroup_hanley
        draws = np.vstack([sim(value, npos, nneg, n_sim, rng) for npos, nneg in counts])
    return np.nanmax(draws, axis=0) - np.nanmin(draws, axis=0)

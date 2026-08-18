"""
mitigation/ceiling.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

from typing import Dict, List

import numpy as np


def compute_ceiling(methods: List[Dict], unmit_gap: float, unmit_auroc: float,
                    tol_pts: float = 1.0) -> Dict:
    floor = unmit_auroc - tol_pts / 100.0
    eligible = [m for m in methods
                if np.isfinite(m.get("disease_auroc", np.nan))
                and np.isfinite(m.get("gap", np.nan))
                and m["disease_auroc"] >= floor]
    pool = eligible + [{"name": "unmitigated", "disease_auroc": unmit_auroc,
                        "gap": unmit_gap}]
    best = min(pool, key=lambda m: m["gap"])
    return {
        "ceiling_gap": float(best["gap"]),
        "ceiling_method": best["name"],
        "unmit_gap": float(unmit_gap),
        "unmit_auroc": float(unmit_auroc),
        "gap_reduction_at_ceiling": float(unmit_gap - best["gap"]),
        "n_eligible": len(eligible),
        "auroc_tolerance_pts": float(tol_pts),
    }


def tradeoff_table(methods: List[Dict], unmit_gap: float, unmit_auroc: float):
    cost, reduction = [], []
    for m in methods:
        a, g = m.get("disease_auroc", np.nan), m.get("gap", np.nan)
        if np.isfinite(a) and np.isfinite(g):
            cost.append(unmit_auroc - a)
            reduction.append(unmit_gap - g)
    return np.asarray(cost, float), np.asarray(reduction, float)

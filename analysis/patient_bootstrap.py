"""
analysis/patient_bootstrap.py
Created on November 20, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


MEASURES = ("auroc", "tpr", "fpr")


def boot_weights(n_clusters: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    w = np.empty((n_boot, n_clusters), dtype=np.uint8)
    for b in range(n_boot):
        w[b] = np.minimum(np.bincount(rng.randint(0, n_clusters, n_clusters), minlength=n_clusters), 255)
    return w


class GroupRows:

    def __init__(self, y: np.ndarray, s: np.ndarray, cluster: np.ndarray, threshold: float):
        order = np.argsort(s, kind="stable")
        self.s = np.asarray(s, dtype=np.float64)[order]
        self.pos = (np.asarray(y)[order] == 1).astype(np.float32)
        self.neg = 1.0 - self.pos
        self.cluster = np.asarray(cluster)[order]
        self.called = (self.s >= float(threshold)).astype(np.float32)
        _, first, counts = np.unique(self.s, return_index=True, return_counts=True)
        block = np.repeat(np.arange(first.size), counts)
        self.first = first[block]
        self.last = (first + counts - 1)[block]

    def stats(self, w: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        if w is None:
            w = np.ones((1, self.s.size), dtype=np.float32)
        wp, wn = w * self.pos, w * self.neg
        cneg = np.cumsum(wn, axis=1)
        below = np.where(self.first > 0, cneg[:, np.maximum(self.first - 1, 0)], 0.0)
        equal = cneg[:, self.last] - below
        u = (wp * (below + 0.5 * equal)).sum(axis=1, dtype=np.float64)
        n_pos, n_neg = wp.sum(axis=1, dtype=np.float64), wn.sum(axis=1, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            return {"auroc": u / (n_pos * n_neg),
                    "tpr": (wp * self.called).sum(axis=1, dtype=np.float64) / n_pos,
                    "fpr": (wn * self.called).sum(axis=1, dtype=np.float64) / n_neg}

    def boot(self, weights: np.ndarray, chunk: int = 64) -> Dict[str, np.ndarray]:
        out = {m: np.empty(weights.shape[0]) for m in MEASURES}
        for a in range(0, weights.shape[0], chunk):
            st = self.stats(weights[a:a + chunk][:, self.cluster].astype(np.float32))
            for m in MEASURES:
                out[m][a:a + chunk] = st[m]
        return out


def difference_against_chance(point: np.ndarray, boot: np.ndarray) -> Dict[str, float]:
    obs = float(np.nanmax(point) - np.nanmin(point))
    ok = np.isfinite(boot).all(axis=1)
    b = boot[ok]
    dev = b - point[None, :]
    chance = dev.max(axis=1) - dev.min(axis=1)
    spread = b.max(axis=1) - b.min(axis=1)
    return {"difference": obs, "difference_std": float(spread.std(ddof=1)), "difference_ci_low": float(np.percentile(spread, 2.5)),
            "difference_ci_high": float(np.percentile(spread, 97.5)),
            "chance": float(chance.mean()), "chance_std": float(chance.std(ddof=1)), "chance_ci_low": float(np.percentile(chance, 2.5)),
            "chance_ci_high": float(np.percentile(chance, 97.5)),
            "p_raw": float((1 + (chance >= obs - 1e-12).sum()) / (chance.size + 1)), "n_boot_used": int(ok.sum())}


def summarize(point: float, boot: np.ndarray) -> Tuple[float, float, float, float]:
    b = boot[np.isfinite(boot)]
    if b.size < 2:
        return float(point), np.nan, np.nan, np.nan
    return float(point), float(b.std(ddof=1)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def eligible_groups(y: np.ndarray, group: np.ndarray, min_n: int) -> List[str]:
    out = []
    for g in sorted({str(v) for v in group if v is not None and str(v) not in ("nan", "", "None")}):
        m = group == g
        if m.sum() >= min_n and (y[m] == 1).any() and (y[m] == 0).any():
            out.append(g)
    return out

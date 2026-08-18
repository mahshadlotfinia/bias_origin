"""
theory/proposition.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

from typing import Tuple

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def two_directions(d_signal: int, rho: float, rng: np.random.RandomState):
    w_d = _unit(rng.randn(d_signal))
    r = rng.randn(d_signal)
    w_perp = _unit(r - (r @ w_d) * w_d)
    rho = float(np.clip(rho, -0.999, 0.999))
    w_a = rho * w_d + np.sqrt(1 - rho ** 2) * w_perp
    return w_d, _unit(w_a)


def make_synthetic(
    n: int = 4000,
    d_signal: int = 32,
    d_nuisance: int = 128,
    rho: float = 0.5,
    alpha: float = 1.2,
    beta: float = 0.5,
    gamma: float = 2.0,
    noise: float = 1.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.RandomState(seed)
    D = d_signal + d_nuisance
    w_d, w_a = two_directions(d_signal, rho, rng)
    realized_rho = float(w_d @ w_a)

    y = rng.randint(0, 2, size=n).astype(np.float64)
    a = rng.randint(0, 2, size=n).astype(np.float64)

    X = rng.randn(n, D) * noise
    disease = (alpha * (2 * y - 1))[:, None] * w_d[None, :]
    mean_shift = (beta * (2 * a - 1))[:, None] * w_a[None, :]
    het = ((a == 1).astype(np.float64) * gamma * rng.randn(n))[:, None] * w_a[None, :]
    X[:, :d_signal] += disease + mean_shift + het
    a_str = np.array([str(int(v)) for v in a], dtype=object)
    return X.astype(np.float32), y, a_str, realized_rho


def split_indices(n: int, seed: int = 0, fracs=(0.6, 0.15, 0.25)):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_tr = int(fracs[0] * n); n_va = int(fracs[1] * n)
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

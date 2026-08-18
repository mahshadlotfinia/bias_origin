"""
Inference/stats_utils.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

N_BOOT    = 1000
N_PERM    = 1000
BOOT_SEED = 0


def bootstrap_proportion(values: np.ndarray, n_boot: int = N_BOOT,
                         seed: int = BOOT_SEED) -> dict:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "n": 0}
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    return {"point": float(values.mean()), "std": float(boot_means.std(ddof=1)),
            "ci_lower": float(np.percentile(boot_means, 2.5)),
            "ci_upper": float(np.percentile(boot_means, 97.5)), "n": int(n)}


def paired_bootstrap_diff(values_a: np.ndarray, values_b: np.ndarray,
                          n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    n = len(values_a)
    assert len(values_b) == n, "Both arrays must have equal length."
    if n == 0:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "p_value": np.nan, "n": 0}
    rng   = np.random.RandomState(seed)
    idx   = rng.randint(0, n, size=(n_boot, n))
    point = float(values_a.mean() - values_b.mean())
    boot_diffs = values_a[idx].mean(axis=1) - values_b[idx].mean(axis=1)
    centered   = boot_diffs - point
    p_value    = float(np.mean(np.abs(centered) >= abs(point)))
    p_value    = max(p_value, 1.0 / n_boot)
    return {"point": point, "std": float(boot_diffs.std(ddof=1)),
            "ci_lower": float(np.percentile(boot_diffs, 2.5)),
            "ci_upper": float(np.percentile(boot_diffs, 97.5)),
            "p_value": p_value, "n": int(n)}


def permutation_test_2groups(values_a: np.ndarray, values_b: np.ndarray,
                             n_perm: int = N_PERM, seed: int = BOOT_SEED) -> float:
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    if len(values_a) < 2 or len(values_b) < 2:
        return float("nan")
    obs    = abs(float(values_a.mean() - values_b.mean()))
    pooled = np.concatenate([values_a, values_b])
    n_a    = len(values_a)
    rng    = np.random.RandomState(seed)
    count  = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        if abs(perm[:n_a].mean() - perm[n_a:].mean()) >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def permutation_test_kgroups(groups: List[np.ndarray], n_perm: int = N_PERM,
                             seed: int = BOOT_SEED) -> float:
    groups = [np.asarray(g, dtype=float) for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return float("nan")
    sizes   = [len(g) for g in groups]
    pooled  = np.concatenate(groups)
    n_total = len(pooled)
    k       = len(groups)

    def _f(gs):
        grand = np.concatenate(gs).mean()
        ss_b  = sum(len(g) * (g.mean() - grand) ** 2 for g in gs)
        ss_w  = sum(((g - g.mean()) ** 2).sum() for g in gs)
        return float("inf") if ss_w == 0 else (ss_b / (k - 1)) / (ss_w / (n_total - k))

    obs   = _f(groups)
    rng   = np.random.RandomState(seed)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        start, pgs = 0, []
        for s in sizes:
            pgs.append(perm[start:start + s]); start += s
        if _f(pgs) >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    if n == 0:
        return np.array([])
    nan_mask  = np.isnan(p_values)
    valid_idx = np.where(~nan_mask)[0]
    if len(valid_idx) == 0:
        return p_values.copy()
    valid_p  = p_values[valid_idx]
    m        = len(valid_p)
    order    = np.argsort(valid_p)
    p_sorted = valid_p[order]
    adj = p_sorted * m / (np.arange(m, dtype=float) + 1)
    for i in range(m - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    adj = np.minimum(adj, 1.0)
    inv_order = np.empty(m, dtype=int)
    inv_order[order] = np.arange(m)
    result = p_values.copy()
    result[valid_idx] = adj[inv_order]
    return result


def _percentile_ci(boot: np.ndarray) -> tuple:
    boot = boot[np.isfinite(boot)]
    if boot.size == 0:
        return (np.nan, np.nan, np.nan)
    std = float(np.std(boot, ddof=1)) if boot.size > 1 else np.nan
    return (std, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def paired_bootstrap_statistic(arrays: List[np.ndarray], stat_fn,
                               n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    n = len(arrays[0])
    assert all(len(a) == n for a in arrays), "all arrays must be equal length"
    if n == 0:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "n": 0}
    point = float(stat_fn(*arrays))
    rng   = np.random.RandomState(seed)
    idx   = rng.randint(0, n, size=(n_boot, n))
    boot  = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        try:
            boot[b] = stat_fn(*[a[idx[b]] for a in arrays])
        except Exception:
            boot[b] = np.nan
    std, lo, hi = _percentile_ci(boot)
    return {"point": point, "std": std, "ci_lower": lo, "ci_upper": hi, "n": int(n)}


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT,
                       seed: int = BOOT_SEED) -> dict:
    from scipy.stats import spearmanr
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 4:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "n": int(len(x))}

    def _rho(a, b):
        r, _ = spearmanr(a, b); return r
    return paired_bootstrap_statistic([x, y], _rho, n_boot=n_boot, seed=seed)


def bootstrap_slope(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT,
                    seed: int = BOOT_SEED) -> dict:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "n": int(len(x))}

    def _slope(a, b):
        A = np.vstack([a, np.ones_like(a)]).T
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        return coef[0]
    return paired_bootstrap_statistic([x, y], _slope, n_boot=n_boot, seed=seed)


def bootstrap_auroc(y_true: np.ndarray, y_score: np.ndarray, n_boot: int = N_BOOT,
                    seed: int = BOOT_SEED) -> dict:
    from sklearn.metrics import roc_auc_score
    y_true  = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    n = len(y_true)
    if n < 2 or len(np.unique(y_true)) < 2:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "n": int(n)}
    point = float(roc_auc_score(y_true, y_score))
    rng   = np.random.RandomState(seed)
    idx   = rng.randint(0, n, size=(n_boot, n))
    boot  = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        yt, ys = y_true[idx[b]], y_score[idx[b]]
        boot[b] = roc_auc_score(yt, ys) if len(np.unique(yt)) >= 2 else np.nan
    std, lo, hi = _percentile_ci(boot)
    return {"point": point, "std": std, "ci_lower": lo, "ci_upper": hi, "n": int(n)}


def _cluster_layout(cluster_ids: np.ndarray):
    arr = np.asarray(cluster_ids)
    clusters = arr if arr.dtype.kind in "iu" else np.asarray(arr, dtype=object).astype(str)
    order = np.argsort(clusters, kind="stable")
    sorted_clusters = clusters[order]
    uniq, start_idx = np.unique(sorted_clusters, return_index=True)
    counts = np.diff(np.append(start_idx, len(order)))
    return uniq, order, start_idx.astype(np.int64), counts.astype(np.int64)


def _gather_rows(order: np.ndarray, starts: np.ndarray, counts: np.ndarray,
                 chosen: np.ndarray) -> np.ndarray:
    c = counts[chosen]
    total = int(c.sum())
    if total == 0:
        return np.empty(0, dtype=order.dtype)
    idx = np.ones(total, dtype=np.int64)
    idx[0] = starts[chosen[0]]
    if len(chosen) > 1:
        idx[np.cumsum(c)[:-1]] = starts[chosen[1:]] - (starts[chosen[:-1]] + c[:-1]) + 1
    np.cumsum(idx, out=idx)
    return order[idx]


def resolve_n_jobs(n_jobs) -> int:
    if isinstance(n_jobs, str):
        s = n_jobs.strip().lower()
        n_jobs = 0 if s in ("auto", "") else int(s)
    n_jobs = int(n_jobs)
    if n_jobs >= 1:
        return n_jobs
    try:
        avail = len(os.sched_getaffinity(0))
    except AttributeError:
        avail = os.cpu_count() or 1
    return max(1, min(32, avail))


_REPLICATE_STATE: Dict[str, object] = {}


def _replicate_chunk(bounds):
    lo, hi = bounds
    one = _REPLICATE_STATE["one"]
    chosen = _REPLICATE_STATE["chosen"]
    return [one(chosen[b]) for b in range(lo, hi)]


def _map_replicates(one: Callable, all_chosen: np.ndarray, n_jobs: int) -> List[dict]:
    n = len(all_chosen)
    if n_jobs <= 1 or n < 2:
        return [one(c) for c in all_chosen]
    _torch = sys.modules.get("torch")
    if _torch is not None and getattr(_torch, "cuda", None) is not None:
        try:
            if _torch.cuda.is_initialized():
                print("[stats] CUDA is initialized in this process; bootstrap runs "
                      "serially rather than forking.")
                return [one(c) for c in all_chosen]
        except Exception:
            pass
    try:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
    except (ImportError, ValueError) as e:
        print(f"[stats] no fork start method ({e}); bootstrap runs serially.")
        return [one(c) for c in all_chosen]
    edges = np.linspace(0, n, min(n_jobs, n) + 1).astype(int)
    bounds = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    _REPLICATE_STATE["one"] = one
    _REPLICATE_STATE["chosen"] = all_chosen
    try:
        with ctx.Pool(processes=len(bounds)) as pool:
            chunks = pool.map(_replicate_chunk, bounds, chunksize=1)
        return [r for chunk in chunks for r in chunk]
    except Exception as e:
        print(f"[stats] parallel bootstrap failed ({e}); falling back to serial.")
        return [one(c) for c in all_chosen]
    finally:
        _REPLICATE_STATE.clear()


def _summarize(point: Dict[str, float], results: List[dict],
               n_clusters: int) -> Dict[str, dict]:
    boot_vals: Dict[str, List[float]] = {k: [] for k in point}
    for res in results:
        for k in boot_vals:
            boot_vals[k].append(res.get(k, np.nan))
    out: Dict[str, dict] = {}
    for k, v in point.items():
        arr = np.asarray(boot_vals[k], dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size > 1:
            std = float(np.std(arr, ddof=1))
            lo, hi = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
        else:
            std, lo, hi = np.nan, np.nan, np.nan
        out[k] = {"point": float(v) if np.isfinite(v) else np.nan,
                  "std": std, "ci_lower": lo, "ci_upper": hi, "n": int(n_clusters)}
    return out


def _cluster_index(cluster_ids: np.ndarray):
    uniq, order, starts, _ = _cluster_layout(cluster_ids)
    return uniq, np.split(order, starts[1:])


def _draw_cluster_resamples(rng, n_clusters: int, n_boot: int) -> np.ndarray:
    idx_all = np.arange(n_clusters)
    out = np.empty((n_boot, n_clusters), dtype=np.int32)
    for b in range(n_boot):
        out[b] = rng.choice(idx_all, size=n_clusters, replace=True)
    return out


def cluster_bootstrap(
    df: pd.DataFrame,
    cluster_col: str,
    point_fn: Callable[[pd.DataFrame], Dict[str, float]],
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    n_jobs: int = 1,
) -> Dict[str, dict]:
    if cluster_col not in df.columns or df[cluster_col].isna().all():
        df = df.copy()
        df = df.reset_index(drop=True)
        df["__row_cluster"] = np.arange(len(df)).astype(str)
        cluster_col = "__row_cluster"

    point = point_fn(df)
    uniq, order, starts, counts = _cluster_layout(df[cluster_col].values)
    n_clusters = len(uniq)
    if n_clusters == 0:
        return {k: {"point": v, "std": np.nan, "ci_lower": np.nan,
                    "ci_upper": np.nan, "n": 0} for k, v in point.items()}

    rng = np.random.RandomState(seed)
    all_chosen = _draw_cluster_resamples(rng, n_clusters, n_boot)

    def _one(chosen):
        return point_fn(df.iloc[_gather_rows(order, starts, counts, chosen)])

    results = _map_replicates(_one, all_chosen, resolve_n_jobs(n_jobs))
    return _summarize(point, results, n_clusters)


def cluster_bootstrap_arrays(
    cluster_ids,
    point_fn: Callable[..., Dict[str, float]],
    arrays: Sequence[np.ndarray],
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    n_jobs: int = 1,
) -> Dict[str, dict]:
    arrays = [np.asarray(a) for a in arrays]
    cluster_ids = np.asarray(cluster_ids)
    point = point_fn(*arrays)
    uniq, order, starts, counts = _cluster_layout(cluster_ids)
    n_clusters = len(uniq)
    if n_clusters == 0:
        return {k: {"point": v, "std": np.nan, "ci_lower": np.nan,
                    "ci_upper": np.nan, "n": 0} for k, v in point.items()}

    rng = np.random.RandomState(seed)
    all_chosen = _draw_cluster_resamples(rng, n_clusters, n_boot)

    def _one(chosen):
        rows = _gather_rows(order, starts, counts, chosen)
        return point_fn(*[a[rows] for a in arrays])

    results = _map_replicates(_one, all_chosen, resolve_n_jobs(n_jobs))
    return _summarize(point, results, n_clusters)


def cluster_bootstrap_paired_diff_arrays(
    cluster_ids,
    point_fn_a: Callable[..., Dict[str, float]],
    point_fn_b: Callable[..., Dict[str, float]],
    arrays: Sequence[np.ndarray],
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    n_jobs: int = 1,
) -> Dict[str, dict]:
    arrays = [np.asarray(a) for a in arrays]
    cluster_ids = np.asarray(cluster_ids)
    pa, pb = point_fn_a(*arrays), point_fn_b(*arrays)
    keys = [k for k in pa if k in pb]
    point = {k: pa[k] - pb[k] for k in keys}

    uniq, order, starts, counts = _cluster_layout(cluster_ids)
    n_clusters = len(uniq)
    rng = np.random.RandomState(seed)
    all_chosen = _draw_cluster_resamples(rng, n_clusters, n_boot)

    def _one(chosen):
        rows = _gather_rows(order, starts, counts, chosen)
        sub = [a[rows] for a in arrays]
        ra, rb = point_fn_a(*sub), point_fn_b(*sub)
        return {k: ra.get(k, np.nan) - rb.get(k, np.nan) for k in keys}

    results = _map_replicates(_one, all_chosen, resolve_n_jobs(n_jobs))
    return _summarize_paired(keys, point, results, n_clusters)


def _summarize_paired(keys, point, results, n_clusters) -> Dict[str, dict]:
    boot_vals: Dict[str, List[float]] = {k: [] for k in keys}
    for res in results:
        for k in keys:
            boot_vals[k].append(res[k])
    out: Dict[str, dict] = {}
    for k in keys:
        arr = np.asarray(boot_vals[k], dtype=float)
        arr = arr[np.isfinite(arr)]
        pt = point[k]
        if arr.size > 1 and np.isfinite(pt):
            std = float(np.std(arr, ddof=1))
            lo, hi = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
            centered = arr - pt
            p = float(np.mean(np.abs(centered) >= abs(pt)))
            p = max(p, 1.0 / max(arr.size, 1))
        else:
            std, lo, hi, p = np.nan, np.nan, np.nan, np.nan
        out[k] = {"point": float(pt) if np.isfinite(pt) else np.nan, "std": std,
                  "ci_lower": lo, "ci_upper": hi, "p_value": p, "n": int(n_clusters)}
    return out


def cluster_bootstrap_paired_diff(
    df: pd.DataFrame,
    cluster_col: str,
    point_fn_a: Callable[[pd.DataFrame], Dict[str, float]],
    point_fn_b: Callable[[pd.DataFrame], Dict[str, float]],
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    n_jobs: int = 1,
) -> Dict[str, dict]:
    if cluster_col not in df.columns or df[cluster_col].isna().all():
        df = df.copy().reset_index(drop=True)
        df["__row_cluster"] = np.arange(len(df)).astype(str)
        cluster_col = "__row_cluster"

    pa, pb = point_fn_a(df), point_fn_b(df)
    keys = [k for k in pa if k in pb]
    point = {k: pa[k] - pb[k] for k in keys}

    uniq, order, starts, counts = _cluster_layout(df[cluster_col].values)
    n_clusters = len(uniq)
    rng = np.random.RandomState(seed)
    all_chosen = _draw_cluster_resamples(rng, n_clusters, n_boot)

    def _one(chosen):
        sub = df.iloc[_gather_rows(order, starts, counts, chosen)]
        ra, rb = point_fn_a(sub), point_fn_b(sub)
        return {k: ra.get(k, np.nan) - rb.get(k, np.nan) for k in keys}

    results = _map_replicates(_one, all_chosen, resolve_n_jobs(n_jobs))
    return _summarize_paired(keys, point, results, n_clusters)


def wilcoxon_signed_rank(x: np.ndarray, y: np.ndarray) -> tuple:
    from scipy.stats import wilcoxon
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 1 or np.all(x == y):
        return (float("nan"), float("nan"))
    try:
        s, p = wilcoxon(x, y)
        return (float(s), float(p))
    except ValueError:
        return (float("nan"), float("nan"))


def friedman_test(*groups: np.ndarray) -> tuple:
    from scipy.stats import friedmanchisquare
    arrs = [np.asarray(g, float) for g in groups]
    n = min(len(a) for a in arrs)
    arrs = [a[:n] for a in arrs]
    if n < 2 or len(arrs) < 3:
        return (float("nan"), float("nan"))
    try:
        s, p = friedmanchisquare(*arrs)
        return (float(s), float(p))
    except ValueError:
        return (float("nan"), float("nan"))


def cohen_kappa_perm(a: np.ndarray, b: np.ndarray, n_perm: int = N_PERM,
                     seed: int = BOOT_SEED) -> tuple:
    from sklearn.metrics import cohen_kappa_score
    a = np.asarray(a); b = np.asarray(b)
    m = pd.notna(a) & pd.notna(b)
    a, b = a[m], b[m]
    if len(a) < 2:
        return (float("nan"), float("nan"))
    obs = float(cohen_kappa_score(a, b))
    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(n_perm):
        if cohen_kappa_score(a, rng.permutation(b)) >= obs:
            count += 1
    return (obs, (count + 1) / (n_perm + 1))


def icc_two_way(ratings: np.ndarray) -> tuple:
    ratings = np.asarray(ratings, float)
    try:
        import pingouin as pg
        n, k = ratings.shape
        long = pd.DataFrame({
            "target": np.repeat(np.arange(n), k),
            "rater":  np.tile(np.arange(k), n),
            "score":  ratings.ravel(),
        })
        res = pg.intraclass_corr(data=long, targets="target", raters="rater",
                                 ratings="score")
        row = res[res["Type"] == "ICC2"].iloc[0]
        return (float(row["ICC"]), float(row["pval"]))
    except Exception:
        n, k = ratings.shape
        grand = ratings.mean()
        ms_r = k * ((ratings.mean(axis=1) - grand) ** 2).sum() / (n - 1)
        ms_c = n * ((ratings.mean(axis=0) - grand) ** 2).sum() / (k - 1)
        resid = ratings - ratings.mean(axis=1, keepdims=True) \
                - ratings.mean(axis=0, keepdims=True) + grand
        ms_e = (resid ** 2).sum() / ((n - 1) * (k - 1))
        denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
        icc = (ms_r - ms_e) / denom if denom > 0 else float("nan")
        from scipy.stats import f as _f
        F = ms_r / ms_e if ms_e > 0 else float("inf")
        p = float(1 - _f.cdf(F, n - 1, (n - 1) * (k - 1))) if np.isfinite(F) else float("nan")
        return (float(icc), p)


def subspace_overlap(A: np.ndarray, B: np.ndarray) -> float:
    Qa, _ = np.linalg.qr(np.asarray(A, float))
    Qb, _ = np.linalg.qr(np.asarray(B, float))
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return float(np.mean(s))


def principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    Qa, _ = np.linalg.qr(np.asarray(A, float))
    Qb, _ = np.linalg.qr(np.asarray(B, float))
    s = np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), -1.0, 1.0)
    return np.arccos(s)


def spearman_with_perm(x: np.ndarray, y: np.ndarray, n_perm: int = N_PERM,
                       seed: int = BOOT_SEED) -> tuple:
    from scipy.stats import spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 4:
        return (float("nan"), float("nan"))
    rho, _ = spearmanr(x, y)
    obs = abs(rho)
    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(n_perm):
        r, _ = spearmanr(x, rng.permutation(y))
        if abs(r) >= obs:
            count += 1
    return (float(rho), (count + 1) / (n_perm + 1))


def permutation_pvalue(observed: float, null_distribution: np.ndarray,
                       tail: str = "greater") -> float:
    null = np.asarray(null_distribution, float)
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(observed):
        return float("nan")
    if tail == "greater":
        count = np.sum(null >= observed)
    elif tail == "less":
        count = np.sum(null <= observed)
    else:
        center = np.median(null)
        count = np.sum(np.abs(null - center) >= abs(observed - center))
    return float((count + 1) / (null.size + 1))

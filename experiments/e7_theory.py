"""
experiments/e7_theory.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config.serde import read_config
from analysis.heads import fit_head, score_head, threshold_at_sensitivity, fit_standardizer, to_float64
from analysis.fairness_metrics import compute_fairness_point
from Inference.stats_utils import spearman_with_perm
from Inference import report_utils as R
from mitigation.erasure import leace, decodability
from mitigation.preprocessing import reweighted
from mitigation.postprocessing import per_group_threshold_shift
from mitigation.ceiling import compute_ceiling
from mechanism.entanglement import geometric_overlap, leace_collateral
from theory.proposition import make_synthetic, split_indices

import warnings
warnings.filterwarnings("ignore")


def _agg(samples, n) -> dict:
    a = np.asarray(samples, float); a = a[np.isfinite(a)]
    if a.size == 0:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "n": n}
    return {"point": float(a.mean()),
            "std": float(a.std(ddof=1)) if a.size > 1 else np.nan,
            "ci_lower": float(np.percentile(a, 2.5)),
            "ci_upper": float(np.percentile(a, 97.5)), "n": int(n)}


def _overlap(Xtr, ytr, atr, seed):
    sc, Xs = fit_standardizer(to_float64(Xtr))
    Wa = LogisticRegression(C=1.0, max_iter=1000, random_state=seed).fit(Xs, atr).coef_
    Wd = LogisticRegression(C=1.0, max_iter=1000, random_state=seed).fit(Xs, ytr).coef_
    return geometric_overlap(Wa, Wd)


def _panel(y, s, g, thr):
    return compute_fairness_point(np.asarray(y, float), np.asarray(s, float),
                                  np.asarray([str(v) for v in g]), thr)


def _one(rho, seed, cfg) -> Dict[str, float]:
    th = cfg["theory"]
    X, y, a, realized = make_synthetic(
        n=int(th["n_synthetic"]), d_signal=int(th["n_signal_dims"]),
        d_nuisance=int(th["n_nuisance_dims"]), rho=rho, seed=seed)
    tr, va, te = split_indices(len(y), seed)
    Xtr, Xva, Xte = X[tr], X[va], X[te]
    ytr, yva, yte = y[tr], y[va], y[te]
    atr, ava, ate = a[tr], a[va], a[te]
    target = float(cfg["stats"]["operating_sensitivity"])
    tol = float(cfg["mitigation"].get("auroc_tolerance_pts", 1.0))

    clf, sc = fit_head("linear", Xtr, ytr)
    base_va = score_head(clf, sc, Xva); base_te = score_head(clf, sc, Xte)
    thr0 = threshold_at_sensitivity(yva, base_va, target)
    p0 = _panel(yte, base_te, ate, thr0)
    gap0, auroc0 = p0["auroc_gap"], p0["auroc_overall"]

    points = [{"name": "unmitigated", "disease_auroc": auroc0, "gap": gap0}]
    try:
        Xtr_e, Xte_e, Xva_e = leace(Xtr, atr, Xte, seed, X_more=Xva)
        clf2, sc2 = fit_head("linear", Xtr_e, ytr)
        lva = score_head(clf2, sc2, Xva_e); lte = score_head(clf2, sc2, Xte_e)
        pl = _panel(yte, lte, ate, threshold_at_sensitivity(yva, lva, target))
        points.append({"name": "leace", "disease_auroc": pl["auroc_overall"], "gap": pl["auroc_gap"]})
    except Exception:
        pass
    try:
        rva, rte = reweighted(Xtr, ytr, atr, Xva, yva, ava, Xte, ate, seed)
        pr = _panel(yte, rte, ate, threshold_at_sensitivity(yva, rva, target))
        points.append({"name": "reweigh", "disease_auroc": pr["auroc_overall"], "gap": pr["auroc_gap"]})
    except Exception:
        pass
    try:
        eva, ete = per_group_threshold_shift(base_va, yva, ava, base_te, ate, target)
        pe = _panel(yte, ete, ate, threshold_at_sensitivity(yva, eva, target))
        points.append({"name": "eo_shift", "disease_auroc": pe["auroc_overall"], "gap": pe["auroc_gap"]})
    except Exception:
        pass

    ceil = compute_ceiling(points, gap0, auroc0, tol)
    return {"rho": realized, "gap": gap0, "ceiling": ceil["ceiling_gap"],
            "disease_auroc": auroc0,
            "decodability": decodability(Xtr, atr, Xte, ate, seed),
            "geometric_overlap": _overlap(Xtr, ytr, atr, seed),
            "leace_collateral": leace_collateral(Xtr, atr, ytr, Xte, yte, seed)["collateral"]}


def main_e7(global_config_path: str) -> Tuple[str, str]:
    cfg = read_config(global_config_path)["BiasOrigin"]
    grid = cfg["theory"]["collinearity_grid"]
    seeds = list(range(int(cfg["theory"]["n_seeds"])))

    rows = []
    for rho in grid:
        for sd in tqdm(seeds, desc=f"[e7] rho={rho}", unit="seed"):
            rows.append({"rho_grid": rho, **_one(rho, sd, cfg)})
        print(f"[e7] rho={rho} done ({len(seeds)} seeds)")
    tab = pd.DataFrame(rows)

    perf_rows: List[Dict] = []
    stat_rows: List[Dict] = []
    metrics = [("gap", "lower", True), ("ceiling", "lower", True),
               ("disease_auroc", "higher", True), ("decodability", "lower", True),
               ("geometric_overlap", "lower", False), ("leace_collateral", "lower", True)]
    for rho in grid:
        sub = tab[tab["rho_grid"] == rho]
        ctx = {"experiment": "e7", "modality": "synthetic", "dataset": "synthetic",
               "encoder": "synthetic", "encoder_objective": "synthetic",
               "attribute": "synthetic_a", "finding": "synthetic",
               "data_composition": f"rho{rho:.2f}", "mitigation": "battery",
               "operating_point": f"sens{float(cfg['stats']['operating_sensitivity']):.2f}"}
        for name, direction, pct in metrics:
            perf_rows.append(R.pack_performance(name, _agg(sub[name].values, len(sub)),
                                                direction, ctx, n_patients=len(sub),
                                                as_percent=pct))

    n_perm = int(cfg["mechanism"]["n_perm"]); seed = int(cfg["stats"]["boot_seed"])
    ctx_all = {"experiment": "e7", "modality": "synthetic", "dataset": "synthetic",
               "attribute": "synthetic_a", "mitigation": "battery"}
    for xcol, label in [("rho_grid", "rho_vs_ceiling"),
                        ("geometric_overlap", "overlap_vs_ceiling"),
                        ("leace_collateral", "collateral_vs_ceiling")]:
        rho_, p = spearman_with_perm(tab[xcol].values, tab["ceiling"].values, n_perm, seed)
        R.report_spearman(stat_rows, {**ctx_all, "finding": label}, rho_, p,
                          fdr_family="e7_theory", n_units=len(tab))
    rho_, p = spearman_with_perm(tab["rho_grid"].values, tab["decodability"].values, n_perm, seed)
    R.report_spearman(stat_rows, {**ctx_all, "finding": "rho_vs_decodability"}, rho_, p,
                      fdr_family="e7_theory", n_units=len(tab))

    R.add_fdr(stat_rows, alpha=float(cfg["stats"]["fdr_alpha"]))
    out_dir = cfg["theory"]["results_e7_dir"]
    os.makedirs(out_dir, exist_ok=True)
    perf_csv = os.path.join(out_dir, "results_performance_e7.csv")
    stat_csv = os.path.join(out_dir, "results_statistics_e7.csv")
    R.write_frame_atomic(R.perf_frame(perf_rows), perf_csv)
    R.write_frame_atomic(R.stat_frame(stat_rows), stat_csv)
    print(f"\n[e7] grid x seeds rows: {len(tab)}")
    print(f"[e7] performance rows: {len(perf_rows)} -> {perf_csv}")
    print(f"[e7] statistics rows:  {len(stat_rows)} -> {stat_csv}")
    return perf_csv, stat_csv

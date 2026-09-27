"""
experiments/e3_mechanism.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.serde import read_config
from data_loader.cxr_harmonization import CANONICAL_PATHOLOGY_FINDINGS
from analysis.embedding_io import load_encoder_pool_frame
from Inference.stats_utils import spearman_with_perm
from Inference import report_utils as R
from mechanism.entanglement import (
    entanglement_for, prepare_pool, cross_fit_r2, logo_r2,
    permutation_r2_null,
)
from experiments.e2_ceiling import _e2_encoders

import warnings
warnings.filterwarnings("ignore")

FINDINGS = CANONICAL_PATHOLOGY_FINDINGS
ATTR = "race_grp"

E3_MANIFEST_COLUMNS = ["split", ATTR] + list(FINDINGS)


def _point(value, n) -> dict:
    return {"point": float(value) if np.isfinite(value) else np.nan,
            "std": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "n": int(n)}


def _emit_point(perf_rows, ctx, metric, value, direction, n, as_percent=True):
    perf_rows.append(R.pack_performance(metric, _point(value, n), direction, ctx,
                                        n_patients=n, as_percent=as_percent))


def _load_ceiling_target(e2_perf_csv: str, attr: str = ATTR) -> Tuple[Dict, Dict]:
    df = pd.read_csv(e2_perf_csv, float_precision="round_trip")
    if "attribute" in df.columns:
        df = df[df["attribute"].astype(str) == attr]
    if df.empty:
        raise ValueError(
            f"[e3] no E2 rows for attribute '{attr}' in {e2_perf_csv}. E3's "
            f"mechanism analysis is defined against one attribute at a time.")
    ceil, unmit = {}, {}
    cg = df[df["metric_name"] == "ceiling_gap"]
    for _, r in cg.iterrows():
        ceil[(str(r["encoder"]), str(r["finding"]))] = float(r["value_raw"])
    ug = df[(df["metric_name"] == "auroc_gap") & (df["mitigation"] == "none")]
    for _, r in ug.iterrows():
        unmit[(str(r["encoder"]), str(r["finding"]))] = float(r["value_raw"])
    return ceil, unmit


def _rows_from_perf(perf_rows: List[Dict]) -> List[Dict]:
    by_key: Dict[Tuple[str, str], Dict] = {}
    for r in perf_rows:
        key = (r.get("encoder"), r.get("finding"))
        by_key.setdefault(key, {})[r.get("metric_name")] = r.get("value_raw")
    rows = []
    for (encoder, finding), metrics in by_key.items():
        if "ceiling_gap_target" not in metrics:
            continue
        rows.append({"encoder": encoder, "finding": finding,
                     "decodability": metrics.get("decodability"),
                     "decodability_nonlinear": metrics.get("decodability_nonlinear"),
                     "geometric_overlap": metrics.get("geometric_overlap"),
                     "leace_collateral": metrics.get("leace_collateral"),
                     "ceiling_gap": metrics.get("ceiling_gap_target")})
    return rows


def main_e3(global_config_path: str) -> Tuple[str, str]:
    cfg = read_config(global_config_path)["BiasOrigin"]
    pool_csv = cfg["cxr"]["pool_manifest_csv"]
    e2_perf = os.path.join(cfg["mitigation"]["results_e2_dir"], "results_performance_e2.csv")
    if not os.path.exists(e2_perf):
        raise FileNotFoundError(f"[e3] E2 performance CSV not found at {e2_perf}. Run E2 first.")
    ceil, unmit = _load_ceiling_target(e2_perf)

    seed = int(cfg["stats"]["boot_seed"])
    n_splits = int(cfg["mechanism"]["n_splits"])
    n_perm = int(cfg["mechanism"]["n_perm"])
    encoders = _e2_encoders(cfg)
    out_dir = cfg["mechanism"]["results_e3_dir"]

    done, perf_rows, stat_rows = R.load_partial(out_dir, "e3_entanglement")
    rows = _rows_from_perf(perf_rows)

    remaining = [e for e in encoders if e not in done]
    pbar = tqdm(remaining, desc="[e3] entanglement", unit="encoder")
    for encoder in pbar:
        try:
            X, man = load_encoder_pool_frame(global_config_path, encoder, "cxr_pool",
                                             pool_csv, columns=E3_MANIFEST_COLUMNS)
        except FileNotFoundError:
            continue
        prep = prepare_pool(X, man, ATTR)
        for finding in FINDINGS:
            ent = entanglement_for(X, man, finding, ATTR, seed, prep=prep)
            if ent is None:
                continue
            key = (encoder, finding)
            target = ceil.get(key, np.nan)
            ctx = {"experiment": "e3", "modality": "cxr", "dataset": "cxr_pool",
                   "encoder": encoder, "encoder_objective": encoder,
                   "attribute": ATTR, "finding": finding, "mitigation": "none"}
            _emit_point(perf_rows, ctx, "decodability", ent["decodability"], "lower", ent["n_test"])
            _emit_point(perf_rows, ctx, "decodability_nonlinear", ent["decodability_nonlinear"], "lower", ent["n_test"])
            _emit_point(perf_rows, ctx, "leace_collateral", ent["leace_collateral"], "lower", ent["n_test"])
            _emit_point(perf_rows, ctx, "geometric_overlap", ent["geometric_overlap"], "lower", ent["n_test"], as_percent=False)
            _emit_point(perf_rows, ctx, "disease_auroc", ent["disease_auroc"], "higher", ent["n_test"])
            if np.isfinite(target):
                _emit_point(perf_rows, ctx, "ceiling_gap_target", target, "lower", ent["n_test"])
                rows.append({"encoder": encoder, "finding": finding,
                             "decodability": ent["decodability"],
                             "decodability_nonlinear": ent["decodability_nonlinear"],
                             "geometric_overlap": ent["geometric_overlap"],
                             "leace_collateral": ent["leace_collateral"],
                             "ceiling_gap": target})
        done.add(encoder)
        R.save_partial(out_dir, "e3_entanglement", done, perf_rows, stat_rows)

    tab = pd.DataFrame(rows)
    if tab.empty:
        raise RuntimeError("[e3] no encoder x finding rows with a ceiling target.")

    y = tab["ceiling_gap"].values
    ctx_all = {"experiment": "e3", "modality": "cxr", "dataset": "cxr_pool",
               "attribute": ATTR, "mitigation": "battery"}

    for q, direction in [("decodability", "decodability"),
                         ("leace_collateral", "leace_collateral"),
                         ("geometric_overlap", "geometric_overlap")]:
        rho, p = spearman_with_perm(tab[q].values, y, n_perm=n_perm, seed=seed)
        R.report_spearman(stat_rows, {**ctx_all, "finding": f"vs_ceiling::{q}"},
                          rho, p, fdr_family="e3_univariate", n_units=len(tab))

    F_ent = tab[["leace_collateral", "geometric_overlap"]].values
    F_dec = tab[["decodability"]].values
    F_dec_nl = tab[["decodability_nonlinear"]].values
    for name, F in [("entanglement", F_ent), ("decodability_only", F_dec),
                    ("decodability_nonlinear_only", F_dec_nl)]:
        r2, p = permutation_r2_null(F, y, n_perm=n_perm, n_splits=n_splits, seed=seed)
        stat_rows.append(R.pack_statistic(
            estimate=r2, estimate_name=f"cross_fit_r2_{name}",
            test_name="r2_target_permutation", p_raw=p,
            fdr_family="e3_predictor", context={**ctx_all, "finding": name},
            n_units=len(tab)))

    r2_logo = logo_r2(F_ent, y, tab["encoder"].values, seed)
    rng = np.random.RandomState(seed)
    if np.isfinite(r2_logo):
        cnt = sum(1 for _ in range(n_perm)
                  if logo_r2(F_ent, rng.permutation(y), tab["encoder"].values, seed) >= r2_logo)
        p_logo = (cnt + 1) / (n_perm + 1)
    else:
        p_logo = float("nan")
    stat_rows.append(R.pack_statistic(
        estimate=r2_logo, estimate_name="logo_r2_entanglement",
        test_name="r2_target_permutation_logo", p_raw=p_logo,
        fdr_family="e3_predictor_logo", context={**ctx_all, "finding": "leave_one_encoder_out"},
        n_units=len(tab)))


    R.add_fdr(stat_rows, alpha=float(cfg["stats"]["fdr_alpha"]))
    os.makedirs(out_dir, exist_ok=True)
    perf_csv = os.path.join(out_dir, "results_performance_e3.csv")
    stat_csv = os.path.join(out_dir, "results_statistics_e3.csv")
    R.write_frame_atomic(R.perf_frame(perf_rows), perf_csv)
    R.write_frame_atomic(R.stat_frame(stat_rows), stat_csv)
    R.clear_partial(out_dir, "e3_entanglement")
    return perf_csv, stat_csv

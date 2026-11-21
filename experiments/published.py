"""
experiments/published.py
Created on November 20, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import json
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config.serde import read_config
from Inference import report_utils as R
from Inference.stats_utils import bh_fdr
from analysis.count_chance import simulate_null_gap


PUBLISHED_VERSION = 2
REQUIRED = ["claim_id", "paper", "modality", "dataset", "split", "task", "attribute", "condition",
            "subgroup", "metric", "reported_value", "reported_difference_direct", "denominator",
            "n_pos", "n_neg", "overall_value", "counts_source", "src", "note"]
KEY_COLS = ["paper", "dataset", "split", "task", "attribute", "condition", "metric"]
METRIC_MODE = {"auroc": "hanley", "tpr": "rate_tpr", "fpr": "rate_fpr"}


def load_claims(cfg, pcfg) -> pd.DataFrame:
    path = pcfg["claims_csv"]
    if not os.path.exists(path):
        raise R.MissingInput(f"[published] claims table absent: {path}; "
                             f"run main_build_published_claims(cfg) first")
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"[published] claims table is missing columns: {missing}")
    asked, have = set(pcfg["papers"]), set(df["paper"].astype(str))
    if sorted(asked - have):
        raise ValueError(f"[published] config lists papers with no claim rows: {sorted(asked - have)}")
    if sorted(have - asked):
        raise ValueError(f"[published] claims table has papers absent from config: {sorted(have - asked)}")
    bad = sorted(set(df["metric"].astype(str)) - set(METRIC_MODE))
    if bad:
        raise ValueError(f"[published] unsupported metric(s) {bad}; known: {sorted(METRIC_MODE)}")
    k = df.groupby(KEY_COLS, dropna=False)["claim_id"].nunique()
    if (k > 1).any():
        raise ValueError(f"[published] one key maps to several claim_ids: {k[k > 1].index.tolist()}")
    k2 = df.groupby("claim_id")[KEY_COLS].nunique(dropna=False).max(axis=1)
    if (k2 > 1).any():
        raise ValueError(f"[published] one claim_id spans several keys: {k2[k2 > 1].index.tolist()}")
    return df


def _draws(mode: str, overall: float, counts, n_sim: int, seed: int) -> np.ndarray:
    if mode == "bound":
        rng = np.random.default_rng(seed)
        d = np.vstack([np.clip(rng.normal(overall, np.sqrt(overall * (1 - overall) / min(p, n)), n_sim), 0, 1) for p, n in counts])
        return d.max(axis=0) - d.min(axis=0)
    return simulate_null_gap(overall, counts, n_sim, np.random.RandomState(seed), method=mode)


def audit_claim(sub: pd.DataFrame, n_sim: int, seed: int, min_sub: int, ratio: Dict[str, float]) -> Optional[Dict]:
    v = pd.to_numeric(sub["reported_value"], errors="coerce").to_numpy(float)
    npos = pd.to_numeric(sub["n_pos"], errors="coerce").to_numpy(float)
    nneg = pd.to_numeric(sub["n_neg"], errors="coerce").to_numpy(float)
    den = pd.to_numeric(sub["denominator"], errors="coerce").to_numpy(float)
    metric = str(sub["metric"].iloc[0])
    mode = METRIC_MODE[metric]
    direct = pd.to_numeric(sub["reported_difference_direct"], errors="coerce").dropna()
    ov = pd.to_numeric(sub["overall_value"], errors="coerce").dropna()
    ok = np.isfinite(den) & (den >= 1)
    if not len(direct):
        ok &= np.isfinite(v)
    if mode == "hanley":
        ok &= np.isfinite(npos) & np.isfinite(nneg) & (npos >= 1) & (nneg >= 1)
    if ok.sum() < min_sub:
        return None
    v, npos, nneg, den = v[ok], npos[ok], nneg[ok], den[ok]
    obs = float(direct.iloc[0]) if len(direct) else float(v.max() - v.min())
    overall = float(ov.iloc[0]) if len(ov) else float(np.average(v, weights=den))
    if mode == "rate_tpr":
        counts = [(int(x), 0) for x in den]
    elif mode == "rate_fpr":
        counts = [(0, int(x)) for x in den]
    else:
        counts = [(int(a), int(b)) for a, b in zip(npos, nneg)]
    out = {"metric": metric, "n_subgroups": int(ok.sum()), "smallest_denominator": int(den.min()), "overall_used": overall, "difference": obs}
    measure = {"auroc": "auroc", "tpr": "tpr", "fpr": "fpr"}[metric]
    forms = [("count", mode)] + ([("bound", "bound")] if mode == "hanley" else [])
    for name, m in forms:
        g = _draws(m, overall, counts, n_sim, seed)
        g = g[np.isfinite(g)]
        out.update({f"chance_{name}": float(g.mean()), f"p_{name}": float((1 + (g >= obs - 1e-12).sum()) / (g.size + 1))})
        if name == "count" and str(sub["modality"].iloc[0]) == "chest radiograph":
            ga = g * ratio[measure]
            out.update({"chance_patient_adjusted": float(ga.mean()), "p_patient_adjusted": float((1 + (ga >= obs - 1e-12).sum()) / (ga.size + 1))})
    return out


def measured_ratio(cfg: Dict) -> Dict[str, float]:
    path = os.path.join(cfg["subgroups"]["results_chance_dir"], "chance_differences.csv")
    if not os.path.exists(path):
        raise R.MissingInput(f"[published] {path} does not exist; run main_chance_merge.")
    d = pd.read_csv(path, low_memory=False)
    d = d[(d["pool"] == "cxr_pool") & (d["stratum"] == "all") & d["attribute"].isin(["race_grp", "age_grp", "sex_grp", "insurance_grp"])]
    return {m: float((g["chance"] / g["formula_chance"]).median()) for m, g in d.groupby("measure")}


def main_published(global_config_path: str) -> None:
    cfg = read_config(global_config_path)["BiasOrigin"]
    pcfg = cfg["published"]
    out_dir = cfg["subgroups"]["results_chance_dir"]
    out_c, out_s = os.path.join(out_dir, "published_comparisons.csv"), os.path.join(out_dir, "published_studies.csv")
    ratio = measured_ratio(cfg)
    claims = load_claims(cfg, pcfg)
    sig = {"version": PUBLISHED_VERSION, "ratio": ratio, "claims": [int(os.stat(pcfg["claims_csv"]).st_size), int(os.stat(pcfg["claims_csv"]).st_mtime_ns)],
           "n_sim": int(pcfg["n_sim"]), "seed": int(pcfg["seed"])}
    try:
        with open(out_c + ".params.json") as fh:
            if json.load(fh) == json.loads(json.dumps(sig)) and os.path.exists(out_c) and os.path.exists(out_s):
                return
    except (OSError, ValueError):
        pass
    rows = []
    for cid, sub in claims.groupby("claim_id", sort=False):
        res = audit_claim(sub, int(pcfg["n_sim"]), int(pcfg["seed"]), int(pcfg["min_subgroups"]), ratio)
        head = sub.iloc[0]
        base = {"claim_id": cid, "paper": head["paper"], "modality": head["modality"], "dataset": head["dataset"], "task": head["task"],
                "attribute": head["attribute"], "condition": head["condition"], "counts_source": head.get("counts_source"),
                "clinician": str(head["condition"]) == "dermatologist_ensemble"}
        rows.append({**base, "auditable": res is not None, **(res or {})})
    df = pd.DataFrame(rows)
    aud = df["auditable"]
    df["family"] = np.where(df["metric"] == "auroc", "auroc", "rate")
    for col in ("p_count", "p_bound", "p_patient_adjusted"):
        df[col.replace("p_", "p_fdr_")] = np.nan
        for _, idx in df[aud & df[col].notna()].groupby("family").groups.items():
            df.loc[idx, col.replace("p_", "p_fdr_")] = bh_fdr(df.loc[idx, col].values)
    alpha = float(cfg["stats"]["fdr_alpha"])
    for form in ("count", "bound", "patient_adjusted"):
        df[f"larger_{form}"] = np.where(df[f"p_fdr_{form}"].notna(), df[f"p_fdr_{form}"] < alpha, np.nan)
    a = df[aud]

    def n_larger(x: pd.Series) -> float:
        x = pd.to_numeric(x, errors="coerce")
        return float(x.sum()) if x.notna().any() else np.nan
    study = a.groupby(["paper", "modality"]).agg(n_comparisons=("claim_id", "size"), n_rate=("family", lambda f: int((f == "rate").sum())),
                                                 n_auroc=("family", lambda f: int((f == "auroc").sum())),
                                                 n_clinician=("clinician", "sum"), larger_count=("larger_count", n_larger),
                                                 larger_bound=("larger_bound", n_larger), larger_patient_adjusted=("larger_patient_adjusted", n_larger),
                                                 n_datasets=("dataset", "nunique")).reset_index()
    if os.path.exists(out_c + ".params.json"):
        os.remove(out_c + ".params.json")
    R.write_frame_atomic(df, out_c)
    R.write_frame_atomic(study, out_s)
    with open(out_c + ".params.json.tmp", "w") as fh:
        json.dump(sig, fh)
    os.replace(out_c + ".params.json.tmp", out_c + ".params.json")

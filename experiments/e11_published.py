"""
experiments/e11_published.py
Created on August 12, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.serde import read_config
from Inference import report_utils as R
from Inference import stats_utils as S
from experiments.e8_gapnull import simulate_null_gap

REQUIRED = ["claim_id", "paper", "modality", "dataset", "split", "task", "attribute", "condition",
            "subgroup", "metric", "reported_value", "reported_difference_direct", "denominator",
            "n_pos", "n_neg", "overall_value", "counts_source", "src", "note"]

METRIC_MODE = {"auroc": "hanley", "tpr": "rate_tpr", "fpr": "rate_fpr"}

KEY_COLS = ["paper", "dataset", "split", "task", "attribute", "condition", "metric"]

META_COLS = ["paper", "modality", "dataset", "split", "task", "attribute", "condition", "metric",
             "counts_source", "src", "note"]


def _cfg(cfg) -> Dict:
    return cfg["BiasOrigin"]["published"]


def _audit_path(out_dir: str, tag: str) -> str:
    return os.path.join(out_dir, f"{tag}__audit.json")


def _load_audit(out_dir: str, tag: str) -> Dict[str, Dict]:
    p = _audit_path(out_dir, tag)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "rb") as f:
            return json.loads(f.read().decode("utf-8", errors="strict"))
    except Exception as e:
        print(f"[e11] audit store unreadable ({e}); rebuilding it", flush=True)
        return {}


def _save_audit(out_dir: str, tag: str, audit: Dict[str, Dict]) -> None:
    p = _audit_path(out_dir, tag)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="ascii") as f:
        f.write(json.dumps(audit, ensure_ascii=True))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def load_claims(cfg, pcfg) -> pd.DataFrame:
    path = pcfg["claims_csv"]
    if not os.path.exists(path):
        raise R.MissingInput(f"[e11] claims table absent: {path}; "
                             f"run main_build_published_claims(cfg) first")
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"[e11] claims table is missing columns: {missing}")
    asked, have = set(pcfg["papers"]), set(df["paper"].astype(str))
    if sorted(asked - have):
        raise ValueError(f"[e11] config lists papers with no claim rows: {sorted(asked - have)}")
    if sorted(have - asked):
        raise ValueError(f"[e11] claims table has papers absent from config: {sorted(have - asked)}")
    bad = sorted(set(df["metric"].astype(str)) - set(METRIC_MODE))
    if bad:
        raise ValueError(f"[e11] unsupported metric(s) {bad}; known: {sorted(METRIC_MODE)}")
    k = df.groupby(KEY_COLS, dropna=False)["claim_id"].nunique()
    if (k > 1).any():
        raise ValueError(f"[e11] one key maps to several claim_ids: {k[k > 1].index.tolist()}")
    k2 = df.groupby("claim_id")[KEY_COLS].nunique(dropna=False).max(axis=1)
    if (k2 > 1).any():
        raise ValueError(f"[e11] one claim_id spans several keys: {k2[k2 > 1].index.tolist()}")
    return df


def _audit_one(sub: pd.DataFrame, n_sim: int, seed: int, min_sub: int) -> Optional[Dict]:
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

    if len(direct):
        obs = float(direct.iloc[0])
        if not len(ov):
            raise ValueError(f"[e11] {sub['claim_id'].iloc[0]}: a difference given directly needs "
                             f"an overall_value, since it cannot be pooled from subgroup values")
        overall = float(ov.iloc[0])
    else:
        obs = float(v.max() - v.min())
        overall = float(ov.iloc[0]) if len(ov) else float(np.average(v, weights=den))

    if mode == "rate_tpr":
        counts = [(int(x), 0) for x in den]
    elif mode == "rate_fpr":
        counts = [(0, int(x)) for x in den]
    else:
        counts = [(int(a), int(b)) for a, b in zip(npos, nneg)]

    G = simulate_null_gap(overall, counts, n_sim, np.random.RandomState(seed), method=mode)
    G = G[np.isfinite(G)]
    if G.size == 0:
        return None
    r = float(G.mean())
    return {
        "n_subgroups": int(ok.sum()),
        "smallest_denominator": int(den.min()),
        "largest_denominator": int(den.max()),
        "overall_used": overall,
        "reported_difference": obs,
        "reference": r,
        "reference_sd": float(G.std(ddof=1)) if G.size > 1 else np.nan,
        "reference_ci_low": float(np.percentile(G, 2.5)),
        "reference_ci_high": float(np.percentile(G, 97.5)),
        "excess_over_reference": obs - r,
        "reference_share_pct": (r / obs * 100.0) if obs > 0 else np.nan,
        "p_raw": (1.0 + float((G >= obs).sum())) / (G.size + 1.0),
        "n_sim": int(G.size),
        "subgroups": "; ".join(str(x) for x in sub.loc[ok, "subgroup"]),
        "reported_per_subgroup": ("; ".join(f"{float(x):.4f}" for x in v)
                                  if np.isfinite(v).all() else "difference printed, values in a figure"),
        "denominator_per_subgroup": "; ".join(str(int(x)) for x in den),
    }


def run_e11(cfg, cfg_path, out_dir, tag):
    pcfg = _cfg(cfg)
    df = load_claims(cfg, pcfg)
    n_sim, seed = int(pcfg["n_sim"]), int(pcfg["seed"])
    min_sub = int(pcfg["min_subgroups"])

    done, perf_rows, stat_rows = R.load_partial(out_dir, tag)
    audit = _load_audit(out_dir, tag)
    ids = [c for c in df["claim_id"].astype(str).unique() if c not in done]
    print(f"[e11] {df['claim_id'].nunique()} claims, {len(ids)} to run", flush=True)

    for cid in tqdm(ids, desc="[e11] claims", ncols=100):
        sub = df[df["claim_id"].astype(str) == cid]
        head = sub.iloc[0]
        res = _audit_one(sub, n_sim, seed, min_sub)
        rec = {c: (None if pd.isna(head[c]) else head[c]) for c in META_COLS}
        rec["claim_id"] = cid
        if res is None:
            rec["auditable"] = False
            print(f"[e11] NOT AUDITABLE {cid} ({head['paper']}): "
                  f"fewer than {min_sub} subgroups with a value and a denominator", flush=True)
        else:
            rec["auditable"] = True
            rec.update(res)
            ctx = {"experiment": "e11", "modality": str(head["modality"]), "dataset": str(head["dataset"]),
                   "eval_dataset": str(head["dataset"]), "encoder": str(head["paper"]),
                   "attribute": str(head["attribute"]), "finding": str(head["task"]),
                   "mitigation": str(head["condition"]), "data_composition": str(head["split"]),
                   "operating_point": str(head["metric"])}
            for name in ("reported_difference", "reference", "excess_over_reference"):
                is_ref = name == "reference"
                R.report_metric(perf_rows, ctx, f"{head['metric']}__{name}",
                                {"point": res[name],
                                 "std": res["reference_sd"] if is_ref else np.nan,
                                 "ci_lower": res["reference_ci_low"] if is_ref else np.nan,
                                 "ci_upper": res["reference_ci_high"] if is_ref else np.nan,
                                 "n": res["n_sim"]}, "lower")
            stat_rows.append(R.pack_statistic(
                estimate=res["excess_over_reference"], estimate_name="excess_over_reference",
                test_name="fair_model_simulation", p_raw=res["p_raw"],
                fdr_family=f"e11_exceedance::{head['metric']}",
                context=ctx, n_units=res["n_subgroups"]))
        audit[cid] = rec
        done.add(cid)
        R.save_partial(out_dir, tag, done, perf_rows, stat_rows)
        _save_audit(out_dir, tag, audit)

    return perf_rows, stat_rows, audit


def main_e11(global_config_path: str, force: bool = False):
    cfg = read_config(global_config_path)
    pcfg = _cfg(cfg)
    out_dir = pcfg["results_e11_dir"]
    os.makedirs(out_dir, exist_ok=True)
    tag = "e11_published"
    if not force and R.shard_exists(out_dir, tag):
        print(f"[e11] shard present, skipping: {tag}", flush=True)
        return
    if force:
        for p in (_audit_path(out_dir, tag),):
            if os.path.exists(p):
                os.remove(p)
        R.clear_partial(out_dir, tag)
    stale = float(cfg["BiasOrigin"]["stats"].get("claim_stale_after_s", 21600))
    if not force and not R.claim_unit(out_dir, tag, stale):
        print(f"[e11] {tag} claimed by another job, skipping", flush=True)
        return
    try:
        perf_rows, stat_rows, _ = run_e11(cfg, global_config_path, out_dir, tag)
        R.write_shard(out_dir, tag, perf_rows, stat_rows)
        R.clear_partial(out_dir, tag)
    except R.MissingInput as e:
        print(f"[e11] SKIP: {e}", flush=True)
    finally:
        R.release_claim(out_dir, tag)


def main_e11_merge(global_config_path: str) -> str:
    cfg = read_config(global_config_path)
    pcfg = _cfg(cfg)
    out_dir, tag = pcfg["results_e11_dir"], "e11_published"
    alpha = float(cfg["BiasOrigin"]["stats"]["fdr_alpha"])
    audit = _load_audit(out_dir, tag)
    if not audit:
        raise RuntimeError("[e11] no audited claims; run main_e11 first")
    want = set(load_claims(cfg, pcfg)["claim_id"].astype(str))
    absent = sorted(want - set(audit))
    if absent:
        raise RuntimeError(f"[e11] audit store covers {len(audit)} of {len(want)} claims; "
                           f"missing {absent}. Re-run main_e11(cfg, force=True).")
    stray = sorted(set(audit) - want)
    if stray:
        raise RuntimeError(f"[e11] audit store holds claims the table no longer has: {stray}. "
                           f"Re-run main_e11(cfg, force=True).")

    out = pd.DataFrame(list(audit.values()))
    out["p_fdr"], out["significant_fdr05"] = np.nan, np.nan
    aud = out["auditable"].fillna(False).to_numpy(bool)
    for metric, idx in out[aud].groupby("metric").groups.items():
        p = out.loc[idx, "p_raw"].to_numpy(float)
        q = S.bh_fdr(p)
        out.loc[idx, "p_fdr"] = q
        out.loc[idx, "significant_fdr05"] = q < alpha

    order = ["claim_id", "paper", "modality", "dataset", "split", "task", "attribute", "condition",
             "metric", "auditable", "n_subgroups", "subgroups", "reported_per_subgroup",
             "denominator_per_subgroup", "smallest_denominator", "largest_denominator",
             "overall_used", "reported_difference", "reference", "reference_sd",
             "reference_ci_low", "reference_ci_high", "excess_over_reference",
             "reference_share_pct", "p_raw", "p_fdr", "significant_fdr05", "n_sim",
             "counts_source", "src", "note"]
    out = out.reindex(columns=[c for c in order if c in out.columns])
    out = out.sort_values(["auditable", "paper", "claim_id"], ascending=[False, True, True])
    R.write_frame_atomic(out, pcfg["out_csv"])
    print(f"[e11] {int(aud.sum())} audited and {int((~aud).sum())} not auditable "
          f"-> {pcfg['out_csv']}", flush=True)
    return pcfg["out_csv"]

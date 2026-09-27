"""
merge/build_final_tables.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import json
import os
import shutil
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from config.serde import read_config
from Inference import report_utils as R


def _read_many(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if os.path.exists(p):
            try:
                frames.append(pd.read_csv(p, float_precision="round_trip"))
            except Exception as e:
                pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _arm_dirs(cfg) -> Dict[str, str]:
    return {
        "e1": cfg["controlled"]["results_e1_dir"],
        "e2": cfg["mitigation"]["results_e2_dir"],
        "e3": cfg["mechanism"]["results_e3_dir"],
        "e4": cfg["generalization"]["results_e4_dir"],
        "e5": cfg["generalization"]["results_e5_dir"],
        "e6": cfg["finetuning"]["results_e6_dir"],
        "e7": cfg["theory"]["results_e7_dir"],
        "e8": cfg["gap_null"]["results_e8_dir"],
        "e9": cfg["positive_control"]["results_e9_dir"],
        "e10": cfg["acquisition"]["results_e10_dir"],
    }


def _collect(cfg) -> (pd.DataFrame, pd.DataFrame):
    dirs = _arm_dirs(cfg)
    perf_paths = [os.path.join(d, f"results_performance_{a}.csv") for a, d in dirs.items()]
    stat_paths = [os.path.join(d, f"results_statistics_{a}.csv") for a, d in dirs.items()]
    perf = _read_many(perf_paths)
    stat = _read_many(stat_paths)
    if not perf.empty:
        perf = perf.reindex(columns=R.PERF_COLUMNS)
    if not stat.empty:
        stat = stat.reindex(columns=R.STAT_COLUMNS)
    return perf, stat


def _global_fdr(stat: pd.DataFrame, alpha: float) -> pd.DataFrame:
    if stat.empty:
        return stat
    recs = stat.to_dict("records")
    R.add_fdr(recs, alpha=alpha)
    return pd.DataFrame(recs, columns=R.STAT_COLUMNS)


def _frontier_curves(perf: pd.DataFrame) -> pd.DataFrame:
    if perf.empty:
        return pd.DataFrame()
    b = perf[perf["experiment"].isin(["e2", "e4", "e9"])]
    rows = []
    keys = ["experiment", "modality", "dataset", "encoder", "attribute", "finding",
            "mitigation", "data_composition"]
    for key, grp in b.groupby(keys, dropna=False):
        auroc = grp.loc[grp["metric_name"] == "auroc_overall", "value_raw"]
        gap = grp.loc[grp["metric_name"] == "auroc_gap", "value_raw"]
        if auroc.empty or gap.empty:
            continue
        d = dict(zip(keys, key))
        d["disease_auroc"] = float(auroc.iloc[0])
        d["gap"] = float(gap.iloc[0])
        rows.append(d)
    return pd.DataFrame(rows)


def _run_metadata(cfg, global_config_path) -> pd.DataFrame:
    rows = []
    for name, spec in cfg["encoder_panel"]["image"].items():
        rows.append({"kind": "encoder", "name": name,
                     "info": json.dumps({k: spec.get(k) for k in
                                         ("type", "objective", "dim", "modality")})})
    try:
        from controlled.train_encoder import list_controlled_runs
        for rid in list_controlled_runs(global_config_path):
            rows.append({"kind": "controlled_run", "name": rid, "info": ""})
    except Exception as e:
        rows.append({"kind": "controlled_run", "name": "(unavailable)", "info": str(e)})
    for pool, pc in cfg["embeddings"]["pools"].items():
        rows.append({"kind": "pool", "name": pool, "info": pc.get("manifest", "")})
    rows.append({"kind": "config", "name": "stats",
                 "info": json.dumps({"n_boot": cfg["stats"]["n_boot"],
                                     "boot_seed": cfg["stats"]["boot_seed"],
                                     "operating_sensitivity": cfg["stats"]["operating_sensitivity"],
                                     "fdr_alpha": cfg["stats"]["fdr_alpha"]})})
    rows.append({"kind": "config", "name": "generated_at",
                 "info": time.strftime("%Y-%m-%d %H:%M:%S")})
    return pd.DataFrame(rows, columns=["kind", "name", "info"])


def validate_cross_consistency(perf: pd.DataFrame, stat: pd.DataFrame, cfg) -> None:
    problems = []

    if not perf.empty:
        v = pd.to_numeric(perf["value"], errors="coerce")
        is_diff = perf["metric_name"].astype(str).str.endswith("__diff")
        bad_level = perf[(v.notna()) & (~is_diff) & ((v < -0.001) | (v > 100.001))]
        bad_diff = perf[(v.notna()) & (is_diff) & ((v < -100.001) | (v > 100.001))]
        if len(bad_level):
            problems.append(f"{len(bad_level)} performance level rows with percent "
                            f"value outside [0,100]")
        if len(bad_diff):
            problems.append(f"{len(bad_diff)} performance difference rows with "
                            f"percentage-point value outside [-100,100]")

    if not stat.empty:
        forbidden = [c for c in ("value", "std", "ci_low", "ci_high") if c in stat.columns]
        if forbidden:
            problems.append(f"statistics table has forbidden performance columns: {forbidden}")
        if stat["fdr_family"].isna().any():
            problems.append("statistics rows with missing fdr_family")
        pr = pd.to_numeric(stat["p_raw"], errors="coerce")
        pf = pd.to_numeric(stat["p_fdr"], errors="coerce")
        if ((pr < -1e-9) | (pr > 1 + 1e-9)).any():
            problems.append("p_raw outside [0,1]")
        both = pr.notna() & pf.notna()
        if (pf[both] < pr[both] - 1e-6).any():
            problems.append("adjusted p_fdr smaller than raw p_raw (BH violation)")

    if not perf.empty:
        diffs = perf[perf["metric_name"].astype(str).str.endswith("__diff")]
        if len(diffs):
            has_diff_stat = (not stat.empty) and \
                stat["estimate_name"].astype(str).str.endswith("_difference").any()
            if not has_diff_stat:
                problems.append(f"{len(diffs)} difference-effect rows but no matching "
                                f"statistics rows")

    if not perf.empty:
        known = set(cfg["encoder_panel"]["image"].keys()) | {"synthetic"}
        seen = set(perf["encoder"].dropna().astype(str))
        unknown = {e for e in seen if e not in known}

    if problems:
        raise ValueError("[merge] cross-consistency FAILED:\n  - " + "\n  - ".join(problems))


def main_build_final_tables(global_config_path: str) -> str:
    cfg = read_config(global_config_path)["BiasOrigin"]
    ft = cfg["final_tables"]
    os.makedirs(ft["dir"], exist_ok=True)
    alpha = float(cfg["stats"]["fdr_alpha"])

    perf, stat = _collect(cfg)
    stat = _global_fdr(stat, alpha)
    validate_cross_consistency(perf, stat, cfg)

    R.write_frame_atomic(perf, ft["performance_csv"])
    R.write_frame_atomic(stat, ft["statistics_csv"])
    R.write_frame_atomic(_frontier_curves(perf), ft["frontier_curves_csv"])
    R.write_frame_atomic(_run_metadata(cfg, global_config_path), ft["run_metadata_csv"])

    src_counts = cfg.get("subgroup_counts", {}).get("out_csv")
    if src_counts and os.path.exists(src_counts):
        tmp_counts = ft["subgroup_counts_csv"] + ".tmp"
        shutil.copyfile(src_counts, tmp_counts)
        os.replace(tmp_counts, ft["subgroup_counts_csv"])

    return ft["dir"]

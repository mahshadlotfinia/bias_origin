"""
experiments/summary_tables.py
Created on November 20, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import glob
import json
import os
from typing import Dict

import numpy as np
import pandas as pd

from config.serde import read_config
from Inference import report_utils as R


TABLES_VERSION = 6


def _q(x: pd.Series, q: float) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return float(x.quantile(q)) if len(x) else np.nan


def _panel(encoder: pd.Series) -> np.ndarray:
    return np.select([encoder.str.startswith("controlled/"), encoder == "random_init_vit_s"], ["controlled", "random_init"], "public")


def chance_table(d: pd.DataFrame) -> pd.DataFrame:
    d = d.assign(panel=_panel(d["encoder"]))
    rows = []
    for key, g in d.groupby(["pool", "panel", "attribute", "stratum", "measure"]):
        rows.append(dict(zip(["pool", "panel", "attribute", "stratum", "measure"], key), n=len(g), n_encoders=g["encoder"].nunique(),
                         difference_median=_q(g["difference"], .5), difference_q1=_q(g["difference"], .25), difference_q3=_q(g["difference"], .75),
                         chance_median=_q(g["chance"], .5), chance_q1=_q(g["chance"], .25), chance_q3=_q(g["chance"], .75),
                         count_chance_median=_q(g["formula_chance"], .5), larger_than_chance=int(g["larger_than_chance"].sum()),
                         larger_than_chance_pct=float(100 * g["larger_than_chance"].mean()),
                         larger_count_form=int((g["formula_p_fdr"] < 0.05).sum()), chance_share_median_pct=float(100 * (g["chance"] / g["difference"]).median())))
    return pd.DataFrame(rows)


def strata_table(s: pd.DataFrame) -> pd.DataFrame:
    s = s.assign(panel=_panel(s["encoder"]))
    s = s.assign(detectable=s["chance_within"] + 1.645 * s["chance_within_std"])
    rows = []
    for key, g in s.groupby(["pool", "panel", "by", "attribute", "measure"]):
        rows.append(dict(zip(["pool", "panel", "by", "attribute", "measure"], key), n=len(g), n_levels_median=_q(g["n_levels"], .5),
                         difference_all_median=_q(g["difference_all"], .5), difference_within_median=_q(g["difference_within"], .5),
                         difference_within_q1=_q(g["difference_within"], .25), difference_within_q3=_q(g["difference_within"], .75),
                         chance_within_median=_q(g["chance_within"], .5), detectable_median=_q(g["detectable"], .5),
                         larger_than_chance=int(g["larger_than_chance"].sum()), larger_than_chance_pct=float(100 * g["larger_than_chance"].mean())))
    return pd.DataFrame(rows)


def threshold_table(groups: pd.DataFrame) -> pd.DataFrame:
    g = groups[groups["fpr"].notna() & groups["tpr"].notna()].assign(panel=lambda d: _panel(d["encoder"]))
    rows = []
    for key, h in g.groupby(["pool", "panel", "attribute", "stratum", "encoder", "finding"], sort=False):
        if len(h) < 2:
            continue
        fpr, tpr = h["fpr"].to_numpy(float), h["tpr"].to_numpy(float)
        rho = pd.Series(fpr).corr(pd.Series(tpr), method="spearman") if len(h) > 2 else float(np.sign((fpr[0] - fpr[1]) * (tpr[0] - tpr[1])))
        rows.append(dict(zip(["pool", "panel", "attribute", "stratum", "encoder", "finding"], key), k=len(h), same_lowest=bool(np.argmin(fpr) == np.argmin(tpr)), rho=rho))
    r = pd.DataFrame(rows)
    out = []
    for key, h in r.groupby(["pool", "panel", "attribute", "stratum"]):
        out.append(dict(zip(["pool", "panel", "attribute", "stratum"], key), n=len(h), k_median=_q(h["k"], .5), n_same_lowest=int(h["same_lowest"].sum()),
                        expected_same_lowest=float((1.0 / h["k"]).sum()), n_rho_positive=int((h["rho"] > 0).sum()), rho_median=_q(h["rho"], .5)))
    return pd.DataFrame(out)


def group_rates_table(groups: pd.DataFrame) -> pd.DataFrame:
    keys = ["pool", "panel", "attribute", "stratum"]
    g = groups[groups["fpr"].notna() & groups["tpr"].notna()].assign(panel=lambda d: _panel(d["encoder"])).reset_index(drop=True)
    rows = []
    for key, h in g.groupby(keys, sort=False):
        h = h[h.groupby(["encoder", "finding"])["group"].transform("size") >= 2]
        if h.empty:
            continue
        per_model = h.groupby(["encoder", "finding"])
        lo_fpr = h.loc[per_model["fpr"].idxmin(), "group"].value_counts()
        hi_fpr = h.loc[per_model["fpr"].idxmax(), "group"].value_counts()
        lo_tpr = h.loc[per_model["tpr"].idxmin(), "group"].value_counts()
        med = h.groupby("group")[["fpr", "tpr", "n_patients"]].median()
        for grp in sorted(h["group"].astype(str).unique()):
            rows.append({**dict(zip(keys, key)), "group": grp, "n_models": per_model.ngroups, "lowest_fpr": int(lo_fpr.get(grp, 0)),
                         "highest_fpr": int(hi_fpr.get(grp, 0)), "lowest_sensitivity": int(lo_tpr.get(grp, 0)),
                         "fpr_median": float(med.loc[grp, "fpr"]), "sensitivity_median": float(med.loc[grp, "tpr"]), "patients_median": float(med.loc[grp, "n_patients"])})
    return pd.DataFrame(rows)


def group_sizes_table(cfg: Dict) -> pd.DataFrame:
    from experiments.chance import pool_cases
    rows = []
    for pool in cfg["subgroups"]["attributes"]:
        cases = pool_cases(cfg, pool)
        test = cases["cluster"] >= 0
        for att, stratum, g in cases["analyses"]:
            if att in ("view", "site") or not (stratum == "all" or stratum.startswith("site=")):
                continue
            for grp in sorted({v for v in g[test] if v is not None}):
                m = test & (g == grp)
                rows.append({"pool": pool, "attribute": att, "stratum": stratum, "group": grp,
                             "n_patients": int(np.unique(cases["cluster"][m]).size), "n_images": int(m.sum())})
    return pd.DataFrame(rows)


def contrast_table(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in c.groupby(["contrast", "attribute", "statistic"]):
        rows.append(dict(zip(["contrast", "attribute", "statistic"], key), n=len(g), delta_median=_q(g["delta"], .5), delta_q1=_q(g["delta"], .25),
                         delta_q3=_q(g["delta"], .75), n_significant=int(g["significant"].sum()), n_significant_positive=int((g["significant"] & (g["delta"] > 0)).sum()),
                         n_significant_negative=int((g["significant"] & (g["delta"] < 0)).sum())))
    return pd.DataFrame(rows)


def published_table(c: pd.DataFrame) -> pd.DataFrame:
    c = c.assign(family=np.where(c["metric"] == "auroc", "auroc", "rate"))
    rows = []
    for key, g in c.groupby(["clinician", "family"]):
        row = dict(zip(["clinician", "family"], key), n=len(g), n_studies=g["paper"].nunique(), difference_median=_q(g["difference"], .5), chance_count_median=_q(g["chance_count"], .5))
        for form in ("count", "bound", "patient_adjusted"):
            col = g[f"larger_{form}"]
            row[f"n_{form}"] = int(col.notna().sum())
            row[f"larger_{form}"] = int(pd.to_numeric(col, errors="coerce").fillna(0).astype(bool).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main_summary_tables(global_config_path: str) -> None:
    cfg = read_config(global_config_path)["BiasOrigin"]
    src = cfg["subgroups"]["results_chance_dir"]
    need = {"differences": "chance_differences.csv", "groups": "chance_groups.csv", "overall": "chance_overall.csv", "strata": "chance_strata.csv",
            "strata_joint": "chance_strata_joint.csv", "contrasts": "contrasts.csv", "pub_c": "published_comparisons.csv", "pub_s": "published_studies.csv",
            "val_real": os.path.join("validation", "shuffled_groups_summary.csv"), "val_level": os.path.join("validation", "shuffled_groups_level_summary.csv"),
            "val_sim": os.path.join("validation", "simulation.csv")}
    missing = [f for f in need.values() if not os.path.exists(os.path.join(src, f))]
    if missing:
        raise R.MissingInput(f"[summary_tables] inputs not built yet: {missing}")
    joint_units = sorted(glob.glob(os.path.join(src, "units", "*.joint.csv")))
    if not joint_units:
        raise R.MissingInput(f"[summary_tables] no joint units in {os.path.join(src, 'units')}; run main_chance_joint.")
    out_dir = cfg["subgroups"]["final_tables_dir"]
    os.makedirs(out_dir, exist_ok=True)
    sig = {"version": TABLES_VERSION, **{f: [int(os.stat(os.path.join(src, f)).st_size), int(os.stat(os.path.join(src, f)).st_mtime_ns)] for f in need.values()},
           "joint_units": [[os.path.basename(f), int(os.stat(f).st_size), int(os.stat(f).st_mtime_ns)] for f in joint_units],
           "manifests": {p: [int(os.stat(cfg["embeddings"]["pools"][p]["manifest"]).st_size), int(os.stat(cfg["embeddings"]["pools"][p]["manifest"]).st_mtime_ns)] for p in cfg["subgroups"]["attributes"]}}
    record = os.path.join(out_dir, "tables.params.json")

    def current() -> bool:
        try:
            with open(record) as fh:
                return json.load(fh) == json.loads(json.dumps(sig))
        except (OSError, ValueError):
            return False
    if not R.claim_or_wait(out_dir, "summary_tables", current):
        return
    try:
        t: Dict[str, pd.DataFrame] = {k: pd.read_csv(os.path.join(src, f), low_memory=False) for k, f in need.items()}
        tables = {"T1_chance": chance_table(t["differences"]), "T1b_chance_within_strata": strata_table(t["strata"]),
                  "T1c_chance_within_joint_strata": strata_table(t["strata_joint"]), "T1d_threshold_tradeoff": threshold_table(t["groups"]),
                  "T2_published_studies": t["pub_s"], "T2b_published_by_metric": published_table(t["pub_c"]),
                  "T3_validation_shuffled": t["val_real"], "T3_validation_level": t["val_level"], "T3_validation_simulation": t["val_sim"],
                  "T4_contrasts": contrast_table(t["contrasts"])}
        joint_groups = pd.concat([pd.read_csv(f, low_memory=False) for f in joint_units], ignore_index=True)
        joint_groups = joint_groups[joint_groups["table"] == "group"].drop(columns=["table"])
        tables["T1e_group_rates"] = group_rates_table(pd.concat([t["groups"], joint_groups], ignore_index=True))
        tables["T5_group_sizes"] = group_sizes_table(cfg)
        if os.path.exists(record):
            os.remove(record)
        for name, df in tables.items():
            R.write_frame_atomic(df, os.path.join(out_dir, f"{name}.csv"))
        tmp = f"{record}.{os.getpid()}.tmp"
        with open(tmp, "w") as fh:
            json.dump(sig, fh)
        os.replace(tmp, record)
    finally:
        R.release_claim(out_dir, "summary_tables")

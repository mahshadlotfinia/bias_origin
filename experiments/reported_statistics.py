"""
experiments/reported_statistics.py
Created on November 21, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_utils import read_csv_defensively
from Inference.stats_utils import bh_fdr

PUBLIC_CHEST = ("rad_dino", "biomedclip_image", "txrv_densenet", "dinov2_large", "dinov3_s", "dinov3_b", "dinov3_l",
                "clip_vitl14", "siglip2_large")
OVERLAP_ENCODERS = ("rad_dino", "txrv_densenet")
DEMOGRAPHIC = ("race_grp", "sex_grp", "age_grp", "insurance_grp")
ACQUISITION = ("view", "site")
CRITICAL_Z = 1.645
ALPHA = 0.05
DATA_CONDITIONS = ("natural", "balanced", "scrubbed", "amplified")
NO_PATIENT_IDENTIFIER = ("vindr_cxr", "vindr_pcxr", "ddi", "fitz17k", "isic2019")
OBJECTIVES = ("image_text", "ssl", "supervised")


def panel_of(pool: str, encoder: str) -> str:
    if encoder.startswith("controlled/") or encoder == "random_init_vit_s":
        return "replication"
    return "public" if (pool != "cxr_pool" or encoder in PUBLIC_CHEST) else "other"


def condition_of(encoder: str) -> str:
    if encoder == "random_init_vit_s":
        return "random_init"
    m = re.search(r"__(%s)__seed" % "|".join(DATA_CONDITIONS), encoder)
    return m.group(1) if m else "public"


def objective_of(encoder: str) -> str:
    m = re.search(r"__(%s)__vit_" % "|".join(OBJECTIVES), encoder)
    return m.group(1) if m else ""


def _row(section: str, statistic: str, value, **keys) -> Dict:
    out = {"section": section, "pool": "", "panel": "", "attribute": "", "stratum": "", "measure": "", "group": "",
           "statistic": statistic, "value": value}
    out.update({k: v for k, v in keys.items()})
    return out


def _tested(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[frame["p_raw"].notna()].copy()
    frame["panel"] = [panel_of(p, e) for p, e in zip(frame["pool"], frame["encoder"])]
    return frame


def difference_summaries(diff: pd.DataFrame) -> List[Dict]:
    rows = []
    diff = diff.assign(critical=diff["chance"] + CRITICAL_Z * diff["chance_std"])
    for (pool, panel, att, stratum, measure), g in diff.groupby(["pool", "panel", "attribute", "stratum", "measure"]):
        keys = dict(pool=pool, panel=panel, attribute=att, stratum=stratum, measure=measure)
        rows += [_row("differences", "n_models", len(g), **keys),
                 _row("differences", "n_significant", int((g["p_fdr"] < ALPHA).sum()), **keys),
                 _row("differences", "difference_median", float(g["difference"].median()), **keys),
                 _row("differences", "chance_median", float(g["chance"].median()), **keys),
                 _row("differences", "critical_value_median", float(g["critical"].median()), **keys)]
        sig = g[g["p_fdr"] < ALPHA]
        if 0 < len(sig) < len(g):
            for finding, gf in sig.groupby("finding"):
                rows += [_row("significant_findings", "n_significant", len(gf), group=finding, **keys),
                         _row("significant_findings", "difference_min", float(gf["difference"].min()), group=finding, **keys),
                         _row("significant_findings", "difference_max", float(gf["difference"].max()), group=finding, **keys)]
            rows += [_row("significant_findings", "p_fdr_min", float(sig["p_fdr"].min()), **keys),
                     _row("significant_findings", "p_fdr_max", float(sig["p_fdr"].max()), **keys)]
            for grp, n in sig["best_group"].value_counts().items():
                rows.append(_row("significant_findings", "n_highest_group", int(n), group=str(grp), **keys))
    return rows


def modality_totals(diff: pd.DataFrame) -> List[Dict]:
    rows = []
    allp = diff[(diff["panel"] == "public") & (diff["stratum"] == "all") & ~diff["attribute"].isin(ACQUISITION)]
    for measure, g in allp.groupby("measure"):
        rows += [_row("modality_totals", "n_differences", len(g), measure=measure),
                 _row("modality_totals", "n_significant", int((g["p_fdr"] < ALPHA).sum()), measure=measure)]
    return rows


def image_count_comparison(diff: pd.DataFrame) -> List[Dict]:
    rows = []
    chest = diff[(diff["pool"] == "cxr_pool") & (diff["stratum"] == "all") & diff["attribute"].isin(DEMOGRAPHIC)].copy()
    chest["ratio"] = chest["chance"] / chest["formula_chance"]
    for scope, g in (("public", chest[chest["panel"] == "public"]), ("all_chest", chest)):
        for measure, gm in g.groupby("measure"):
            keys = dict(pool="cxr_pool", panel=scope, measure=measure)
            rows += [_row("image_counts", "n_differences", len(gm), **keys),
                     _row("image_counts", "ratio_median", float(gm["ratio"].median()), **keys),
                     _row("image_counts", "n_ratio_above_1", int((gm["ratio"] > 1).sum()), **keys)]
        rows += [_row("image_counts", "n_significant_patients", int((g["p_fdr"] < ALPHA).sum()), pool="cxr_pool", panel=scope),
                 _row("image_counts", "n_significant_image_counts", int((g["formula_p_fdr"] < ALPHA).sum()), pool="cxr_pool", panel=scope)]
    return rows


def stratum_summaries(strata: pd.DataFrame) -> List[Dict]:
    rows = []
    s = strata.copy()
    s["condition"] = [condition_of(e) for e in s["encoder"]]
    s["objective"] = [objective_of(e) for e in s["encoder"]]
    for (by, att, measure), g in s[s["pool"] == "cxr_pool"].groupby(["by", "attribute", "measure"]):
        keys = dict(pool="cxr_pool", attribute=att, stratum=by, measure=measure)
        for cond, gc in g.groupby("condition"):
            rows += [_row("strata_by_condition", "n_models", len(gc), group=cond, **keys),
                     _row("strata_by_condition", "n_significant", int((gc["p_fdr"] < ALPHA).sum()), group=cond, **keys),
                     _row("strata_by_condition", "difference_median", float(gc["difference_within"].median()), group=cond, **keys)]
        for obj, go in g[g["objective"] != ""].groupby("objective"):
            rows += [_row("strata_by_objective", "n_models", len(go), group=obj, **keys),
                     _row("strata_by_objective", "n_significant", int((go["p_fdr"] < ALPHA).sum()), group=obj, **keys)]
        for enc, ge in g[g["panel"] == "public"].groupby("encoder"):
            rows.append(_row("strata_by_encoder", "n_significant", int((ge["p_fdr"] < ALPHA).sum()), group=enc, **keys))
        valid = g[g["panel"] == "public"]["n_boot_used"]
        rows.append(_row("strata_valid_resamples", "n_boot_used_median", float(valid.median()), panel="public", **keys))
    return rows


def reduced_families(strata: pd.DataFrame, diff: pd.DataFrame) -> List[Dict]:
    rows = []
    for name, frame, by_col in (("strata", strata, "by"), ("differences", diff, "stratum")):
        frame = frame[(frame["pool"] == "cxr_pool") & ~frame["encoder"].isin(OVERLAP_ENCODERS)].copy()
        frame["p_reduced"] = np.nan
        for _, idx in frame.groupby(["pool", by_col, "attribute", "measure"]).groups.items():
            frame.loc[idx, "p_reduced"] = bh_fdr(frame.loc[idx, "p_raw"].values)
        for (by, att, measure), g in frame.groupby([by_col, "attribute", "measure"]):
            pub = g[g["panel"] == "public"]
            keys = dict(pool="cxr_pool", panel="public_reduced", attribute=att, stratum=by, measure=measure)
            rows += [_row(f"reduced_{name}", "n_family", len(g), **keys),
                     _row(f"reduced_{name}", "n_models", len(pub), **keys),
                     _row(f"reduced_{name}", "n_significant", int((pub["p_reduced"] < ALPHA).sum()), **keys)]
            changed = pub[(pub["p_reduced"] < ALPHA) != (pub["p_fdr"] < ALPHA)]
            for _, r in changed.iterrows():
                rows += [_row(f"reduced_{name}", "p_fdr_full_family", float(r["p_fdr"]), group=f"{r['encoder']}/{r['finding']}", **keys),
                         _row(f"reduced_{name}", "p_fdr_reduced_family", float(r["p_reduced"]), group=f"{r['encoder']}/{r['finding']}", **keys)]
    return rows


def contrast_summaries(con: pd.DataFrame) -> List[Dict]:
    rows = []
    for (contrast, att, stat), g in con.groupby(["contrast", "attribute", "statistic"]):
        sig = g["significant"].astype(bool)
        keys = dict(attribute=att, measure=stat, group=contrast)
        rows += [_row("contrasts", "n_pairs", len(g), **keys),
                 _row("contrasts", "n_significant", int(sig.sum()), **keys),
                 _row("contrasts", "n_significant_increase", int((sig & (g["delta"] > 0)).sum()), **keys),
                 _row("contrasts", "n_significant_decrease", int((sig & (g["delta"] < 0)).sum()), **keys),
                 _row("contrasts", "p_fdr_min", float(g["p_fdr"].min()), **keys),
                 _row("contrasts", "delta_median", float(g["delta"].median()), **keys)]
    race = con[con["attribute"] == "race_grp"]
    for stat, g in race.groupby("statistic"):
        rows += [_row("contrasts_all", "n_pairs", len(g), attribute="race_grp", measure=stat),
                 _row("contrasts_all", "n_significant", int(g["significant"].astype(bool).sum()), attribute="race_grp", measure=stat),
                 _row("contrasts_all", "p_fdr_min", float(g["p_fdr"].min()), attribute="race_grp", measure=stat)]
    edited = race[race["contrast"].isin(["data:scrubbed_minus_natural", "data:amplified_minus_natural"])]
    for stat, g in edited.groupby("statistic"):
        rows += [_row("contrasts_edited_reports", "n_pairs", len(g), attribute="race_grp", measure=stat),
                 _row("contrasts_edited_reports", "n_significant", int(g["significant"].astype(bool).sum()), attribute="race_grp", measure=stat),
                 _row("contrasts_edited_reports", "p_fdr_min", float(g["p_fdr"].min()), attribute="race_grp", measure=stat)]
    nat = race[race["contrast"].str.startswith("objective:") & race["contrast"].str.endswith(":natural")]
    for stat, g in nat.groupby("statistic"):
        rows.append(_row("contrasts_objectives", "abs_delta_median", float(g["delta"].abs().median()), attribute="race_grp", measure=stat))
    it = race[race["contrast"].str.startswith("objective:image_text_minus_") & race["contrast"].str.endswith(":natural")
              & (race["statistic"] == "difference_fpr") & race["significant"].astype(bool)]
    rows += [_row("contrasts_image_text", "n_significant", len(it), attribute="race_grp", measure="difference_fpr"),
             _row("contrasts_image_text", "n_smaller_with_image_text", int((it["delta"] < 0).sum()), attribute="race_grp", measure="difference_fpr"),
             _row("contrasts_image_text", "n_larger_with_image_text", int((it["delta"] > 0).sum()), attribute="race_grp", measure="difference_fpr")]
    return rows


def seed_summaries(spread: pd.DataFrame) -> List[Dict]:
    return [_row("seed_spread", "spread_median", float(g["spread"].median()), attribute=att, measure=stat)
            for (att, stat), g in spread.groupby(["attribute", "statistic"])]


def insurance_age_shares(man: pd.DataFrame, age_limit: float = 60) -> List[Dict]:
    test = man[man["split"].astype(str).str.lower().eq("test") & man["insurance_grp"].notna() & man["age"].notna()]
    rows = []
    for grp, g in test.groupby("insurance_grp"):
        rows.append(_row("insurance_age", "share_age_at_least_60", float((pd.to_numeric(g["age"]) >= age_limit).mean()),
                         pool="cxr_pool", attribute="insurance_grp", group=str(grp)))
    return rows


def dataset_counts(manifests: Dict[str, pd.DataFrame]) -> List[Dict]:
    rows = []
    for pool, man in manifests.items():
        for ds, g in man.groupby("dataset"):
            test = g[g["split"].astype(str).str.lower().eq("test")]
            keys = dict(pool=pool, group=str(ds))
            who = test["subject_id"] if "subject_id" in test.columns else test["case_id"]
            who = who.where(who.notna(), test["case_id"])
            rows += [_row("datasets", "n_images", len(g), **keys), _row("datasets", "n_test_images", len(test), **keys),
                     _row("datasets", "n_test_clusters", int(who.nunique()), **keys)]
            if ds not in NO_PATIENT_IDENTIFIER and "subject_id" in g.columns:
                rows += [_row("datasets", "n_test_patients", int(test["subject_id"].nunique()), **keys),
                         _row("datasets", "n_patients", int(g["subject_id"].nunique()), **keys)]
    return rows


def pretraining_overlap(man: pd.DataFrame, training_list_csv: str) -> List[Dict]:
    ids = set(read_csv_defensively(training_list_csv)["image_id"].astype(str))
    test = man[man["split"].astype(str).str.lower().eq("test")]
    rows = []
    for ds, g in test.groupby("dataset"):
        keys = g["image_key"].astype(str).map(lambda k: os.path.splitext(os.path.basename(k))[0])
        if ds == "chexpert":
            m = g["image_key"].astype(str).str.extract(r"patient(\d+)/study(\d+)/(view\d+_\w+)\.jpg")
            keys = "p" + m[0] + "_s" + m[1] + "_" + m[2]
        hit = int(keys.isin(ids).sum())
        rows += [_row("pretraining_overlap", "n_test_images", len(g), pool="cxr_pool", group=str(ds)),
                 _row("pretraining_overlap", "n_in_training_list", hit, pool="cxr_pool", group=str(ds))]
    return rows


def main_reported_statistics(global_config_path: str, out_dir: Optional[str] = None,
                             pretraining_list_csv: Optional[str] = None) -> str:
    cfg = read_config(global_config_path)["BiasOrigin"]
    src = cfg["subgroups"]["results_chance_dir"]
    out_dir = out_dir or cfg["subgroups"]["final_tables_dir"]
    diff = _tested(read_csv_defensively(os.path.join(src, "chance_differences.csv")))
    strata = _tested(pd.concat([read_csv_defensively(os.path.join(src, "chance_strata.csv")),
                                read_csv_defensively(os.path.join(src, "chance_strata_joint.csv"))], ignore_index=True))
    con = read_csv_defensively(os.path.join(src, "contrasts.csv"))
    spread = read_csv_defensively(os.path.join(src, "seed_spread.csv"))
    manifests = {p: read_csv_defensively(cfg["embeddings"]["pools"][p]["manifest"]) for p in cfg["subgroups"]["attributes"]}
    rows = (difference_summaries(diff) + modality_totals(diff) + image_count_comparison(diff) + stratum_summaries(strata)
            + reduced_families(strata, diff) + contrast_summaries(con) + seed_summaries(spread) + insurance_age_shares(manifests["cxr_pool"])
            + dataset_counts(manifests))
    if pretraining_list_csv:
        rows += pretraining_overlap(manifests["cxr_pool"], pretraining_list_csv)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "reported_statistics.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path

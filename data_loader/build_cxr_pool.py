"""
data_loader/build_cxr_pool.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_utils import (
    assert_unique_case_ids, finalize_manifest, make_case_ids, read_csv_defensively,
)
from data_loader.cxr_harmonization import (
    CANONICAL_CXR_FINDINGS, EXTENDED_MAPS, LABEL_MAPS, LABEL_POLICY,
    VIEW_KEEP, candidate_cxr_paths, harmonize_view,
)
from data_loader.sensitive_harmonization import attach_cxr_sensitive, CXR_SENSITIVE_COLS


_META_COLS: List[str] = [
    "site", "subject_id", "study_id", "age", "sex",
    "race", "ethnicity", "insurance", "language", "marital_status",
    "view", "report_rel_path",
] + CXR_SENSITIVE_COLS


def _binarize_series(s: pd.Series, positive: int, negatives: set, excludes: set) -> pd.Series:
    num = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[num == positive] = 1.0
    for nc in negatives:
        out[(out.isna()) & (num == nc)] = 0.0
    return out


def _decode_labels(site: str, df: pd.DataFrame, mapping: Dict[str, str],
                   canonical_list: List[str]) -> pd.DataFrame:
    policy = LABEL_POLICY[site]
    pos = int(policy["positive_code"])
    negs = {int(c) for c in policy["negative_codes"]}
    excl = {int(c) for c in policy["exclude_codes"]}

    result: Dict[str, pd.Series] = {}
    for canonical in canonical_list:
        native_cols = [nc for nc, c in mapping.items() if c == canonical and nc in df.columns]
        if not native_cols:
            result[canonical] = pd.Series(np.nan, index=df.index, dtype=float)
            continue
        decoded = [_binarize_series(df[nc], pos, negs, excl) for nc in native_cols]
        result[canonical] = decoded[0] if len(decoded) == 1 else pd.concat(decoded, axis=1).max(axis=1)
    return pd.DataFrame(result, index=df.index)


def _load_site(site: str, scfg: dict, drop_lateral: bool) -> pd.DataFrame:
    master = scfg["master_csv"]
    if not os.path.exists(master):
        print(f"[build_cxr_pool] {site}: master not found ({master}); skipping site.")
        return pd.DataFrame()

    id_like_cols = ["subject_id_col", "study_id_col", "image_key_col", "image_subdir_col"]
    dtype_map = {scfg[k]: str for k in id_like_cols if scfg.get(k)}
    df = read_csv_defensively(master, dtype=dtype_map) if dtype_map else read_csv_defensively(master)
    n0 = len(df)

    key_col = scfg.get("image_key_col")
    if not key_col or key_col not in df.columns:
        print(f"[build_cxr_pool] {site}: image_key_col '{key_col}' missing; skipping site.")
        return pd.DataFrame()

    view_col = scfg.get("view_col")
    if view_col and view_col in df.columns:
        view_harm = df[view_col].apply(lambda v: harmonize_view(site, v))
        if drop_lateral:
            keep = view_harm.isin(VIEW_KEEP)
            df = df[keep].copy()
            view_harm = view_harm[keep]
        df["_view_harm"] = view_harm.values
    else:
        df["_view_harm"] = "UNKNOWN"

    if df.empty:
        print(f"[build_cxr_pool] {site}: no rows after frontal filter.")
        return pd.DataFrame()

    out = pd.DataFrame(index=df.index)
    out["site"]     = site
    out["dataset"]  = site
    out["modality"] = "cxr"
    out["split"]    = df[scfg["split_col"]].astype(str) if scfg.get("split_col") in df.columns else "train"
    out["image_key"] = df[key_col].astype(str)
    sub_col = scfg.get("image_subdir_col")
    out["image_subdir"] = df[sub_col].astype(str) if sub_col and sub_col in df.columns else np.nan

    def _col(name):
        c = scfg.get(name)
        return df[c] if c and c in df.columns else np.nan

    out["subject_id"]      = _col("subject_id_col")
    out["study_id"]        = _col("study_id_col")
    out["age"]             = _col("age_col")
    out["sex"]             = _col("sex_col")
    out["race"]            = _col("race_col")
    out["ethnicity"]       = _col("ethnicity_col")
    out["insurance"]       = _col("insurance_col")
    out["language"]        = np.nan
    out["marital_status"]  = np.nan
    out["report_rel_path"] = _col("report_col")
    out["view"]            = df["_view_harm"].values

    out["case_id"] = make_case_ids(site, out["image_key"].tolist())

    sid = out["subject_id"]
    missing = sid.isna() if hasattr(sid, "isna") else pd.Series([True] * len(out))
    n_missing = int(missing.sum())
    if n_missing:
        out["subject_id"] = np.where(missing, out["case_id"], sid.astype(object))
        print(f"[cxr_pool] {site}: {n_missing} rows have no patient id; using case_id "
              f"as the bootstrap cluster (one image per cluster).")

    canon = _decode_labels(site, df, LABEL_MAPS[site], CANONICAL_CXR_FINDINGS)
    out = pd.concat([out, canon], axis=1)
    ext_map = EXTENDED_MAPS.get(site, {})
    if ext_map:
        ext = _decode_labels(site, df, {k: v for k, v in ext_map.items()},
                             sorted(set(ext_map.values())))
        out = pd.concat([out, ext], axis=1)

    if site == "chexpert" and scfg.get("chexpert_plus_csv") and os.path.exists(scfg["chexpert_plus_csv"]):
        join_col = scfg.get("report_join_col", "jpg_rel_path")
        usecols = [join_col, scfg.get("race_col"), scfg.get("ethnicity_col"), scfg.get("insurance_col")]
        usecols = [c for c in usecols if c]
        try:
            plus = read_csv_defensively(scfg["chexpert_plus_csv"], usecols=usecols)
            plus = plus.drop_duplicates(subset=[join_col])
            plus = plus.rename(columns={join_col: "image_key",
                                        scfg.get("race_col"): "race",
                                        scfg.get("ethnicity_col"): "ethnicity",
                                        scfg.get("insurance_col"): "insurance"})
            out = out.drop(columns=["race", "ethnicity", "insurance"], errors="ignore").merge(
                plus[["image_key", "race", "ethnicity", "insurance"]], on="image_key", how="left")
            print(f"[build_cxr_pool] chexpert: merged CheXpert Plus race/insurance.")
        except Exception as e:
            print(f"[build_cxr_pool] chexpert: CheXpert Plus merge skipped ({e}).")

    print(f"[build_cxr_pool] {site}: {len(out)}/{n0} rows after frontal filter.")
    return out


def _verify_exists(df: pd.DataFrame, res: int) -> pd.DataFrame:
    keep_mask = []
    n_by_site_missing: Dict[str, int] = {}
    for row in df.itertuples(index=False):
        cands = candidate_cxr_paths(row.site, _SITE_IMAGE_ROOT[row.site],
                                    row.image_key, getattr(row, "image_subdir", None), res)
        ok = any(os.path.exists(c) for c in cands)
        keep_mask.append(ok)
        if not ok:
            n_by_site_missing[row.site] = n_by_site_missing.get(row.site, 0) + 1
    if any(n_by_site_missing.values()):
        for s, n in n_by_site_missing.items():
            print(f"[build_cxr_pool] {s}: {n} rows dropped (preprocessed image not found).")
    return df[pd.Series(keep_mask, index=df.index)].copy()


_SITE_IMAGE_ROOT: Dict[str, str] = {}


def main_build_cxr_pool(global_config_path: str) -> str:
    cfg     = read_config(global_config_path)["BiasOrigin"]
    cxr     = cfg["cxr"]
    sens    = cfg["sensitive"]
    res     = int(cfg.get("target_resolution", 224))
    out_csv = cxr["pool_manifest_csv"]
    drop_lat = bool(cxr.get("drop_lateral", True))

    parts: List[pd.DataFrame] = []
    for site, scfg in cxr["sites"].items():
        if not scfg.get("enabled", True):
            continue
        _SITE_IMAGE_ROOT[site] = scfg["image_root"]
        site_df = _load_site(site, scfg, drop_lat)
        if not site_df.empty:
            parts.append(site_df)

    if not parts:
        raise RuntimeError("[build_cxr_pool] no site produced rows.")

    pool = pd.concat(parts, ignore_index=True)

    demo_csv = cxr["mimic_demographics"]["out_csv"]
    if os.path.exists(demo_csv):
        demo = read_csv_defensively(demo_csv)
        demo_cols = [c for c in ["subject_id", "race", "insurance", "language", "marital_status"] if c in demo.columns]
        demo = demo[demo_cols].copy()
        demo["subject_id"] = pd.to_numeric(demo["subject_id"], errors="coerce")
        is_mimic = pool["site"] == "mimic"
        mimic_part = pool[is_mimic].copy()
        mimic_part["subject_id"] = pd.to_numeric(mimic_part["subject_id"], errors="coerce")
        mimic_part = mimic_part.drop(columns=["race", "insurance", "language", "marital_status"], errors="ignore")
        mimic_part = mimic_part.merge(demo, on="subject_id", how="left")
        pool = pd.concat([mimic_part, pool[~is_mimic]], ignore_index=True)
        print(f"[build_cxr_pool] merged MIMIC-IV demographics for {is_mimic.sum()} MIMIC rows.")
    else:
        print(f"[build_cxr_pool] MIMIC-IV demographics CSV not found ({demo_csv}); "
              f"MIMIC race/insurance will be NaN. Run build_mimic_demographics first.")

    pool = attach_cxr_sensitive(pool, sens)

    if cxr.get("verify_image_exists", True):
        pool = _verify_exists(pool, res)

    label_cols = CANONICAL_CXR_FINDINGS + sorted(
        {v for s in EXTENDED_MAPS.values() for v in s.values()})
    pool = finalize_manifest(pool, label_cols=label_cols, meta_cols=_META_COLS)
    assert_unique_case_ids(pool)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pool.to_csv(out_csv, index=False)

    print(f"\n[build_cxr_pool] pool -> {out_csv}  ({len(pool)} rows)")
    for s, grp in pool.groupby("site"):
        print(f"  {s}: {len(grp)}")
    for col in CXR_SENSITIVE_COLS:
        n_known = pool[col].notna().sum()
        print(f"  {col}: {n_known} non-missing")
    return out_csv

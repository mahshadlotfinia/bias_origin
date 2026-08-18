"""
data_loader/build_scin_pool.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import List, Optional

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_utils import (
    assert_unique_case_ids, finalize_manifest, make_case_ids, read_csv_defensively,
)
from data_loader.derm_fundus_build_utils import assign_split, ensure_resized
from data_loader.sensitive_harmonization import (
    collapse_fst_from_scale, collapse_monk, collapse_race, DERM_SENSITIVE_COLS,
)

_LABEL_COLS = ["malignant"]
_META_COLS = ["condition", "efst_raw", "monk_raw", "race_raw"] + DERM_SENSITIVE_COLS

_RACE_ETHNICITY_SUFFIXES = [
    "american_indian_or_alaska_native", "asian", "black_or_african_american",
    "hispanic_latino_or_spanish_origin", "middle_eastern_or_north_african",
    "native_hawaiian_or_pacific_islander", "white", "other_race",
    "prefer_not_to_answer", "two_or_more_after_mitigation",
]

_IMAGE_PATH_COLS = ["image_1_path", "image_2_path", "image_3_path"]


def _combined_race_from_onehot(row: pd.Series) -> Optional[str]:
    hits = []
    for suf in _RACE_ETHNICITY_SUFFIXES:
        col = f"race_ethnicity_{suf}"
        if col in row.index and str(row[col]).strip().upper() in ("TRUE", "1", "YES"):
            hits.append(suf)
    return ",".join(hits) if hits else None


def _load_keywords(path: str) -> List[str]:
    if not path or not os.path.exists(path):
        print(f"[scin] malignant-keywords file not found ({path}); using built-in list.")
        return ["melanoma", "basal cell carcinoma", "squamous cell carcinoma",
                "actinic keratosis", "carcinoma", "malignant", "sarcoma",
                "mycosis fungoides", "bowen"]
    with open(path) as f:
        return [ln.strip().lower() for ln in f if ln.strip() and not ln.startswith("#")]


def _malignant_from_condition(cond, keywords: List[str]) -> float:
    s = str(cond or "").strip().lower()
    if not s or s in ("nan", "none"):
        return np.nan
    return 1.0 if any(k in s for k in keywords) else 0.0


def _coalesce_col(df: pd.DataFrame, name: Optional[str]):
    return df[name] if name and name in df.columns else pd.Series(np.nan, index=df.index)


def main_build_scin_pool(global_config_path: str) -> str:
    cfg  = read_config(global_config_path)["BiasOrigin"]
    dcfg = cfg["derm"]
    scfg = dcfg["sources"]["scin"]
    if not scfg.get("enabled", True):
        print("[scin] disabled; skipping.")
        return dcfg["pool_manifest_csv"].replace(".csv", "_scin.csv")

    cases_csv, labels_csv = scfg["cases_csv"], scfg["labels_csv"]
    for p in (cases_csv, labels_csv):
        if not os.path.exists(p):
            raise FileNotFoundError(f"[scin] required file not found: {p}")
    cases  = read_csv_defensively(cases_csv)
    labels = read_csv_defensively(labels_csv)

    cid = scfg["case_id_col"]
    df = cases.merge(labels, on=cid, how="left", suffixes=("", "_lab"))

    keywords = _load_keywords(scfg.get("malignant_keywords_file"))
    src_root = scfg["dir"]
    res = int(cfg.get("target_resolution", 224))
    pre_root = os.path.join(dcfg["image_root"], "preprocessed224", "scin")

    efst_col = scfg.get("efst_col")
    emst_col = scfg.get("emst_col")
    cond_col = scfg.get("condition_col")

    rows, n_missing = [], 0
    for _, r in df.iterrows():
        race_raw = _combined_race_from_onehot(r)
        for img_col_n in _IMAGE_PATH_COLS:
            rel = str(r.get(img_col_n, "")) if img_col_n in df.columns else ""
            if not rel or rel.lower() in ("nan", "none"):
                continue
            stem = os.path.splitext(os.path.basename(rel))[0] + ".png"
            src = os.path.join(src_root, rel)
            if not os.path.exists(src):
                alt = os.path.join(src_root, scfg.get("images_subdir", "images"), os.path.basename(rel))
                src = alt if os.path.exists(alt) else src
            dst = os.path.join(pre_root, stem)
            if not ensure_resized(src, dst, res):
                n_missing += 1
                continue
            rows.append({
                "dataset": "scin", "modality": "derm", "split": "",
                "image_key": stem, "image_subdir": "scin",
                "malignant": _malignant_from_condition(r.get(cond_col), keywords),
                "condition": str(r.get(cond_col, "")),
                "efst_raw": r.get(efst_col), "monk_raw": r.get(emst_col),
                "race_raw": race_raw,
                "fst_grp": collapse_fst_from_scale(r.get(efst_col)),
                "monk_grp": collapse_monk(r.get(emst_col)),
                "race_grp": collapse_race(race_raw, cfg["sensitive"]),
            })
    if n_missing:
        print(f"[scin] {n_missing} rows skipped (no resolvable image).")
    if not rows:
        raise RuntimeError("[scin] no rows produced; check SCIN image paths.")

    out = pd.DataFrame(rows)
    out["case_id"] = make_case_ids("scin", out["image_key"].tolist())
    out = out.drop_duplicates(subset=["case_id"]).reset_index(drop=True)
    split_map = assign_split(out["case_id"].tolist(), dcfg["split_fractions"],
                             seed=int(cfg.get("seed", 42)))
    out["split"] = out["case_id"].map(split_map)

    out = finalize_manifest(out, label_cols=_LABEL_COLS, meta_cols=_META_COLS)
    assert_unique_case_ids(out)

    scin_csv = dcfg["pool_manifest_csv"].replace(".csv", "_scin.csv")
    os.makedirs(os.path.dirname(scin_csv), exist_ok=True)
    out.to_csv(scin_csv, index=False)
    print(f"[scin] {len(out)} rows -> {scin_csv}")
    for col in ("fst_grp", "monk_grp", "race_grp"):
        print(f"    {col}: {out[col].notna().sum()} non-missing")
    return scin_csv


def main_build_derm_pool(global_config_path: str) -> str:
    from data_loader.build_ddi_pool import main_build_ddi_pool
    from data_loader.build_fitz17k_pool import main_build_fitz17k_pool

    cfg  = read_config(global_config_path)["BiasOrigin"]
    dcfg = cfg["derm"]
    pool_csv = dcfg["pool_manifest_csv"]

    standalone = []
    for fn in (main_build_ddi_pool, main_build_fitz17k_pool, main_build_scin_pool):
        try:
            standalone.append(fn(global_config_path))
        except Exception as e:
            print(f"[derm_pool] {fn.__name__} failed/skipped: {e}")

    frames = [read_csv_defensively(p) for p in standalone if p and os.path.exists(p)]
    if not frames:
        raise RuntimeError("[derm_pool] no derm source manifests were produced.")
    pool = pd.concat(frames, ignore_index=True)
    pool = finalize_manifest(pool, label_cols=_LABEL_COLS,
                             meta_cols=["condition"] + DERM_SENSITIVE_COLS)
    assert_unique_case_ids(pool)
    os.makedirs(os.path.dirname(pool_csv), exist_ok=True)
    pool.to_csv(pool_csv, index=False)
    print(f"\n[derm_pool] combined -> {pool_csv}  ({len(pool)} rows)")
    for ds, grp in pool.groupby("dataset"):
        print(f"  {ds}: {len(grp)}")
    return pool_csv

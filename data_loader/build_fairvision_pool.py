"""
data_loader/build_fairvision_pool.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_utils import (
    assert_unique_case_ids, finalize_manifest, make_case_ids, read_csv_defensively,
)
from data_loader.sensitive_harmonization import (
    bin_age, collapse_fundus_race, normalize_sex, FUNDUS_SENSITIVE_COLS,
)

_DEF = {
    "image_path_col": "filename", "label_col": "glaucoma", "split_col": "use",
    "race_col": "race", "gender_col": "gender", "ethnicity_col": "ethnicity",
    "age_col": "age",
}
_USE_TO_FOLDER = {"training": "train", "train": "train",
                  "validation": "valid", "valid": "valid", "val": "valid",
                  "test": "test"}
_USE_TO_SPLIT = {"training": "train", "validation": "val", "test": "test"}


def _to_binary(v) -> float:
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    if s in ("1", "1.0", "yes", "true", "positive", "present"):
        return 1.0
    if s in ("0", "0.0", "no", "false", "negative", "absent", "normal"):
        return 0.0
    try:
        return 1.0 if float(v) >= 0.5 else 0.0
    except (TypeError, ValueError):
        return np.nan


def _ethnicity_grp(v):
    s = str(v or "").strip().lower()
    if not s or s in ("nan", "none"):
        return None
    if "non" in s and "hispanic" in s:
        return "Non_Hispanic"
    if "hispanic" in s or "latino" in s:
        return "Hispanic"
    return "Other"


def _digits(s: str) -> str:
    return "".join(ch for ch in str(s) if ch.isdigit())


def main_build_fairvision_pool(global_config_path: str) -> str:
    cfg  = read_config(global_config_path)["BiasOrigin"]
    fcfg = cfg["fundus"]
    scfg = fcfg["sources"]["fairvision"]
    sens = cfg["sensitive"]

    root = scfg["dir"]
    meta = scfg.get("metadata_csv") or os.path.join(root, "data_summary.csv")
    if not os.path.exists(meta):
        raise FileNotFoundError(
            f"[fairvision] data_summary.csv not found at {meta}. Expected the flat "
            f"glaucoma-only release with a single CSV at the dataset root.")

    cols   = {**_DEF, **{k: scfg.get(k) for k in _DEF if scfg.get(k)}}
    prefix = scfg.get("slo_prefix", "slo_fundus_")
    use_map   = {**_USE_TO_FOLDER, **(scfg.get("split_folders") or {})}
    split_map = {k.lower(): v for k, v in (scfg.get("split_folders") or {}).items()}
    image_root = fcfg["image_root"]

    df = read_csv_defensively(meta)
    rows, n_missing = [], 0
    for _, r in df.iterrows():
        fid = _digits(r.get(cols["image_path_col"], ""))
        if not fid:
            n_missing += 1
            continue
        use_val = str(r.get(cols["split_col"], "")).strip().lower()
        folder = use_map.get(use_val)
        if not folder:
            n_missing += 1
            continue
        stem = f"{prefix}{fid}.png"
        src = os.path.join(root, folder, "images", f"{prefix}{fid}.jpg")
        if not os.path.exists(src):
            n_missing += 1
            continue
        split = _USE_TO_SPLIT.get(use_val) or split_map.get(use_val, use_val)
        rows.append({
            "dataset": "fairvision", "modality": "fundus", "split": split,
            "image_key": f"glaucoma/{stem}", "image_subdir": "fairvision",
            "image_relpath": os.path.relpath(src, image_root),
            "disease_subset": "glaucoma",
            "race_grp": collapse_fundus_race(r.get(cols["race_col"]), sens),
            "sex_grp": normalize_sex(r.get(cols["gender_col"])),
            "ethnicity_grp": _ethnicity_grp(r.get(cols["ethnicity_col"])),
            "age_grp": bin_age(r.get(cols["age_col"]), sens),
            "race_raw": r.get(cols["race_col"]),
            "age_raw": r.get(cols["age_col"]),
            "glaucoma": _to_binary(r.get(cols["label_col"])),
        })
    if n_missing:
        print(f"[fairvision] {n_missing} rows skipped (no resolvable SLO jpg or "
              f"unknown split).")
    if not rows:
        raise RuntimeError("[fairvision] produced no rows; check that "
                           f"{root}/<train|valid|test>/images/{prefix}<id>.jpg exist.")

    out = pd.DataFrame(rows)
    out["case_id"] = make_case_ids("fairvision", out["image_key"].tolist())
    out = out.drop_duplicates(subset=["case_id"]).reset_index(drop=True)
    meta_cols = ["disease_subset", "race_raw", "age_raw"] + FUNDUS_SENSITIVE_COLS
    out = finalize_manifest(out, label_cols=["glaucoma"], meta_cols=meta_cols)
    assert_unique_case_ids(out)

    pool_csv = fcfg["pool_manifest_csv"]
    os.makedirs(os.path.dirname(pool_csv), exist_ok=True)
    out.to_csv(pool_csv, index=False)
    print(f"\n[fairvision] {len(out)} rows -> {pool_csv}")
    print(f"  glaucoma: race_grp non-missing={out['race_grp'].notna().sum()} | "
          f"pos={int(out['glaucoma'].sum(skipna=True))}")
    return pool_csv

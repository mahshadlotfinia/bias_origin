"""
data_loader/build_fitz17k_pool.py
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
from data_loader.derm_fundus_build_utils import assign_split
from data_loader.sensitive_harmonization import collapse_fst_from_scale, DERM_SENSITIVE_COLS

_LABEL_COLS = ["malignant"]
_META_COLS = ["subject_id", "condition", "three_partition", "fst_scale_raw"] + DERM_SENSITIVE_COLS


def _malignant_from_partition(v) -> float:
    s = str(v).strip().lower()
    if "malignant" in s:
        return 1.0
    if "benign" in s:
        return 0.0
    return np.nan


def main_build_fitz17k_pool(global_config_path: str) -> str:
    cfg  = read_config(global_config_path)["BiasOrigin"]
    dcfg = cfg["derm"]
    scfg = dcfg["sources"]["fitz17k"]
    if not scfg.get("enabled", True):
        print("[fitz17k] disabled; skipping.")
        return dcfg["pool_manifest_csv"].replace(".csv", "_fitz17k.csv")

    csv = scfg["csv"]
    if not os.path.exists(csv):
        raise FileNotFoundError(f"[fitz17k] csv not found: {csv}")
    df = read_csv_defensively(csv)

    id_col   = scfg["id_col"]
    fst_col  = scfg["fst_col"]
    part_col = scfg["three_part_col"]
    cond_col = scfg.get("condition_col")
    src_root = os.path.join(scfg["dir"], scfg.get("images_subdir", "images"))
    image_root = dcfg["image_root"]

    rows, n_missing, n_dropfst = [], 0, 0
    for _, r in df.iterrows():
        fst_grp = collapse_fst_from_scale(r.get(fst_col))
        if fst_grp is None and scfg.get("drop_unknown_fst", True):
            n_dropfst += 1
            continue
        key = str(r[id_col])
        stem = key + ".png"
        src = None
        for ext in (".jpg", ".png", ".jpeg"):
            cand = os.path.join(src_root, key + ext)
            if os.path.exists(cand):
                src = cand
                break
        if src is None:
            n_missing += 1
            continue
        rows.append({
            "dataset": "fitz17k", "modality": "derm", "split": "",
            "image_key": stem, "image_subdir": "fitz17k",
            "image_relpath": os.path.relpath(src, image_root),
            "malignant": _malignant_from_partition(r.get(part_col)),
            "condition": str(r.get(cond_col, "")) if cond_col else "",
            "three_partition": str(r.get(part_col, "")),
            "fst_scale_raw": r.get(fst_col),
            "fst_grp": fst_grp, "monk_grp": np.nan, "race_grp": np.nan,
        })
    if n_dropfst:
        print(f"[fitz17k] {n_dropfst} rows dropped (FST not in 1..6).")
    if n_missing:
        print(f"[fitz17k] {n_missing} images not found on disk; skipped "
              f"(download images from `url` into {src_root}).")
    if not rows:
        raise RuntimeError("[fitz17k] no rows produced; check images dir.")

    out = pd.DataFrame(rows)
    out["case_id"] = make_case_ids("fitz17k", out["image_key"].tolist())
    out["subject_id"] = out["case_id"]
    split_map = assign_split(out["case_id"].tolist(), dcfg["split_fractions"],
                             seed=int(cfg.get("seed", 42)))
    out["split"] = out["case_id"].map(split_map)

    out = finalize_manifest(out, label_cols=_LABEL_COLS, meta_cols=_META_COLS)
    assert_unique_case_ids(out)

    fz_csv = dcfg["pool_manifest_csv"].replace(".csv", "_fitz17k.csv")
    os.makedirs(os.path.dirname(fz_csv), exist_ok=True)
    out.to_csv(fz_csv, index=False)
    print(f"[fitz17k] {len(out)} rows -> {fz_csv}")
    for g, n in out["fst_grp"].value_counts(dropna=False).items():
        print(f"    fst_grp {g}: {n}")
    return fz_csv

"""
data_loader/build_ddi_pool.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Optional

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_utils import (
    assert_unique_case_ids, finalize_manifest, make_case_ids, read_csv_defensively,
)
from data_loader.derm_fundus_build_utils import assign_split
from data_loader.sensitive_harmonization import collapse_fst_from_ddi_code, DERM_SENSITIVE_COLS

_LABEL_COLS = ["malignant"]
_META_COLS = ["subject_id", "condition", "skin_tone_raw"] + DERM_SENSITIVE_COLS


def _to_binary(v) -> float:
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    if s in ("1", "1.0", "true", "yes", "malignant", "y", "t"):
        return 1.0
    if s in ("0", "0.0", "false", "no", "benign", "n", "f"):
        return 0.0
    try:
        return 1.0 if float(v) >= 0.5 else 0.0
    except (TypeError, ValueError):
        return np.nan


def main_build_ddi_pool(global_config_path: str) -> str:
    cfg  = read_config(global_config_path)["BiasOrigin"]
    dcfg = cfg["derm"]
    scfg = dcfg["sources"]["ddi"]
    if not scfg.get("enabled", True):
        print("[ddi] disabled; skipping.")
        return dcfg["pool_manifest_csv"]

    meta_csv = scfg["metadata_csv"]
    if not os.path.exists(meta_csv):
        raise FileNotFoundError(f"[ddi] metadata not found: {meta_csv}")
    df = read_csv_defensively(meta_csv)

    file_col = scfg["file_col"]
    src_root = os.path.join(scfg["dir"], scfg.get("images_subdir", "") or "")
    image_root = dcfg["image_root"]

    rows = []
    n_missing = 0
    for _, r in df.iterrows():
        rel = str(r[file_col])
        stem = rel.replace("/", "__")
        stem = os.path.splitext(stem)[0] + ".png"
        src = os.path.join(src_root, rel)
        if not os.path.exists(src):
            alt = os.path.join(scfg["dir"], rel)
            src = alt if os.path.exists(alt) else src
        if not os.path.exists(src):
            n_missing += 1
            continue
        rows.append({
            "dataset": "ddi", "modality": "derm", "split": "",
            "image_key": stem, "image_subdir": "ddi",
            "image_relpath": os.path.relpath(src, image_root),
            "malignant": _to_binary(r.get(scfg["malignant_col"])),
            "condition": str(r.get(scfg.get("disease_col"), "")),
            "skin_tone_raw": r.get(scfg["skin_tone_col"]),
            "fst_grp": collapse_fst_from_ddi_code(r.get(scfg["skin_tone_col"])),
            "monk_grp": np.nan, "race_grp": np.nan,
        })
    if n_missing:
        print(f"[ddi] {n_missing} images not found on disk; skipped.")
    if not rows:
        raise RuntimeError("[ddi] no rows produced; check image paths.")

    out = pd.DataFrame(rows)
    out["case_id"] = make_case_ids("ddi", out["image_key"].tolist())
    out["subject_id"] = out["case_id"]

    patient_col = scfg.get("patient_col")
    groups = df[patient_col].astype(str).tolist() if patient_col and patient_col in df.columns and len(df) == len(out) else None
    split_map = assign_split(out["case_id"].tolist(), dcfg["split_fractions"],
                             seed=int(cfg.get("seed", 42)), groups=groups)
    out["split"] = out["case_id"].map(split_map)

    out = finalize_manifest(out, label_cols=_LABEL_COLS, meta_cols=_META_COLS)
    assert_unique_case_ids(out)

    pool_csv = dcfg["pool_manifest_csv"]
    os.makedirs(os.path.dirname(pool_csv), exist_ok=True)
    ddi_csv = pool_csv.replace(".csv", "_ddi.csv")
    out.to_csv(ddi_csv, index=False)
    print(f"[ddi] {len(out)} rows -> {ddi_csv}")
    for g, n in out["fst_grp"].value_counts(dropna=False).items():
        print(f"    fst_grp {g}: {n}")
    return ddi_csv

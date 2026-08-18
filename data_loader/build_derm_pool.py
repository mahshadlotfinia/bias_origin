"""
data_loader/build_derm_pool.py
Created on June 30, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os

import pandas as pd

from config.serde import read_config
from data_loader.build_utils import (
    assert_unique_case_ids, finalize_manifest, read_csv_defensively,
)
from data_loader.sensitive_harmonization import DERM_SENSITIVE_COLS

_LABEL_COLS = ["malignant"]


def main_build_derm_pool(global_config_path: str) -> str:
    from data_loader.build_ddi_pool import main_build_ddi_pool
    from data_loader.build_fitz17k_pool import main_build_fitz17k_pool
    from data_loader.build_isic2019_pool import main_build_isic2019_pool

    cfg  = read_config(global_config_path)["BiasOrigin"]
    dcfg = cfg["derm"]
    pool_csv = dcfg["pool_manifest_csv"]

    standalone, failed = [], []
    for fn in (main_build_ddi_pool, main_build_fitz17k_pool, main_build_isic2019_pool):
        try:
            standalone.append(fn(global_config_path))
        except Exception as e:
            failed.append(f"{fn.__name__}: {type(e).__name__}: {e}")
    if failed:
        raise RuntimeError("[derm_pool] enabled derm source(s) failed to build; "
                           "fix them or set enabled: false in config.derm.sources:\n  - "
                           + "\n  - ".join(failed))

    frames = [read_csv_defensively(p) for p in standalone if p and os.path.exists(p)]
    if not frames:
        raise RuntimeError("[derm_pool] no derm source manifests were produced.")
    pool = pd.concat(frames, ignore_index=True)
    pool = finalize_manifest(pool, label_cols=_LABEL_COLS,
                             meta_cols=["subject_id", "condition"] + DERM_SENSITIVE_COLS)
    assert_unique_case_ids(pool)
    n_bad = int(pool["subject_id"].isna().sum())
    if n_bad:
        raise ValueError(
            f"[derm_pool] {n_bad} rows have no subject_id. Every source builder must "
            f"set it (the lesion for ISIC-2019, the case elsewhere); it is the "
            f"bootstrap cluster unit and the split grouping key.")
    spanning = pool.groupby("subject_id")["split"].nunique()
    n_span = int((spanning > 1).sum())
    if n_span:
        raise ValueError(
            f"[derm_pool] {n_span} subject_id groups span more than one split. "
            f"Splits must be group-disjoint or the test set leaks.")
    os.makedirs(os.path.dirname(pool_csv), exist_ok=True)
    pool.to_csv(pool_csv, index=False)
    print(f"\n[derm_pool] combined -> {pool_csv}  ({len(pool)} rows)")
    for ds, grp in pool.groupby("dataset"):
        print(f"  {ds}: {len(grp)}")
    return pool_csv

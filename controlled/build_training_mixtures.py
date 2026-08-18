"""
controlled/build_training_mixtures.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import List

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_utils import read_csv_defensively
from data_loader.cxr_harmonization import CANONICAL_PATHOLOGY_FINDINGS

_KEEP = ["case_id", "site", "dataset", "image_key", "image_subdir", "split",
         "subject_id", "race_grp"] + CANONICAL_PATHOLOGY_FINDINGS


def _mimic_train(pool: pd.DataFrame) -> pd.DataFrame:
    df = pool[(pool["site"] == "mimic") &
              (pool["split"].astype(str).str.lower().isin(["train", "training"]))].copy()
    if df.empty:
        df = pool[(pool["site"] == "mimic") &
                  (~pool["split"].astype(str).str.lower().isin(["test"]))].copy()
    keep = [c for c in _KEEP if c in df.columns]
    return df[keep].reset_index(drop=True)


def _balance_by_race(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    sub = df[df["race_grp"].notna() & (df["race_grp"].astype(str).str.lower() != "nan")].copy()
    if sub.empty:
        print("[mixtures] no race-labeled MIMIC train rows; balanced mixture empty.")
        return sub
    counts = sub["race_grp"].value_counts()
    n = int(counts.min())
    parts = [g.sample(n=n, random_state=seed) for _, g in sub.groupby("race_grp")]
    out = pd.concat(parts, ignore_index=True)
    print(f"[mixtures] balanced to {n}/group across {len(counts)} race groups "
          f"-> {len(out)} rows.")
    return out.reset_index(drop=True)


def main_build_training_mixtures(global_config_path: str) -> List[str]:
    cfg  = read_config(global_config_path)["BiasOrigin"]
    pool_csv = cfg["cxr"]["pool_manifest_csv"]
    out_dir  = cfg["controlled"]["mixtures_dir"]
    seed     = int(cfg.get("seed", 42))
    if not os.path.exists(pool_csv):
        raise FileNotFoundError(f"[mixtures] CXR pool not found: {pool_csv}. Run build_cxr_pool first.")

    pool = read_csv_defensively(pool_csv)
    natural = _mimic_train(pool)
    if natural.empty:
        raise RuntimeError("[mixtures] no MIMIC train rows found in the pool.")
    balanced = _balance_by_race(natural, seed)

    os.makedirs(out_dir, exist_ok=True)
    nat_csv = os.path.join(out_dir, "cxr_natural.csv")
    bal_csv = os.path.join(out_dir, "cxr_balanced.csv")
    natural.to_csv(nat_csv, index=False)
    balanced.to_csv(bal_csv, index=False)

    print(f"[mixtures] natural  -> {nat_csv}  ({len(natural)} rows)")
    print(f"[mixtures] balanced -> {bal_csv}  ({len(balanced)} rows)")
    for g, c in natural["race_grp"].value_counts(dropna=False).items():
        print(f"    natural race_grp {g}: {c}")
    return [nat_csv, bal_csv]

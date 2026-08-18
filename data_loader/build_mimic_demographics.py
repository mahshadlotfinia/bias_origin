"""
data_loader/build_mimic_demographics.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Optional

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_utils import read_csv_defensively
from data_loader.sensitive_harmonization import collapse_race, collapse_insurance


def _modal_nonnull(s: pd.Series) -> Optional[str]:
    vals = s.dropna()
    vals = vals[vals.astype(str).str.strip().str.lower() != ""]
    if vals.empty:
        return None
    return vals.value_counts().idxmax()


def main_build_mimic_demographics(global_config_path: str) -> str:
    cfg  = read_config(global_config_path)["BiasOrigin"]
    mcfg = cfg["cxr"]["mimic_demographics"]
    sens = cfg["sensitive"]

    if not mcfg.get("enabled", True):
        print("[mimic_demographics] disabled in config; skipping.")
        return mcfg["out_csv"]

    adm_csv = mcfg["admissions_csv"]
    pat_csv = mcfg["patients_csv"]
    out_csv = mcfg["out_csv"]

    for p in (adm_csv, pat_csv):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"[mimic_demographics] required MIMIC-IV file not found: {p}. "
                f"Place admissions.csv.gz and patients.csv.gz under "
                f"{os.path.dirname(adm_csv)} (PhysioNet MIMIC-IV v3.1 hosp module)."
            )

    print(f"[mimic_demographics] reading {adm_csv}")
    adm = read_csv_defensively(
        adm_csv,
        usecols=lambda c: c in {"subject_id", "race", "insurance",
                                "language", "marital_status"},
    )
    print(f"[mimic_demographics] reading {pat_csv}")
    pat = read_csv_defensively(
        pat_csv,
        usecols=lambda c: c in {"subject_id", "gender", "anchor_age"},
    )

    agg = (
        adm.groupby("subject_id")
        .agg({
            "race":           _modal_nonnull,
            "insurance":      _modal_nonnull,
            "language":       _modal_nonnull,
            "marital_status": _modal_nonnull,
        })
        .reset_index()
    )

    if not pat.empty:
        agg = agg.merge(pat, on="subject_id", how="left")

    agg["race_grp"]      = agg["race"].apply(lambda v: collapse_race(v, sens))
    agg["insurance_grp"] = agg["insurance"].apply(lambda v: collapse_insurance(v, sens))

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    agg.to_csv(out_csv, index=False)

    print(f"[mimic_demographics] {len(agg)} subjects -> {out_csv}")
    print("  race_grp distribution:")
    for g, n in agg["race_grp"].value_counts(dropna=False).items():
        print(f"    {g}: {n}")
    print("  insurance_grp distribution:")
    for g, n in agg["insurance_grp"].value_counts(dropna=False).items():
        print(f"    {g}: {n}")
    return out_csv

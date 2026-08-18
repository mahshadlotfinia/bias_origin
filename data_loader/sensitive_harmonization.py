"""
data_loader/sensitive_harmonization.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _match_patterns(
    value,
    patterns: Dict[str, List[str]],
    default: str = "Other",
) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    s = str(value).strip().lower()
    if s in ("", "nan", "none", "unknown", "unable to obtain", "not specified"):
        return default
    for group, pats in patterns.items():
        for p in pats:
            if p.lower() in s:
                return group
    return default


def collapse_race(value, sens_cfg: dict) -> str:
    return _match_patterns(value, sens_cfg.get("race_patterns", {}), default="Other")


def collapse_insurance(value, sens_cfg: dict) -> str:
    return _match_patterns(value, sens_cfg.get("insurance_patterns", {}), default="Other")


def collapse_fundus_race(value, sens_cfg: dict) -> str:
    g = _match_patterns(value, sens_cfg.get("race_patterns", {}), default="Other")
    allowed = set(sens_cfg.get("fundus_race_groups", ["White", "Black", "Asian"]))
    return g if g in allowed else "Other"


def normalize_sex(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip().upper()
    if s in ("M", "MALE", "1"):
        return "M"
    if s in ("F", "FEMALE", "0", "2"):
        return "F"
    return None


def bin_age(value, sens_cfg: dict) -> Optional[str]:
    try:
        a = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(a) or a < 0:
        return None
    bins = sens_cfg.get("age_bins", [0, 40, 60, 80, 200])
    labels = sens_cfg.get("age_bin_labels", ["0_40", "40_60", "60_80", "80_plus"])
    for i in range(len(labels)):
        if bins[i] <= a < bins[i + 1]:
            return labels[i]
    return None


def collapse_fst_from_scale(value) -> Optional[str]:
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    if v in (1, 2):
        return "I_II"
    if v in (3, 4):
        return "III_IV"
    if v in (5, 6):
        return "V_VI"
    return None


def collapse_fst_from_ddi_code(value) -> Optional[str]:
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return {12: "I_II", 34: "III_IV", 56: "V_VI"}.get(v, None)


def collapse_monk(value) -> Optional[str]:
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    if 1 <= v <= 3:
        return "light"
    if 4 <= v <= 7:
        return "medium"
    if 8 <= v <= 10:
        return "dark"
    return None


def attach_cxr_sensitive(
    df: pd.DataFrame,
    sens_cfg: dict,
    sex_col: str = "sex",
    age_col: str = "age",
    race_col: Optional[str] = "race",
    insurance_col: Optional[str] = "insurance",
) -> pd.DataFrame:
    df = df.copy()
    df["sex_grp"] = df[sex_col].apply(normalize_sex) if sex_col in df.columns else np.nan
    df["age_grp"] = df[age_col].apply(lambda v: bin_age(v, sens_cfg)) if age_col in df.columns else np.nan
    if race_col and race_col in df.columns:
        df["race_grp"] = df[race_col].apply(lambda v: collapse_race(v, sens_cfg))
    else:
        df["race_grp"] = np.nan
    if insurance_col and insurance_col in df.columns:
        df["insurance_grp"] = df[insurance_col].apply(lambda v: collapse_insurance(v, sens_cfg))
    else:
        df["insurance_grp"] = np.nan
    return df


CXR_SENSITIVE_COLS = ["sex_grp", "age_grp", "race_grp", "insurance_grp"]
DERM_SENSITIVE_COLS = ["fst_grp", "monk_grp", "race_grp", "sex_grp", "age_grp"]
FUNDUS_SENSITIVE_COLS = ["race_grp", "sex_grp", "ethnicity_grp", "age_grp"]

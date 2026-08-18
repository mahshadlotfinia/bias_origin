"""
data_loader/build_subgroup_counts.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import List, Optional

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_utils import read_csv_defensively
from data_loader.cxr_harmonization import CANONICAL_CXR_FINDINGS
from data_loader.sensitive_harmonization import (
    CXR_SENSITIVE_COLS, DERM_SENSITIVE_COLS, FUNDUS_SENSITIVE_COLS,
)

_OUT_COLS = ["modality", "dataset", "split", "attribute", "subgroup",
             "finding", "n_patients", "n_images", "n_positive", "prevalence"]


def _emit(df: pd.DataFrame, modality: str, attributes: List[str],
          findings: List[str]) -> pd.DataFrame:
    findings = [f for f in findings if f in df.columns]
    attributes = [a for a in attributes if a in df.columns]
    has_pid = "subject_id" in df.columns and df["subject_id"].notna().any()
    df = df.copy()
    df["__split"] = df["split"].fillna("unknown") if "split" in df.columns else "all"

    rows = []
    for dataset in df["dataset"].dropna().unique():
        d_ds = df[df["dataset"] == dataset]
        for split in d_ds["__split"].unique():
            d_sp = d_ds[d_ds["__split"] == split]
            for attr in attributes:
                for subgroup in d_sp[attr].dropna().unique():
                    d_sub = d_sp[d_sp[attr] == subgroup]
                    for finding in findings:
                        lab = pd.to_numeric(d_sub[finding], errors="coerce")
                        labeled = d_sub[lab.notna()]
                        n_images = len(labeled)
                        if n_images == 0:
                            continue
                        n_pos = int((pd.to_numeric(labeled[finding], errors="coerce") == 1.0).sum())
                        if has_pid and labeled["subject_id"].notna().any():
                            n_pat = int(labeled["subject_id"].nunique())
                        else:
                            n_pat = n_images
                        rows.append({
                            "modality": modality, "dataset": dataset, "split": split,
                            "attribute": attr, "subgroup": subgroup, "finding": finding,
                            "n_patients": n_pat, "n_images": n_images,
                            "n_positive": n_pos,
                            "prevalence": round(n_pos / n_images, 6) if n_images else np.nan,
                        })
    return pd.DataFrame(rows, columns=_OUT_COLS)


def main_build_subgroup_counts(global_config_path: str) -> str:
    cfg = read_config(global_config_path)["BiasOrigin"]
    out_csv = cfg["subgroup_counts"]["out_csv"]

    specs = [
        ("cxr",    cfg["cxr"]["pool_manifest_csv"],    CXR_SENSITIVE_COLS,    CANONICAL_CXR_FINDINGS),
        ("derm",   cfg["derm"]["pool_manifest_csv"],   DERM_SENSITIVE_COLS,   ["malignant"]),
        ("fundus", cfg["fundus"]["pool_manifest_csv"], FUNDUS_SENSITIVE_COLS, ["glaucoma"]),
    ]

    parts = []
    for modality, csv, attrs, findings in specs:
        if not os.path.exists(csv):
            print(f"[subgroup_counts] {modality}: manifest missing ({csv}); skipped.")
            continue
        df = read_csv_defensively(csv)
        part = _emit(df, modality, attrs, findings)
        print(f"[subgroup_counts] {modality}: {len(part)} count rows.")
        parts.append(part)

    if not parts:
        raise RuntimeError("[subgroup_counts] no manifests found; build the pools first.")

    counts = pd.concat(parts, ignore_index=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    counts.to_csv(out_csv, index=False)
    print(f"\n[subgroup_counts] -> {out_csv}  ({len(counts)} rows)")
    return out_csv

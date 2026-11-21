"""
data_loader/build_utils.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

from typing import List, Sequence

import numpy as np
import pandas as pd


CORE_COLUMNS: List[str] = [
    "case_id", "dataset", "modality", "split", "image_key", "image_subdir",
]


def finalize_manifest(
    df: pd.DataFrame,
    label_cols: Sequence[str] = (),
    meta_cols: Sequence[str] = (),
) -> pd.DataFrame:
    df = df.copy()
    declared = list(CORE_COLUMNS) + list(meta_cols) + list(label_cols)
    for col in declared:
        if col not in df.columns:
            df[col] = np.nan
    rest = [c for c in df.columns if c not in declared]
    return df[declared + rest]


def read_csv_defensively(path: str, **kwargs) -> pd.DataFrame:
    kwargs.setdefault("low_memory", False)
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", **kwargs)


def make_case_ids(dataset: str, keys: Sequence[str]) -> List[str]:
    out = []
    for k in keys:
        norm = str(k).strip().replace("/", "_").replace(" ", "_")
        out.append(f"{dataset}__{norm}")
    return out


def assert_unique_case_ids(df: pd.DataFrame) -> None:
    dups = df["case_id"][df["case_id"].duplicated()].unique()
    if len(dups):
        raise ValueError(
            f"{len(dups)} duplicate case_id values, e.g. {list(dups[:5])}. "
            f"Fix the per-dataset key before writing the manifest."
        )

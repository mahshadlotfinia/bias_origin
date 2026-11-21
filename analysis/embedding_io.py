"""
analysis/embedding_io.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

from typing import Tuple

import numpy as np
import pandas as pd


def load_embeddings(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    return (d["embeddings"].astype(np.float32, copy=False),
            d["case_ids"].astype(str))


def join_with_manifest(
    emb: np.ndarray,
    ids: np.ndarray,
    manifest: pd.DataFrame,
) -> Tuple[np.ndarray, pd.DataFrame]:
    id_to_row = {str(c): i for i, c in enumerate(ids)}
    keep_idx, emb_rows = [], []
    for j, cid in enumerate(manifest["case_id"].astype(str).tolist()):
        i = id_to_row.get(cid)
        if i is not None:
            keep_idx.append(j)
            emb_rows.append(i)
    if not keep_idx:
        raise ValueError("[embedding_io] no overlap between embeddings and manifest case_ids.")
    man = manifest.iloc[keep_idx].reset_index(drop=True)
    X = emb[emb_rows]
    return X, man

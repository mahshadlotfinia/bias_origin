"""
analysis/embedding_io.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_utils import read_csv_defensively


def load_embeddings(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    return (d["embeddings"].astype(np.float32, copy=False),
            d["case_ids"].astype(str))


def align_by_case_ids(
    emb_a: np.ndarray, ids_a: np.ndarray,
    emb_b: np.ndarray, ids_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    map_a = {str(c): i for i, c in enumerate(ids_a)}
    map_b = {str(c): i for i, c in enumerate(ids_b)}
    shared = sorted(map_a.keys() & map_b.keys())
    if not shared:
        raise ValueError("No shared case_ids between the two embedding files.")
    ia = [map_a[c] for c in shared]
    ib = [map_b[c] for c in shared]
    return emb_a[ia], emb_b[ib], shared


def embedding_path(cfg_path: str, encoder: str, pool: str) -> str:
    cfg = read_config(cfg_path)["BiasOrigin"]
    return os.path.join(cfg["embeddings"]["output_dir"], encoder, f"{pool}.npz")


def load_pool_embeddings(cfg_path: str, encoder: str, pool: str) -> Tuple[np.ndarray, np.ndarray]:
    npz = embedding_path(cfg_path, encoder, pool)
    if not os.path.exists(npz):
        raise FileNotFoundError(
            f"[embedding_io] embeddings not found for encoder='{encoder}', "
            f"pool='{pool}' at {npz}. Run extract_embeddings first.")
    return load_embeddings(npz)


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


def load_encoder_pool_frame(
    cfg_path: str,
    encoder: str,
    pool: str,
    manifest_csv: str,
    columns: Optional[List[str]] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    emb, ids = load_pool_embeddings(cfg_path, encoder, pool)
    man = read_csv_defensively(manifest_csv)
    if columns is not None:
        keep = ["case_id"] + [c for c in columns if c in man.columns]
        man = man[keep]
    return join_with_manifest(emb, ids, man)

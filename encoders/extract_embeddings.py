"""
encoders/extract_embeddings.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.serde import read_config
from data_loader.derm_fundus_embedding_loaders import (
    get_embedding_loader, embedding_collate_fn,
)
from encoders.image_encoders import extract_image_embeddings, list_encoder_names
from encoders.cache_utils import purge_encoders_after_stage

import warnings
warnings.filterwarnings("ignore")


def _npz_path(output_dir: str, encoder: str, pool: str) -> str:
    return os.path.join(output_dir, encoder, f"{pool}.npz")


def _is_done(npz: str, expected_ids: List[str]) -> bool:
    if not os.path.exists(npz):
        return False
    try:
        d = np.load(npz, allow_pickle=True)
        emb = d["embeddings"]
        if int(emb.shape[0]) != len(expected_ids):
            return False
        if "case_ids" in d.files:
            cached = [str(c) for c in d["case_ids"]]
            if cached != [str(c) for c in expected_ids]:
                n_diff = sum(a != b for a, b in zip(cached, expected_ids))
                same_set = set(cached) == set(map(str, expected_ids))
                print(f"[extract] {npz}: cached case_ids do not match the manifest "
                      f"({n_diff} positions differ; "
                      f"{'same rows in a different order' if same_set else 'different rows'}); "
                      f"will re-extract.")
                return False
        if emb.shape[0] > 0 and not np.isfinite(emb).all(axis=1).any():
            print(f"[extract] {npz}: cached embeddings are all non-finite "
                  f"(corrupt); will re-extract.")
            return False
        return True
    except Exception:
        return False


def _save_npz(npz: str, embeddings: np.ndarray, case_ids: List[str]) -> None:
    os.makedirs(os.path.dirname(npz), exist_ok=True)
    tmp = npz + ".tmp.npz"
    np.savez_compressed(tmp, embeddings=embeddings.astype(np.float32),
                        case_ids=np.array(case_ids, dtype=object))
    os.replace(tmp, npz)


def extract_pool(
    encoder: str,
    pool: str,
    pool_cfg: dict,
    output_dir: str,
    cfg_path: str,
    batch_size: int,
    num_workers: int,
    resolution: int,
    device: str,
) -> Optional[str]:
    manifest = pool_cfg["manifest"]
    modality = pool_cfg["modality"]
    if not os.path.exists(manifest):
        print(f"[extract] {encoder}/{pool}: manifest missing ({manifest}); skipped.")
        return None

    npz = _npz_path(output_dir, encoder, pool)
    loader_cls = get_embedding_loader(modality)
    ds = loader_cls(cfg_path, manifest, resolution=resolution, label_cols=None)
    n_expected = len(ds)
    expected_ids = [str(r.get("case_id", "")) for r in ds.records]

    if _is_done(npz, expected_ids):
        print(f"[extract] {encoder}/{pool}: done ({n_expected} rows); skipping.")
        return npz

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, collate_fn=embedding_collate_fn)

    all_emb, all_ids = [], []
    for batch in tqdm(dl, desc=f"{encoder}/{pool}", unit="batch"):
        emb = extract_image_embeddings(encoder, batch["images"], cfg_path, device)
        all_emb.append(emb)
        all_ids.extend(batch["case_ids"])

    embeddings = np.concatenate(all_emb, axis=0) if all_emb else np.zeros((0, 0), np.float32)
    _save_npz(npz, embeddings, all_ids)
    print(f"[extract] {encoder}/{pool}: wrote {embeddings.shape} -> {npz}")
    return npz


def main_extract_image_embeddings(cfg_path: str, encoder_names: Optional[List[str]] = None) -> None:
    cfg = read_config(cfg_path)["BiasOrigin"]
    emb_cfg = cfg["embeddings"]
    output_dir = emb_cfg["output_dir"]
    batch_size = int(emb_cfg.get("batch_size", 64))
    num_workers = int(emb_cfg.get("num_workers", 4))
    resolution = int(cfg.get("target_resolution", 224))
    pools = emb_cfg["pools"]
    panel = cfg["encoder_panel"]["image"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if encoder_names is None:
        encoder_names = list_encoder_names(cfg_path)

    for encoder in encoder_names:
        spec = panel.get(encoder, {})
        enc_modality = spec.get("modality", "general")
        for pool_name, pool_cfg in pools.items():
            if enc_modality not in ("general",) and enc_modality != pool_cfg["modality"]:
                continue
            extract_pool(encoder, pool_name, pool_cfg, output_dir, cfg_path,
                         batch_size, num_workers, resolution, device)
        purge_encoders_after_stage([encoder], panel, cfg_path)

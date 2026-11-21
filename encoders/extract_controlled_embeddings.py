"""
encoders/extract_controlled_embeddings.py
Created on November 20, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from config.serde import read_config
from data_loader.cxr_embedding_loader import CXREmbeddingDataset
from analysis.embedding_io import load_embeddings, join_with_manifest
from Inference import report_utils as R
from controlled.train_encoder import forward_features_cls, eval_transform, _bf16_ctx, _load_hf_offline_first


os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")


class _TransformedCXR(Dataset):
    def __init__(self, inner: CXREmbeddingDataset, transform):
        self.inner = inner
        self.transform = transform

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        item = self.inner[idx]
        return self.transform(item["image"]), item["case_id"]


def _partial_embed_path(cfg: dict, run_id: str) -> str:
    return os.path.join(cfg["embeddings"]["output_dir"], "controlled", run_id,
                        "cxr_pool.inprogress.npz")


def _load_partial_embed(path: str):
    if not os.path.exists(path):
        return None
    try:
        d = np.load(path, allow_pickle=True)
        return d["embeddings"], d["case_ids"]
    except Exception:
        return None


def _save_partial_embed(path: str, emb: np.ndarray, ids: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, embeddings=emb, case_ids=np.array(ids, dtype=object))
    os.replace(tmp, path)


def _load_hf_backbone(hf_id: str, token, device: str, timeout_s: float = 90.0):
    from transformers import AutoModel
    return _load_hf_offline_first(
        lambda hid, **kw: AutoModel.from_pretrained(hid, token=token,
                                                     trust_remote_code=True, **kw),
        hf_id, "backbone", timeout_s)


def _write_status(cfg: dict, message: str) -> None:
    path = os.path.join(cfg["embeddings"]["output_dir"], "controlled", "status.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
        f.flush()
        os.fsync(f.fileno())


def embed_run(run_id: str, cfg: dict, cfg_path: str, device: str) -> Optional[Tuple[np.ndarray, pd.DataFrame]]:
    ckpt_path = os.path.join(cfg["controlled"]["ckpts_dir"], f"{run_id}.pt")
    if not os.path.exists(ckpt_path):
        return None
    npz = os.path.join(cfg["embeddings"]["output_dir"], "controlled", run_id, "cxr_pool.npz")
    pool_csv = cfg["cxr"]["pool_manifest_csv"]

    if os.path.exists(npz):
        _write_status(cfg, f"{run_id}: checking cached embeddings")
        from data_loader.build_utils import read_csv_defensively
        man = read_csv_defensively(pool_csv)
        emb, ids = load_embeddings(npz)
        if emb.shape[0] == len(man):
            _write_status(cfg, f"{run_id}: cache complete, reused")
            return join_with_manifest(emb, ids, man)
        _write_status(cfg, f"{run_id}: cache incomplete ({emb.shape[0]}/{len(man)}), re-embedding")

    _write_status(cfg, f"{run_id}: loading checkpoint")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hf_id = ckpt.get("backbone_hf_id")
    if hf_id:
        _write_status(cfg, f"{run_id}: loading backbone {hf_id}")
        model = _load_hf_backbone(hf_id, cfg.get("hf_token"), device)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model = model.to(device).eval()
        transform = eval_transform(int(cfg.get("target_resolution", 224)))
    else:
        _write_status(cfg, f"{run_id}: loading legacy timm backbone {ckpt['backbone_arch']}")
        import timm
        from timm.data import resolve_data_config, create_transform
        model = timm.create_model(ckpt["backbone_arch"], pretrained=False, num_classes=0)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model = model.to(device).eval()
        transform = create_transform(**resolve_data_config({}, model=model))
    _write_status(cfg, f"{run_id}: backbone loaded")

    _write_status(cfg, f"{run_id}: building dataset")
    ds_inner = CXREmbeddingDataset(cfg_path, pool_csv, resolution=int(cfg.get("target_resolution", 224)))
    ds = _TransformedCXR(ds_inner, transform)
    man = pd.DataFrame(ds_inner.records)

    part_path = _partial_embed_path(cfg, run_id)
    partial = _load_partial_embed(part_path)
    all_emb, all_ids = [], []
    n_done = 0
    if partial is not None:
        pe, pi = partial
        if pe.shape[0] <= len(ds):
            all_emb.append(pe.astype(np.float32))
            all_ids.extend(list(pi))
            n_done = pe.shape[0]

    remaining = Subset(ds, list(range(n_done, len(ds)))) if n_done < len(ds) else None
    if remaining is not None and len(remaining) > 0:
        _write_status(cfg, f"{run_id}: embedding {len(remaining)} rows")
        dl = DataLoader(remaining, batch_size=int(cfg["embeddings"].get("batch_size", 64)),
                        shuffle=False, num_workers=int(cfg["embeddings"].get("num_workers", 8)),
                        pin_memory=True)
        save_every = 200
        pbar = tqdm(dl, desc=f"embed {run_id}", unit="batch",
                   initial=0, total=len(dl))
        with torch.no_grad():
            for it, (px, ids) in enumerate(pbar):
                px = px.to(device, non_blocking=True)
                with _bf16_ctx(cfg, device):
                    feats = forward_features_cls(model, px)
                feats = torch.nn.functional.normalize(feats.float(), dim=-1)
                all_emb.append(feats.cpu().numpy().astype(np.float32))
                all_ids.extend(list(ids))
                if (it + 1) % save_every == 0:
                    _save_partial_embed(part_path, np.concatenate(all_emb, axis=0), all_ids)
                    pbar.set_postfix(saved=f"{len(all_ids)}/{len(ds)}")
    emb = np.concatenate(all_emb, axis=0) if all_emb else np.zeros((0, 0), np.float32)
    os.makedirs(os.path.dirname(npz), exist_ok=True)
    np.savez_compressed(npz, embeddings=emb, case_ids=np.array(all_ids, dtype=object))
    if os.path.exists(part_path):
        os.remove(part_path)
    _write_status(cfg, f"{run_id}: embedding complete, {emb.shape[0]} rows -> {npz}")
    return join_with_manifest(emb, np.array(all_ids), man)


def main_extract_controlled_embeddings(run_id: str, global_config_path: str) -> None:
    cfg = read_config(global_config_path)["BiasOrigin"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if embed_run(run_id, cfg, global_config_path, device) is None:
        raise R.MissingInput(f"[controlled_embeddings] {run_id}: no checkpoint in {cfg['controlled']['ckpts_dir']}; train it with run_one in controlled/train_encoder.py.")

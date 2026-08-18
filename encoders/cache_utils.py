"""
encoders/cache_utils.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import gc
import os
import shutil
from typing import Optional

from config.serde import read_config


def _repo_folder_name(repo_id: str) -> str:
    try:
        from huggingface_hub import repo_folder_name
        return repo_folder_name(repo_id=repo_id, repo_type="model")
    except Exception:
        return "models--" + repo_id.replace("/", "--")


def _release_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def purge_model_cache(hf_id: str, cfg_path: str, enabled: Optional[bool] = None) -> None:
    cfg = read_config(cfg_path)["BiasOrigin"]
    emb = cfg.get("embeddings", {})
    if enabled is None:
        enabled = bool(emb.get("purge_hf_cache_after_use", False))
    if not enabled:
        return

    if hf_id is None or os.path.isdir(str(hf_id)):
        print(f"[purge] '{hf_id}' is a local path or empty; not purging.")
        return

    cache_dir = emb.get("hf_cache_dir", "") or os.environ.get(
        "HF_HUB_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"))
    folder = os.path.join(cache_dir, _repo_folder_name(hf_id))

    _release_memory()
    if not os.path.isdir(folder):
        print(f"[purge] ENABLED but no cache folder for '{hf_id}' at {folder}.")
        return
    try:
        shutil.rmtree(folder)
        print(f"[purge] removed HF weights cache for '{hf_id}' at {folder}")
    except Exception as e:
        print(f"[purge] could NOT remove '{hf_id}' cache at {folder}: "
              f"{type(e).__name__}: {e}")


def purge_encoders_after_stage(encoder_names, panel: dict, cfg_path: str) -> None:
    for name in encoder_names:
        spec = panel.get(name, {})
        hf_id = spec.get("hf_id")
        if not hf_id or hf_id in ("TXRV_BUILTIN",) or str(hf_id).endswith("_LOCAL"):
            continue
        purge_model_cache(hf_id, cfg_path)

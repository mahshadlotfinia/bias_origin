"""
data_loader/derm_fundus_build_utils.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image


def ensure_resized(src_path: str, dst_path: str, res: int = 224) -> bool:
    if dst_path and os.path.exists(dst_path):
        return True
    if not os.path.exists(src_path):
        return False
    try:
        img = Image.open(src_path).convert("RGB").resize((res, res), Image.LANCZOS)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        img.save(dst_path)
        return True
    except (OSError, ValueError):
        return False


def assign_split(
    keys: Sequence[str],
    fractions: Dict[str, float],
    seed: int = 42,
    groups: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    keys = list(keys)
    fr_train = float(fractions.get("train", 0.6))
    fr_val   = float(fractions.get("val", 0.1))
    rng = np.random.default_rng(int(seed))

    if groups is not None:
        groups = list(groups)
        assert len(groups) == len(keys)
        uniq = sorted(set(groups))
        perm = rng.permutation(len(uniq))
        uniq = [uniq[i] for i in perm]
        n = len(uniq)
        n_tr = int(round(fr_train * n))
        n_va = int(round(fr_val * n))
        split_of_group = {}
        for i, g in enumerate(uniq):
            split_of_group[g] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
        return {k: split_of_group[g] for k, g in zip(keys, groups)}

    idx = rng.permutation(len(keys))
    n = len(keys)
    n_tr = int(round(fr_train * n))
    n_va = int(round(fr_val * n))
    out: Dict[str, str] = {}
    for rank, i in enumerate(idx):
        out[keys[i]] = "train" if rank < n_tr else ("val" if rank < n_tr + n_va else "test")
    return out

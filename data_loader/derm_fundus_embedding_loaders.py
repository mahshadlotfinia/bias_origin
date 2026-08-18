"""
data_loader/derm_fundus_embedding_loaders.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Any, Dict, List, Optional

from data_loader.base_embedding_loader import BaseEmbeddingDataset, embedding_collate_fn
from data_loader.cxr_embedding_loader import CXREmbeddingDataset


class DermEmbeddingDataset(BaseEmbeddingDataset):
    def __init__(self, cfg_path, manifest_csv, resolution=224, label_cols=None):
        super().__init__(cfg_path, manifest_csv, resolution, label_cols)
        self._image_root = self.params["BiasOrigin"]["derm"]["image_root"]

    def _resolve_path(self, row: Dict[str, Any]) -> str:
        return _resolve_original(self._image_root, row, self.resolution)


class FundusEmbeddingDataset(BaseEmbeddingDataset):
    def __init__(self, cfg_path, manifest_csv, resolution=224, label_cols=None):
        super().__init__(cfg_path, manifest_csv, resolution, label_cols)
        self._image_root = self.params["BiasOrigin"]["fundus"]["image_root"]

    def _resolve_path(self, row: Dict[str, Any]) -> str:
        return _resolve_original(self._image_root, row, self.resolution)


def _resolve_original(image_root: str, row: Dict[str, Any], resolution: int) -> str:
    subdir = str(row.get("image_subdir") or "")
    key = str(row.get("image_key") or "")
    cache_224 = cache_512 = ""
    if subdir and key:
        cache_224 = os.path.join(image_root, "preprocessed224", subdir, key)
        if os.path.exists(cache_224):
            return cache_224
        cache_512 = os.path.join(image_root, "preprocessed", subdir, key)
        if os.path.exists(cache_512):
            return cache_512

    rel = str(row.get("image_relpath") or "").strip()
    if rel and os.path.exists(os.path.join(image_root, rel)):
        return os.path.join(image_root, rel)

    for d in (os.path.dirname(cache_224), os.path.dirname(cache_512),
              os.path.dirname(os.path.join(image_root, rel)) if rel else ""):
        if not d:
            continue
        for name in _equivalent_names(key):
            cand = os.path.join(d, name)
            if os.path.exists(cand):
                return cand

    if rel:
        return os.path.join(image_root, rel)
    res = "preprocessed224" if resolution == 224 else "preprocessed"
    return os.path.join(image_root, res, subdir, key)


_EXTS = (".jpg", ".png", ".jpeg", ".JPG", ".tif", ".tiff")
_DOWNSAMPLED = "_downsampled"


def _equivalent_names(key: str) -> List[str]:
    stem, _ = os.path.splitext(key)
    base = stem[: -len(_DOWNSAMPLED)] if stem.endswith(_DOWNSAMPLED) else stem
    names: List[str] = []
    for s in (base, stem, base + _DOWNSAMPLED):
        for ext in _EXTS:
            name = s + ext
            if name != key and name not in names:
                names.append(name)
    return names


LOADER_REGISTRY: Dict[str, type] = {
    "cxr":    CXREmbeddingDataset,
    "derm":   DermEmbeddingDataset,
    "fundus": FundusEmbeddingDataset,
}


def get_embedding_loader(modality: str) -> type:
    if modality not in LOADER_REGISTRY:
        raise KeyError(f"Unknown modality '{modality}'. "
                       f"Expected one of: {sorted(LOADER_REGISTRY.keys())}")
    return LOADER_REGISTRY[modality]

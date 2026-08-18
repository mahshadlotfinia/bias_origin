"""
data_loader/cxr_embedding_loader.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Any, Dict, List, Optional

from data_loader.base_embedding_loader import BaseEmbeddingDataset
from data_loader.cxr_harmonization import candidate_cxr_paths


class CXREmbeddingDataset(BaseEmbeddingDataset):
    def __init__(
        self,
        cfg_path: str,
        manifest_csv: str,
        resolution: int = 224,
        label_cols: Optional[List[str]] = None,
    ):
        super().__init__(cfg_path, manifest_csv, resolution, label_cols)
        sites = self.params["BiasOrigin"]["cxr"]["sites"]
        self._image_root = {s: scfg["image_root"] for s, scfg in sites.items()}

    def _resolve_path(self, row: Dict[str, Any]) -> str:
        site = str(row.get("site") or row.get("dataset") or "")
        root = self._image_root.get(site)
        if root is None:
            raise KeyError(f"[CXREmbeddingDataset] unknown site '{site}' for "
                           f"case_id={row.get('case_id')}")
        cands = candidate_cxr_paths(
            site, root, str(row.get("image_key", "")),
            row.get("image_subdir"), self.resolution,
        )
        for c in cands:
            if os.path.exists(c):
                return c
        return cands[0]

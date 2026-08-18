"""
data_loader/cxr_harmonization.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, List, Optional

CANONICAL_CXR_FINDINGS: List[str] = [
    "atelectasis", "cardiomegaly", "consolidation", "edema",
    "enlarged_cardiomediastinum", "fracture", "lung_lesion", "lung_opacity",
    "no_finding", "pleural_effusion", "pleural_other", "pneumonia",
    "pneumothorax", "support_devices",
]

CANONICAL_PATHOLOGY_FINDINGS: List[str] = [
    f for f in CANONICAL_CXR_FINDINGS if f != "no_finding"
]

CXR_SITES: List[str] = [
    "mimic", "chexpert", "vindr_cxr", "nih_cxr14", "padchest", "vindr_pcxr",
]

IMAGE_KEY_COL = "image_key"
IMAGE_SUBDIR_COL = "image_subdir"
VIEW_KEEP = {"PA", "AP"}


_IDENTITY = {f: f for f in CANONICAL_CXR_FINDINGS}

_NIH_ALIASES = {
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Consolidation": "consolidation",
    "Edema": "edema",
    "Effusion": "pleural_effusion",
    "Pleural_Effusion": "pleural_effusion",
    "Mass": "lung_lesion",
    "Nodule": "lung_lesion",
    "Lung Opacity": "lung_opacity",
    "Lung_Opacity": "lung_opacity",
    "Pneumonia": "pneumonia",
    "Pneumothorax": "pneumothorax",
    "Pleural_Thickening": "pleural_other",
    "No Finding": "no_finding",
    "No_Finding": "no_finding",
}

_VINDR_ALIASES = {
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Consolidation": "consolidation",
    "Pulmonary edema": "edema",
    "Lung Opacity": "lung_opacity",
    "Pleural effusion": "pleural_effusion",
    "Pleural thickening": "pleural_other",
    "Pneumonia": "pneumonia",
    "Pneumothorax": "pneumothorax",
    "Rib fracture": "fracture",
    "Clavicle fracture": "fracture",
    "Lung tumor": "lung_lesion",
    "Nodule/Mass": "lung_lesion",
    "No finding": "no_finding",
}

_PADCHEST_ALIASES = {
    "atelectasis": "atelectasis",
    "cardiomegaly": "cardiomegaly",
    "consolidation": "consolidation",
    "pulmonary edema": "edema",
    "pleural effusion": "pleural_effusion",
    "pleural thickening": "pleural_other",
    "pneumonia": "pneumonia",
    "pneumothorax": "pneumothorax",
    "fracture": "fracture",
    "nodule": "lung_lesion",
    "mass": "lung_lesion",
    "normal": "no_finding",
}

LABEL_MAPS: Dict[str, Dict[str, str]] = {
    "mimic":      dict(_IDENTITY),
    "chexpert":   dict(_IDENTITY),
    "nih_cxr14":  {**_IDENTITY, **_NIH_ALIASES},
    "vindr_cxr":  {**_IDENTITY, **_VINDR_ALIASES},
    "padchest":   {**_IDENTITY, **_PADCHEST_ALIASES},
    "vindr_pcxr": dict(_IDENTITY),
}

EXTENDED_MAPS: Dict[str, Dict[str, str]] = {
    "nih_cxr14": {
        "Infiltration": "infiltration",
        "Emphysema": "emphysema",
        "Fibrosis": "fibrosis",
        "Hernia": "hernia",
    },
    "vindr_cxr": {
        "Pulmonary fibrosis": "fibrosis",
        "Tuberculosis": "tuberculosis",
        "COPD": "copd",
    },
    "padchest": {},
    "mimic": {},
    "chexpert": {},
    "vindr_pcxr": {},
}

_CHEXPERT_POLICY = {"positive_code": 1, "negative_codes": [0, 2], "exclude_codes": [3]}
_BINARY_POLICY   = {"positive_code": 1, "negative_codes": [0], "exclude_codes": []}

LABEL_POLICY: Dict[str, Dict[str, list]] = {
    "mimic":      dict(_CHEXPERT_POLICY),
    "chexpert":   dict(_CHEXPERT_POLICY),
    "nih_cxr14":  dict(_BINARY_POLICY),
    "vindr_cxr":  dict(_BINARY_POLICY),
    "padchest":   dict(_BINARY_POLICY),
    "vindr_pcxr": dict(_BINARY_POLICY),
}


def harmonize_view(site: str, value) -> str:
    if value is None:
        return "UNKNOWN"
    v = str(value).strip().upper().replace("-", "").replace("_", "")
    if v in ("", "NAN", "NONE"):
        return "UNKNOWN"
    if v in ("PA", "POSTEROANTERIOR"):
        return "PA"
    if v in ("AP", "ANTEROPOSTERIOR", "APSUPINE", "APERECT", "APHORIZONTAL"):
        return "AP"
    if v in ("LATERAL", "LAT", "LL", "RL", "LLAT", "RLAT"):
        return "LATERAL"
    if v == "FRONTAL":
        return "PA"
    if "LATERAL" in v:
        return "LATERAL"
    if v.startswith("PA"):
        return "PA"
    if v.startswith("AP"):
        return "AP"
    return "UNKNOWN"


def _swap_ext_candidates(path: str) -> List[str]:
    stem, _ = os.path.splitext(path)
    cands = [path, stem + ".png", stem + ".jpg", stem + ".jpeg"]
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _apply_substitution(site: str, rel: str, res: int) -> str:
    sub = "preprocessed224" if res == 224 else "preprocessed"
    rel = str(rel).lstrip("/")
    if sub in rel:
        return rel
    if site == "mimic":
        if "/files/" in rel:
            return rel.replace("/files/", f"/{sub}/", 1)
        if rel.startswith("files/"):
            return rel.replace("files/", f"{sub}/", 1)
        return os.path.join(sub, rel)
    if site == "chexpert":
        if "CheXpert-v1.0/" in rel:
            return rel.replace("CheXpert-v1.0/", f"CheXpert-v1.0/{sub}/", 1)
        return os.path.join(sub, rel)
    return os.path.join(sub, rel)


def candidate_cxr_paths(
    site: str,
    image_root: str,
    image_key: str,
    image_subdir: Optional[str] = None,
    res: int = 224,
) -> List[str]:
    rel = _apply_substitution(site, image_key, res)
    subdir = "" if image_subdir in (None, "", "nan") else str(image_subdir)
    bases = []
    if subdir:
        bases.append(os.path.join(image_root, _apply_substitution(site, os.path.join(subdir, image_key), res)))
    bases.append(os.path.join(image_root, rel))
    out: List[str] = []
    for b in bases:
        out.extend(_swap_ext_candidates(b))
    seen, dedup = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    return dedup


def resolve_cxr_image_path(
    site: str,
    image_root: str,
    image_key: str,
    image_subdir: Optional[str] = None,
    res: int = 224,
) -> str:
    cands = candidate_cxr_paths(site, image_root, image_key, image_subdir, res)
    for c in cands:
        if os.path.exists(c):
            return c
    return cands[0]


def reroot_mimic_report_path(path, mimic_root: str) -> str:
    if not isinstance(path, str) or not path:
        return ""
    if os.path.exists(path):
        return path
    marker = os.sep + "MIMIC" + os.sep
    i = path.find(marker)
    if i >= 0 and mimic_root:
        return os.path.join(mimic_root, path[i + len(marker):])
    return path

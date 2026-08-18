"""
data_loader/build_isic2019_pool.py
Created on June 30, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_utils import (
    assert_unique_case_ids, finalize_manifest, make_case_ids, read_csv_defensively,
)
from data_loader.derm_fundus_build_utils import assign_split
from data_loader.sensitive_harmonization import bin_age, normalize_sex, DERM_SENSITIVE_COLS

_LABEL_COLS = ["malignant"]
_META_COLS = ["subject_id", "condition", "age_raw"] + DERM_SENSITIVE_COLS

_MALIGNANT_CLASSES = ["MEL", "BCC", "AK", "SCC"]
_BENIGN_CLASSES    = ["NV", "BKL", "DF", "VASC"]

_EXTS = (".jpg", ".png", ".jpeg", ".JPG")
_DOWNSAMPLED = "_downsampled"


def base_image_id(key: str) -> str:
    return key[: -len(_DOWNSAMPLED)] if key.endswith(_DOWNSAMPLED) else key


def candidate_filenames(key: str) -> List[str]:
    names: List[str] = []
    base = base_image_id(key)
    for stem in (base, key, base + _DOWNSAMPLED):
        for ext in _EXTS:
            fname = stem + ext
            if fname not in names:
                names.append(fname)
    return names


def find_image(key: str, roots: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    for root in roots:
        for fname in candidate_filenames(key):
            cand = os.path.join(root, fname)
            if os.path.exists(cand):
                return cand, fname
    return None, None


def _malignant_from_onehot(row, mal_cols, ben_cols) -> float:
    def _hot(col):
        try:
            return float(row.get(col, 0)) >= 0.5
        except (TypeError, ValueError):
            return False
    if any(_hot(c) for c in mal_cols):
        return 1.0
    if any(_hot(c) for c in ben_cols):
        return 0.0
    return np.nan


def _condition_from_onehot(row, all_cols) -> str:
    for c in all_cols:
        try:
            if float(row.get(c, 0)) >= 0.5:
                return c
        except (TypeError, ValueError):
            continue
    return ""


def main_build_isic2019_pool(global_config_path: str) -> str:
    cfg  = read_config(global_config_path)["BiasOrigin"]
    dcfg = cfg["derm"]
    scfg = dcfg["sources"]["isic2019"]
    if not scfg.get("enabled", True):
        print("[isic2019] disabled; skipping.")
        return dcfg["pool_manifest_csv"].replace(".csv", "_isic2019.csv")

    gt_csv = scfg["groundtruth_csv"]
    if not os.path.exists(gt_csv):
        raise FileNotFoundError(f"[isic2019] ground-truth csv not found: {gt_csv}")
    gt = read_csv_defensively(gt_csv)

    img_col = scfg.get("image_col", "image")
    mal_cols = scfg.get("malignant_classes", _MALIGNANT_CLASSES)
    ben_cols = scfg.get("benign_classes", _BENIGN_CLASSES)
    all_cols = mal_cols + ben_cols

    meta_csv = scfg.get("metadata_csv")
    meta = None
    if meta_csv and os.path.exists(meta_csv):
        meta = read_csv_defensively(meta_csv).set_index(img_col)
    else:
        print(f"[isic2019] metadata csv not found ({meta_csv}); age/sex will be NaN.")
    age_col = scfg.get("age_col", "age_approx")
    sex_col = scfg.get("sex_col", "sex")
    lesion_col = scfg.get("lesion_col", "lesion_id")

    src_root = os.path.join(scfg["dir"], scfg.get("images_subdir", "ISIC_2019_Training_Input"))
    image_root = dcfg["image_root"]
    cache_root = os.path.join(image_root, "preprocessed224", "isic2019")

    rows, n_missing = [], 0
    for _, r in gt.iterrows():
        key = str(r[img_col])
        src, stem = find_image(key, (src_root, cache_root))
        if src is None:
            n_missing += 1
            continue

        age_raw = sex_raw = lesion = np.nan
        if meta is not None and key in meta.index:
            mrow = meta.loc[key]
            age_raw = mrow.get(age_col, np.nan)
            sex_raw = mrow.get(sex_col, np.nan)
            lesion = mrow.get(lesion_col, np.nan)
        subject = f"lesion__{lesion}" if isinstance(lesion, str) and lesion.strip() \
            else f"image__{key}"

        rows.append({
            "dataset": "isic2019", "modality": "derm", "split": "",
            "image_key": stem, "image_subdir": "isic2019",
            "image_relpath": os.path.relpath(src, image_root),
            "malignant": _malignant_from_onehot(r, mal_cols, ben_cols),
            "condition": _condition_from_onehot(r, all_cols),
            "subject_id": subject,
            "age_raw": age_raw,
            "fst_grp": np.nan, "monk_grp": np.nan, "race_grp": np.nan,
            "sex_grp": normalize_sex(sex_raw),
            "age_grp": bin_age(age_raw, cfg["sensitive"]),
        })
    if n_missing:
        print(f"[isic2019] {n_missing} images not found on disk; skipped "
              f"(expected under {src_root}).")
    if not rows:
        raise RuntimeError("[isic2019] no rows produced; check images dir and csv.")

    out = pd.DataFrame(rows)
    out["case_id"] = make_case_ids("isic2019", out["image_key"].tolist())
    split_map = assign_split(out["case_id"].tolist(), dcfg["split_fractions"],
                             seed=int(cfg.get("seed", 42)),
                             groups=out["subject_id"].tolist())
    out["split"] = out["case_id"].map(split_map)

    out = finalize_manifest(out, label_cols=_LABEL_COLS, meta_cols=_META_COLS)
    assert_unique_case_ids(out)

    isic_csv = dcfg["pool_manifest_csv"].replace(".csv", "_isic2019.csv")
    os.makedirs(os.path.dirname(isic_csv), exist_ok=True)
    out.to_csv(isic_csv, index=False)
    print(f"[isic2019] {len(out)} rows -> {isic_csv}")
    print(f"    malignant pos={int(out['malignant'].sum(skipna=True))} | "
          f"sex_grp non-missing={out['sex_grp'].notna().sum()} | "
          f"age_grp non-missing={out['age_grp'].notna().sum()}")
    return isic_csv

"""
data_loader/preprocess_utils.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

from PIL import Image
from tqdm import tqdm


def _resize_one(src: str, dst_224: str, dst_512: str) -> str:
    try:
        img = Image.open(src).convert("RGB")
        for res, dst in ((224, dst_224), (512, dst_512)):
            if dst and not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                img.resize((res, res), Image.LANCZOS).save(dst)
        return "ok"
    except (OSError, ValueError) as e:
        return f"ERROR {src}: {e}"


def preprocess_manifest(
    manifest_csv: str,
    image_root: str,
    num_workers: int = 8,
    tag: str = "",
) -> None:
    import pandas as pd
    if not manifest_csv or not os.path.exists(manifest_csv):
        return
    df = pd.read_csv(manifest_csv, low_memory=False)
    needed = {"image_relpath", "image_subdir", "image_key"}
    missing_cols = needed - set(df.columns)
    if missing_cols:
        return

    jobs: List[Tuple[str, str, str]] = []
    n_no_original = 0
    self_ref: List[str] = []
    missing_src: List[str] = []
    for _, r in df.iterrows():
        rel = str(r.get("image_relpath") or "").strip()
        subdir = str(r.get("image_subdir") or "").strip()
        key = str(r.get("image_key") or "").strip()
        if not rel or not subdir or not key:
            n_no_original += 1
            continue
        src = os.path.join(image_root, rel)
        d224 = os.path.join(image_root, "preprocessed224", subdir, key)
        if os.path.exists(d224):
            continue
        if os.path.realpath(src) == os.path.realpath(d224):
            self_ref.append(d224)
            continue
        if not os.path.exists(src):
            missing_src.append(src)
            continue
        jobs.append((src, d224, ""))

    for label, bad in (("are their own cache entry and it is missing on this "
                        "machine (this source ships as the preprocessed tree, so "
                        "there is no original to resize from)", self_ref),
                       ("record an original that does not exist on this machine",
                        missing_src)):
        if not bad:
            continue
    if not jobs:
        return

    errors = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futs = {pool.submit(_resize_one, *j): j[0] for j in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), unit="img"):
            r = fut.result()
            if r.startswith("ERROR"):
                errors.append(r)

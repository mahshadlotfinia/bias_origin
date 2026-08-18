"""
data_loader/preprocess_utils.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Sequence, Tuple

from PIL import Image
from tqdm import tqdm

DEFAULT_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


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


def resize_tree(
    raw_root: str,
    out_root_224: str,
    out_root_512: str,
    exts: Sequence[str] = DEFAULT_EXTS,
    num_workers: int = 8,
    tag: str = "",
) -> None:
    if not raw_root or not os.path.isdir(raw_root):
        print(f"[preprocess{tag}] raw root absent or not set ({raw_root}); "
              f"skipping (preprocessed trees already exist).")
        return

    jobs: List[Tuple[str, str, str]] = []
    for dirpath, _, files in os.walk(raw_root):
        for fn in files:
            if not fn.lower().endswith(tuple(exts)):
                continue
            src = os.path.join(dirpath, fn)
            rel = os.path.relpath(src, raw_root)
            stem = os.path.splitext(rel)[0] + ".png"
            d224 = os.path.join(out_root_224, stem)
            d512 = os.path.join(out_root_512, stem)
            if os.path.exists(d224) and os.path.exists(d512):
                continue
            jobs.append((src, d224, d512))

    if not jobs:
        print(f"[preprocess{tag}] nothing to do under {raw_root}.")
        return

    print(f"[preprocess{tag}] resizing {len(jobs)} images under {raw_root}.")
    errors = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futs = {pool.submit(_resize_one, *j): j[0] for j in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), unit="img"):
            r = fut.result()
            if r.startswith("ERROR"):
                errors.append(r)
    print(f"[preprocess{tag}] done. errors={len(errors)}")
    for e in errors[:10]:
        print(" ", e)


def preprocess_manifest(
    manifest_csv: str,
    image_root: str,
    num_workers: int = 8,
    tag: str = "",
) -> None:
    import pandas as pd
    if not manifest_csv or not os.path.exists(manifest_csv):
        print(f"[preprocess{tag}] manifest not found ({manifest_csv}); skipping.")
        return
    df = pd.read_csv(manifest_csv, low_memory=False)
    needed = {"image_relpath", "image_subdir", "image_key"}
    missing_cols = needed - set(df.columns)
    if missing_cols:
        print(f"[preprocess{tag}] manifest is missing {missing_cols}; built "
              f"before image_relpath existed. Rebuild the pool manifest first.")
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

    if n_no_original:
        print(f"[preprocess{tag}] {n_no_original} rows had no usable "
              f"image_relpath/image_subdir/image_key; skipped.")
    for label, bad in (("are their own cache entry and it is missing on this "
                        "machine (this source ships as the preprocessed tree, so "
                        "there is no original to resize from)", self_ref),
                       ("record an original that does not exist on this machine",
                        missing_src)):
        if not bad:
            continue
        print(f"[preprocess{tag}] WARNING: {len(bad)} rows {label}. These images "
              f"cannot be read here, so extraction over this pool WILL fail on "
              f"them. Rebuild the pool manifest on THIS machine, or copy the "
              f"missing files. Examples:")
        for p in bad[:5]:
            print(f"    {p}")
    if not jobs:
        print(f"[preprocess{tag}] nothing to do for {os.path.basename(manifest_csv)}.")
        return

    print(f"[preprocess{tag}] caching {len(jobs)} images (224px) from {os.path.basename(manifest_csv)}.")
    errors = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futs = {pool.submit(_resize_one, *j): j[0] for j in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), unit="img"):
            r = fut.result()
            if r.startswith("ERROR"):
                errors.append(r)
    print(f"[preprocess{tag}] done. errors={len(errors)}")
    for e in errors[:10]:
        print(" ", e)

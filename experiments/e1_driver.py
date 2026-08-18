"""
experiments/e1_driver.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
import time
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from config.serde import read_config
from data_loader.base_embedding_loader import embedding_collate_fn
from data_loader.cxr_embedding_loader import CXREmbeddingDataset
from data_loader.cxr_harmonization import CANONICAL_PATHOLOGY_FINDINGS
from data_loader.sensitive_harmonization import CXR_SENSITIVE_COLS
from analysis.embedding_io import load_embeddings, join_with_manifest
from analysis.heads import run_head_per_finding
from analysis.fairness_metrics import compute_fairness_point
from Inference.stats_utils import (
    cluster_bootstrap, cluster_bootstrap_paired_diff, friedman_test, resolve_n_jobs,
)
from Inference import report_utils as R
from controlled.train_encoder import (
    list_controlled_runs, parse_run_id, forward_features_cls, eval_transform,
    _bf16_ctx, _with_timeout, _load_hf_offline_first,
)

import warnings
warnings.filterwarnings("ignore")

FINDINGS = CANONICAL_PATHOLOGY_FINDINGS
PRIMARY_ATTR = "race_grp"
CONTRAST_ATTRS = ["race_grp", "sex_grp", "age_grp"]
SUMMARY_KEYS = ["auroc_gap", "auroc_worst", "es_auc", "tpr_gap", "fpr_gap",
                "fnr_gap", "ece_gap"]


def _direction(metric_base: str) -> str:
    higher = {"auroc", "subgroup_auroc", "auroc_overall", "es_auc",
              "auroc_worst", "tpr", "subgroup_tpr"}
    return "higher" if metric_base in higher else "lower"


_SUB_RENAME = {"auroc": "subgroup_auroc", "tpr": "subgroup_tpr",
               "fpr": "subgroup_fpr", "fnr": "subgroup_fnr", "ece": "subgroup_ece"}


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
    except Exception as e:
        print(f"[e1] partial embedding cache at {path} unreadable ({e}); "
              f"restarting this run's embedding pass from row 0.")
        return None


def _save_partial_embed(path: str, emb: np.ndarray, ids: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, embeddings=emb, case_ids=np.array(ids, dtype=object))
    os.replace(tmp, path)


os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")


def _load_hf_backbone(hf_id: str, token, device: str, timeout_s: float = 90.0):
    from transformers import AutoModel
    return _load_hf_offline_first(
        lambda hid, **kw: AutoModel.from_pretrained(hid, token=token,
                                                     trust_remote_code=True, **kw),
        hf_id, "backbone", timeout_s)


def _write_status(cfg: dict, message: str) -> None:
    path = os.path.join(cfg["controlled"]["results_e1_dir"], "e1_status.txt")
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
        print(f"[e1] {run_id}: found cached embeddings, checking completeness...")
        _write_status(cfg, f"{run_id}: checking cached embeddings")
        from data_loader.build_utils import read_csv_defensively
        man = read_csv_defensively(pool_csv)
        emb, ids = load_embeddings(npz)
        if emb.shape[0] == len(man):
            print(f"[e1] {run_id}: cache complete ({emb.shape[0]} rows), reusing it.")
            _write_status(cfg, f"{run_id}: cache complete, reused")
            return join_with_manifest(emb, ids, man)
        print(f"[e1] {run_id}: cache incomplete ({emb.shape[0]}/{len(man)} rows); "
             f"re-embedding.")
        _write_status(cfg, f"{run_id}: cache incomplete ({emb.shape[0]}/{len(man)}), re-embedding")

    print(f"[e1] {run_id}: loading checkpoint...")
    _write_status(cfg, f"{run_id}: loading checkpoint")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hf_id = ckpt.get("backbone_hf_id")
    if hf_id:
        print(f"[e1] {run_id}: loading backbone '{hf_id}' (offline, from local cache)...")
        _write_status(cfg, f"{run_id}: loading backbone {hf_id}")
        model = _load_hf_backbone(hf_id, cfg.get("hf_token"), device)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model = model.to(device).eval()
        transform = eval_transform(int(cfg.get("target_resolution", 224)))
    else:
        print(f"[e1] {run_id}: loading legacy timm backbone '{ckpt['backbone_arch']}'...")
        _write_status(cfg, f"{run_id}: loading legacy timm backbone {ckpt['backbone_arch']}")
        import timm
        from timm.data import resolve_data_config, create_transform
        model = timm.create_model(ckpt["backbone_arch"], pretrained=False, num_classes=0)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model = model.to(device).eval()
        transform = create_transform(**resolve_data_config({}, model=model))
    print(f"[e1] {run_id}: backbone loaded.")
    _write_status(cfg, f"{run_id}: backbone loaded")

    print(f"[e1] {run_id}: reading pool manifest and building the dataset...")
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
            print(f"[e1] resuming {run_id} embedding from row {n_done}/{len(ds)} "
                 f"(partial cache found).")

    remaining = Subset(ds, list(range(n_done, len(ds)))) if n_done < len(ds) else None
    if remaining is not None and len(remaining) > 0:
        print(f"[e1] {run_id}: starting embedding loop over {len(remaining)} rows...")
        _write_status(cfg, f"{run_id}: embedding {len(remaining)} rows")
        dl = DataLoader(remaining, batch_size=int(cfg["embeddings"].get("batch_size", 64)),
                        shuffle=False, num_workers=int(cfg["embeddings"].get("num_workers", 8)),
                        pin_memory=True)
        save_every = 200
        pbar = tqdm(dl, desc=f"[e1] embed {run_id}", unit="batch",
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
    print(f"[e1] embedded {run_id}: {emb.shape} -> {npz}")
    _write_status(cfg, f"{run_id}: embedding complete, {emb.shape[0]} rows -> {npz}")
    return join_with_manifest(emb, np.array(all_ids), man)


E1_MANIFEST_COLUMNS = ["case_id", "split", "subject_id"] + list(CXR_SENSITIVE_COLS) + list(FINDINGS)

_EMBED_CACHE: Dict[str, Optional[Tuple[np.ndarray, pd.DataFrame]]] = {}


def _embed_run_cached(run_id, cfg, cfg_path, device):
    if run_id not in _EMBED_CACHE:
        _EMBED_CACHE.clear()
        res = embed_run(run_id, cfg, cfg_path, device)
        if res is not None:
            X, man = res
            keep = [c for c in E1_MANIFEST_COLUMNS if c in man.columns]
            res = (X, man[keep].copy())
        _EMBED_CACHE[run_id] = res
    return _EMBED_CACHE[run_id]


def readout_run(run_id, cfg, cfg_path, head_type, target_sens, device):
    res = _embed_run_cached(run_id, cfg, cfg_path, device)
    if res is None:
        return None
    X, man = res
    max_train = cfg["stats"].get("head_max_train", 100_000)
    per_finding = run_head_per_finding(X, man, FINDINGS, head_type=head_type,
                                       target_sensitivity=target_sens,
                                       max_train=max_train)
    return man, per_finding


def finding_frame(man, per_finding, finding, attribute) -> Optional[Tuple[pd.DataFrame, float]]:
    if finding not in per_finding:
        return None
    d = per_finding[finding]
    idx = d["test_idx"]
    pat = man["subject_id"].values[idx] if "subject_id" in man.columns else man["case_id"].values[idx]
    sub = pd.DataFrame({
        "case_id": man["case_id"].values[idx],
        "label": d["y_true"], "score": d["y_score"],
        "group": man[attribute].values[idx],
        "patient": pat,
    })
    sub = sub[sub["group"].notna() & (sub["group"].astype(str).str.lower() != "nan")].copy()
    sub["patient"] = sub["patient"].astype(str)
    sub.loc[sub["patient"].str.lower() == "nan", "patient"] = sub["case_id"]
    if sub.empty or len(sub["group"].unique()) < 2:
        return None
    return sub, float(d["threshold"])


def _panel(df: pd.DataFrame, threshold: float) -> Dict[str, float]:
    return compute_fairness_point(df["label"].values, df["score"].values,
                                  df["group"].values, threshold)


def emit_panel(perf_rows, context, boot: Dict[str, dict], n_patients: int) -> None:
    for key, b in boot.items():
        if "::" in key:
            base, sub = key.split("::", 1)
            metric_name = _SUB_RENAME.get(base, base)
            ctx = {**context, "subgroup": sub}
            R.report_metric(perf_rows, ctx, metric_name, b, _direction(base), n_patients)
        else:
            R.report_metric(perf_rows, {**context, "subgroup": np.nan}, key, b,
                            _direction(key), n_patients)


def contrast_runs(perf_rows, stat_rows, manA, pfA, manB, pfB, attribute,
                  context_base, fdr_family, n_boot, seed, n_jobs=1):
    for finding in FINDINGS:
        fa = finding_frame(manA, pfA, finding, attribute)
        fb = finding_frame(manB, pfB, finding, attribute)
        if fa is None or fb is None:
            continue
        (dfa, thrA), (dfb, thrB) = fa, fb
        merged = dfa.merge(dfb[["case_id", "score"]], on="case_id",
                           how="inner", suffixes=("_a", "_b"))
        if merged.empty or len(merged["group"].unique()) < 2:
            continue

        def fn_a(d):
            return _panel(d.rename(columns={"score_a": "score"}), thrA)

        def fn_b(d):
            return _panel(d.rename(columns={"score_b": "score"}), thrB)

        diff = cluster_bootstrap_paired_diff(merged, "patient", fn_a, fn_b,
                                             n_boot=n_boot, seed=seed, n_jobs=n_jobs)
        ctx = {**context_base, "attribute": attribute, "finding": finding}
        for key in SUMMARY_KEYS:
            if key in diff:
                R.report_paired_diff(perf_rows, stat_rows, ctx, key, diff[key],
                                     fdr_family=fdr_family,
                                     metric_direction=_direction(key),
                                     n_patients=int(merged["patient"].nunique()))


def main_e1(global_config_path: str) -> Tuple[str, str]:
    cfg = read_config(global_config_path)["BiasOrigin"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_boot = int(cfg["stats"]["n_boot"])
    seed = int(cfg["stats"]["boot_seed"])
    n_jobs = resolve_n_jobs(cfg["stats"].get("bootstrap_n_jobs", 1))
    target_sens = float(cfg["stats"]["operating_sensitivity"])
    head_types = cfg["mitigation"]["head_types"]

    runs = list_controlled_runs(global_config_path)
    cache: Dict[Tuple[str, str], Tuple] = {}

    def get_readout(run_id, head_type):
        key = (run_id, head_type)
        if key not in cache:
            cache[key] = readout_run(run_id, cfg, cfg_path=global_config_path,
                                     head_type=head_type, target_sens=target_sens,
                                     device=device)
        return cache[key]

    out_dir = cfg["controlled"]["results_e1_dir"]
    done, perf_rows, stat_rows = R.load_partial(out_dir, "e1_main")
    if done:
        n_readout_done = sum(1 for d in done if d.startswith("readout::"))
        print(f"[e1] resuming: {n_readout_done}/{len(runs)} run readouts already done, "
             f"{len(perf_rows)} perf rows and {len(stat_rows)} stat rows recovered.")

    n_cached = sum(1 for r in runs if os.path.exists(
        os.path.join(cfg["embeddings"]["output_dir"], "controlled", r, "cxr_pool.npz")))
    n_trained = sum(1 for r in runs if os.path.exists(
        os.path.join(cfg["controlled"]["ckpts_dir"], f"{r}.pt")))
    print(f"[e1] {len(runs)} controlled runs total: {n_trained} trained, "
         f"{n_cached} with a cached embedding, {n_trained - n_cached} still to embed. "
         f"Runs with no trained checkpoint are skipped (return None).")

    remaining_runs = [r for r in runs if f"readout::{r}" not in done]
    for run_id in tqdm(remaining_runs, desc="[e1] readout", unit="run",
                       initial=len(runs) - len(remaining_runs), total=len(runs)):
        meta = parse_run_id(run_id)
        for head_type in head_types:
            ro = get_readout(run_id, head_type)
            if ro is None:
                continue
            man, pf = ro
            attrs_present = [a for a in CXR_SENSITIVE_COLS
                             if a in man.columns and man[a].notna().sum() > 0]
            remaining_attrs = [a for a in attrs_present
                               if f"readout::{run_id}::{head_type}::{a}" not in done]
            for attr in tqdm(remaining_attrs, desc=f"[e1] {run_id[:40]} attrs",
                             unit="attr", leave=False):
                remaining_findings = [f for f in FINDINGS
                                      if f"readout::{run_id}::{head_type}::{attr}::{f}" not in done]
                for finding in tqdm(remaining_findings, desc=f"[e1] {attr}",
                                    unit="finding", leave=False):
                    ff = finding_frame(man, pf, finding, attr)
                    if ff is not None:
                        sub, thr = ff
                        boot = cluster_bootstrap(sub, "patient",
                                                 lambda d: _panel(d, thr),
                                                 n_boot=n_boot, seed=seed, n_jobs=n_jobs)
                        ctx = {"experiment": "e1", "modality": "cxr", "dataset": "cxr_pool",
                               "eval_dataset": "cxr_pool", "encoder": meta["objective"],
                               "encoder_objective": meta["objective"], "backbone": meta["backbone"],
                               "data_composition": meta["data_comp"], "seed": meta["seed"],
                               "head_type": head_type, "attribute": attr, "finding": finding,
                               "mitigation": "none", "operating_point": f"sens{target_sens:.2f}"}
                        emit_panel(perf_rows, ctx, boot, int(sub["patient"].nunique()))
                    done.add(f"readout::{run_id}::{head_type}::{attr}::{finding}")
                    R.save_partial(out_dir, "e1_main", done, perf_rows, stat_rows)
                done.add(f"readout::{run_id}::{head_type}::{attr}")
                R.save_partial(out_dir, "e1_main", done, perf_rows, stat_rows)
        done.add(f"readout::{run_id}")
        R.save_partial(out_dir, "e1_main", done, perf_rows, stat_rows)
        print(f"[e1] readout done: {run_id}")

    if "objective_contrast::done" in done:
        print("[e1] phase: objective contrasts already done; skipping.")
    else:
        print("[e1] phase: objective contrasts...")
        head_type = head_types[0]
        by_group: Dict[Tuple, Dict[str, str]] = {}
        for run_id in runs:
            m = parse_run_id(run_id)
            by_group.setdefault((m["backbone"], m["data_comp"], m["seed"]), {})[m["objective"]] = run_id
        for (bk, dc, sd), objruns in tqdm(list(by_group.items()), desc="[e1] objective contrast", unit="group"):
            for oa, ob in combinations(sorted(objruns), 2):
                roA, roB = get_readout(objruns[oa], head_type), get_readout(objruns[ob], head_type)
                if roA is None or roB is None:
                    continue
                ctx = {"experiment": "e1", "modality": "cxr", "dataset": "cxr_pool",
                       "eval_dataset": "cxr_pool", "encoder": oa,
                       "encoder_objective": f"{oa}_minus_{ob}", "backbone": bk,
                       "data_composition": dc, "seed": sd, "head_type": head_type,
                       "mitigation": "none", "operating_point": f"sens{target_sens:.2f}"}
                for attr in CONTRAST_ATTRS:
                    contrast_runs(perf_rows, stat_rows, *roA, *roB, attr, ctx,
                                  fdr_family=f"e1_objective_contrast::{attr}",
                                  n_boot=n_boot, seed=seed, n_jobs=n_jobs)
        done.add("objective_contrast::done")
        R.save_partial(out_dir, "e1_main", done, perf_rows, stat_rows)
    head_type = head_types[0]

    if "data_comp_contrast::done" in done:
        print("[e1] phase: data-composition contrasts already done; skipping.")
    else:
        print("[e1] phase: data-composition contrasts...")
        for run_id in tqdm(runs, desc="[e1] data-comp contrast", unit="run"):
            m = parse_run_id(run_id)
            if m["data_comp"] != "natural":
                continue
            partner = f"cxr__{m['objective']}__{m['backbone']}__{m['init']}__balanced__seed{m['seed']}"
            if partner not in runs:
                continue
            roN, roB = get_readout(run_id, head_type), get_readout(partner, head_type)
            if roN is None or roB is None:
                continue
            ctx = {"experiment": "e1", "modality": "cxr", "dataset": "cxr_pool",
                   "eval_dataset": "cxr_pool", "encoder": m["objective"],
                   "encoder_objective": m["objective"], "backbone": m["backbone"],
                   "data_composition": "natural_minus_balanced", "seed": m["seed"],
                   "head_type": head_type, "mitigation": "none",
                   "operating_point": f"sens{target_sens:.2f}"}
            for attr in CONTRAST_ATTRS:
                contrast_runs(perf_rows, stat_rows, *roN, *roB, attr, ctx,
                              fdr_family=f"e1_data_composition::{attr}", n_boot=n_boot, seed=seed, n_jobs=n_jobs)
        done.add("data_comp_contrast::done")
        R.save_partial(out_dir, "e1_main", done, perf_rows, stat_rows)

    if "scrubbed_contrast::done" in done:
        print("[e1] phase: scrubbed contrasts already done; skipping.")
    else:
        print("[e1] phase: scrubbed contrasts...")
        for run_id in tqdm(runs, desc="[e1] scrubbed contrast", unit="run"):
            m = parse_run_id(run_id)
            if m["objective"] != "image_text" or m["data_comp"] != "natural":
                continue
            partner = f"cxr__image_text__{m['backbone']}__{m['init']}__scrubbed__seed{m['seed']}"
            if partner not in runs:
                continue
            roI, roS = get_readout(run_id, head_type), get_readout(partner, head_type)
            if roI is None or roS is None:
                continue
            ctx = {"experiment": "e1", "modality": "cxr", "dataset": "cxr_pool",
                   "eval_dataset": "cxr_pool", "encoder": "image_text",
                   "encoder_objective": "image_text", "backbone": m["backbone"],
                   "data_composition": "intact_minus_scrubbed", "seed": m["seed"],
                   "head_type": head_type, "mitigation": "none",
                   "operating_point": f"sens{target_sens:.2f}"}
            for attr in CONTRAST_ATTRS:
                contrast_runs(perf_rows, stat_rows, *roI, *roS, attr, ctx,
                              fdr_family=f"e1_scrubbed::{attr}", n_boot=n_boot, seed=seed, n_jobs=n_jobs)
        done.add("scrubbed_contrast::done")
        R.save_partial(out_dir, "e1_main", done, perf_rows, stat_rows)

    if "amplified_contrast::done" in done:
        print("[e1] phase: amplified contrasts already done; skipping.")
    else:
        print("[e1] phase: amplified contrasts (positive control)...")
        for run_id in tqdm(runs, desc="[e1] amplified contrast", unit="run"):
            m = parse_run_id(run_id)
            if m["objective"] != "image_text" or m["data_comp"] != "natural":
                continue
            partner = f"cxr__image_text__{m['backbone']}__{m['init']}__amplified__seed{m['seed']}"
            if partner not in runs:
                continue
            roI, roA = get_readout(run_id, head_type), get_readout(partner, head_type)
            if roI is None or roA is None:
                continue
            ctx = {"experiment": "e1", "modality": "cxr", "dataset": "cxr_pool",
                   "eval_dataset": "cxr_pool", "encoder": "image_text",
                   "encoder_objective": "image_text", "backbone": m["backbone"],
                   "data_composition": "intact_minus_amplified", "seed": m["seed"],
                   "head_type": head_type, "mitigation": "none",
                   "operating_point": f"sens{target_sens:.2f}"}
            for attr in CONTRAST_ATTRS:
                contrast_runs(perf_rows, stat_rows, *roI, *roA, attr, ctx,
                              fdr_family=f"e1_amplified::{attr}", n_boot=n_boot,
                              seed=seed, n_jobs=n_jobs)
        done.add("amplified_contrast::done")
        R.save_partial(out_dir, "e1_main", done, perf_rows, stat_rows)

    if "friedman::done" in done:
        print("[e1] phase: seed-variance Friedman test already done; skipping.")
    else:
        print("[e1] phase: seed-variance Friedman test...")
        sv = cfg["controlled"]["seed_variance"]
        bk, dc = sv["backbone"], sv["data_comp"]
        seeds = [0] + list(sv["seeds"])
        objs = sv["objectives"]
        for attr in CONTRAST_ATTRS:
            for finding in tqdm(FINDINGS, desc=f"[e1] friedman {attr}", unit="finding"):
                per_obj_gaps = {}
                for obj in objs:
                    gaps = []
                    for sd in seeds:
                        init = cfg["controlled"]["inits"][bk]
                        rid = f"cxr__{obj}__{bk}__{init}__{dc}__seed{sd}"
                        ro = get_readout(rid, head_type)
                        if ro is None:
                            gaps.append(np.nan); continue
                        man, pf = ro
                        ff = finding_frame(man, pf, finding, attr)
                        gaps.append(_panel(ff[0], ff[1])["auroc_gap"] if ff else np.nan)
                    per_obj_gaps[obj] = gaps
                cols = [np.asarray(per_obj_gaps[o], float) for o in objs]
                if all(np.isfinite(c).sum() >= 3 for c in cols):
                    stat, p = friedman_test(*cols)
                    ctx = {"experiment": "e1", "modality": "cxr", "dataset": "cxr_pool",
                           "encoder_objective": "objective_factor", "backbone": bk,
                           "data_composition": dc, "head_type": head_type,
                           "attribute": attr, "finding": finding, "mitigation": "none"}
                    R.report_friedman(stat_rows, ctx, stat, p,
                                      fdr_family=f"e1_seed_variance_friedman::{attr}",
                                      n_units=len(seeds))
        done.add("friedman::done")
        R.save_partial(out_dir, "e1_main", done, perf_rows, stat_rows)

    R.add_fdr(stat_rows, alpha=float(cfg["stats"]["fdr_alpha"]))
    os.makedirs(out_dir, exist_ok=True)
    perf_csv = os.path.join(out_dir, "results_performance_e1.csv")
    stat_csv = os.path.join(out_dir, "results_statistics_e1.csv")
    R.write_frame_atomic(R.perf_frame(perf_rows), perf_csv)
    R.write_frame_atomic(R.stat_frame(stat_rows), stat_csv)
    R.clear_partial(out_dir, "e1_main")
    print(f"\n[e1] performance rows: {len(perf_rows)} -> {perf_csv}")
    print(f"[e1] statistics rows:  {len(stat_rows)} -> {stat_csv}")
    return perf_csv, stat_csv

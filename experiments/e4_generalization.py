"""
experiments/e4_generalization.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from config.serde import read_config
from data_loader.sensitive_harmonization import DERM_SENSITIVE_COLS, FUNDUS_SENSITIVE_COLS
from analysis.embedding_io import load_encoder_pool_frame
from Inference import report_utils as R
from experiments.e2_ceiling import run_encoder_finding

import warnings
warnings.filterwarnings("ignore")


def _encoders_for(cfg, modality_tag) -> List[str]:
    panel = cfg["encoder_panel"]["image"]
    return [name for name, spec in panel.items()
            if spec.get("modality", "general") in ("general", modality_tag)]


def _present_findings(man, candidates) -> List[str]:
    return [f for f in candidates if f in man.columns]


def _modality_spec(cfg, modality):
    gen = cfg["generalization"]
    if modality == "derm":
        return ("derm_pool", cfg.get("derm", {}).get("pool_manifest_csv"),
                DERM_SENSITIVE_COLS, gen["derm_findings"])
    if modality == "fundus":
        return ("fundus_pool", cfg.get("fundus", {}).get("pool_manifest_csv"),
                FUNDUS_SENSITIVE_COLS, gen["fundus_findings"])
    raise ValueError(f"[e4] unknown modality '{modality}'")


def run_e4_encoder(cfg, cfg_path, modality, encoder, out_dir, tag):
    pool, manifest_csv, attrs, finding_candidates = _modality_spec(cfg, modality)
    if not manifest_csv or not os.path.exists(manifest_csv):
        raise R.MissingInput(f"[e4] {modality}: manifest missing ({manifest_csv}).")
    try:
        X, man = load_encoder_pool_frame(cfg_path, encoder, pool, manifest_csv)
    except FileNotFoundError as e:
        raise R.MissingInput(f"[e4] {modality}: no embeddings for {encoder}: {e}")
    if X.shape[0] > 0 and not np.isfinite(X).all(axis=1).any():
        raise R.MissingInput(
            f"[e4] {modality}/{encoder}: {pool} embeddings are all non-finite "
            f"(corrupt cache); re-extract the embeddings before re-running E4.")
    findings = _present_findings(man, finding_candidates)
    if not findings:
        print(f"[e4] {modality}/{encoder}: no finding columns among {finding_candidates}.")
        return [], []

    units = [(attr, f) for attr in attrs if attr in man.columns and man[attr].notna().sum() > 0
             for f in findings]
    done, perf_rows, stat_rows = R.load_partial(out_dir, tag)
    if done:
        print(f"[e4] {tag}: resuming, {len(done)}/{len(units)} (attr,finding) units already done.")
    remaining = [u for u in units if f"{u[0]}::{u[1]}" not in done]
    pbar = tqdm(remaining, desc=f"[e4] {tag}", unit="unit")
    for attr, finding in pbar:
        run_encoder_finding(perf_rows, stat_rows, encoder, X, man, finding, cfg,
                            attr=attr, modality=modality, dataset=pool, experiment="e4")
        done.add(f"{attr}::{finding}")
        R.save_partial(out_dir, tag, done, perf_rows, stat_rows)
        R.heartbeat_claim(out_dir, tag)
    print(f"[e4] {modality} battery done: {encoder}")
    return perf_rows, stat_rows


def main_e4_encoder(modality: str, encoder: str, global_config_path: str,
                    force: bool = False):
    cfg = read_config(global_config_path)["BiasOrigin"]
    out_dir = cfg["generalization"]["results_e4_dir"]
    tag = f"{modality}__{encoder}"
    if not force and R.shard_exists(out_dir, tag):
        print(f"[e4] shard for {tag} exists; skipping (force=True to redo).")
        return
    stale = float(cfg["stats"].get("claim_stale_after_s", 21600))
    if not force and not R.claim_unit(out_dir, tag, stale):
        print(f"[e4] {tag} is claimed by another running job; skipping to the next.")
        return
    try:
        perf_rows, stat_rows = run_e4_encoder(cfg, global_config_path, modality, encoder, out_dir, tag)
    except R.MissingInput as e:
        print(f"{e} SKIP (no shard written; re-runs once the input exists).")
        R.release_claim(out_dir, tag)
        return
    if not perf_rows and not stat_rows:
        print(f"[e4] {tag}: produced no rows; not writing a shard.")
        R.release_claim(out_dir, tag)
        return
    R.write_shard(out_dir, tag, perf_rows, stat_rows)
    R.clear_partial(out_dir, tag)
    R.release_claim(out_dir, tag)


def main_e4_merge(global_config_path: str) -> Tuple[str, str]:
    cfg = read_config(global_config_path)["BiasOrigin"]
    return R.merge_shards(cfg["generalization"]["results_e4_dir"], "e4",
                          float(cfg["stats"]["fdr_alpha"]))


def main_e4(global_config_path: str) -> Tuple[str, str]:
    cfg = read_config(global_config_path)["BiasOrigin"]
    for modality in ("derm", "fundus"):
        for encoder in _encoders_for(cfg, modality):
            main_e4_encoder(modality, encoder, global_config_path)
    return main_e4_merge(global_config_path)

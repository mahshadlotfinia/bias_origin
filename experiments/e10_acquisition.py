"""
experiments/e10_acquisition.py
Created on August 11, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.serde import read_config
from data_loader.cxr_harmonization import CANONICAL_PATHOLOGY_FINDINGS
from analysis.embedding_io import load_encoder_pool_frame
from analysis.heads import fit_head, score_head, split_masks, threshold_at_sensitivity
from analysis import fairness_metrics as FM
from Inference.stats_utils import resolve_n_jobs
from Inference import report_utils as R
from experiments.e2_ceiling import _Eval, _emit, _e2_encoders
from experiments.e8_gapnull import simulate_null_gap, _agg

import warnings
warnings.filterwarnings("ignore")

FINDINGS = CANONICAL_PATHOLOGY_FINDINGS


def _cfg(cfg) -> Dict:
    a = cfg.get("acquisition", {})
    return {"out_dir": a["results_e10_dir"],
            "encoders": list(a.get("encoders", [])),
            "attributes": list(a.get("attributes", ["race_grp"])),
            "stratifiers": list(a.get("stratifiers", ["view"])),
            "technical": list(a.get("technical_attributes", ["view", "site"])),
            "min_stratum_n": int(a.get("min_stratum_n", 500)),
            "min_subgroup_n": int(a.get("min_subgroup_n", 20)),
            "drop_levels": {str(v).lower() for v in a.get("drop_levels", ["unknown"])}}


def _fit_cell(X, man, finding, cap, seed, target):
    if finding not in man.columns:
        return None
    y = pd.to_numeric(man[finding], errors="coerce").values
    has = np.isfinite(y) & np.isfinite(X).all(axis=1)
    tr_m, va_m, te_m = split_masks(man)
    tr = np.where(tr_m & has)[0]
    va = np.where(va_m & has)[0]
    te = np.where(te_m & has)[0]
    if min(tr.size, te.size) < 30 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        return None
    if va.size == 0:
        cut = max(1, int(0.1 * tr.size))
        sh = np.random.RandomState(seed).permutation(tr)
        va, tr = np.sort(sh[:cut]), np.sort(sh[cut:])
    idx = tr
    if cap and len(tr) > int(cap):
        idx = np.sort(np.random.RandomState(seed).choice(tr, int(cap), replace=False))
    clf, sc = fit_head("linear", X[idx], y[idx])
    return {"y": y, "te": te, "s_te": score_head(clf, sc, X[te]),
            "thr": threshold_at_sensitivity(y[va], score_head(clf, sc, X[va]), target)}


def _reference_gap(y, s, g, auroc_overall, n_sim, min_n, seed, method) -> Optional[dict]:
    if not np.isfinite(auroc_overall):
        return None
    counts = []
    for name in np.unique(g):
        m = g == name
        npos = int((y[m] == 1).sum()); nneg = int((y[m] == 0).sum())
        if npos >= 1 and nneg >= 1 and (npos + nneg) >= min_n:
            counts.append((npos, nneg))
    if len(counts) < 2:
        return None
    draws = simulate_null_gap(float(auroc_overall), counts, int(n_sim),
                              np.random.RandomState(seed), method)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return None
    return _agg(draws, len(counts))


def _panel(perf_rows, ctx, y, s, g, pat, thr, cfg, ref_cfg):
    ev = _Eval(y, s, g, pat)
    if ev.empty or len(ev.names) < 2:
        return
    boot = ev.boot(s, thr, ref_cfg["n_boot"], ref_cfg["seed"], ref_cfg["n_jobs"])
    _emit(perf_rows, ctx, boot, ev.n_patients)
    ref = _reference_gap(ev.y, ev.score(s), ev.g,
                         boot.get("auroc_overall", {}).get("point", np.nan),
                         ref_cfg["n_sim"], cfg["min_subgroup_n"], ref_cfg["seed"],
                         ref_cfg["method"])
    if ref is not None:
        perf_rows.append(R.pack_performance("gap_fair_reference", ref, "lower", ctx,
                                            n_patients=ev.n_patients, as_percent=True))


def run_e10_encoder(cfg, cfg_path, encoder, out_dir, tag):
    a = _cfg(cfg)
    seed = int(cfg["stats"]["boot_seed"])
    ref_cfg = {"n_boot": int(cfg["stats"]["n_boot"]), "seed": seed,
               "n_jobs": resolve_n_jobs(cfg["stats"].get("bootstrap_n_jobs", 1)),
               "n_sim": int(cfg["gap_null"]["n_sim"]),
               "method": str(cfg["gap_null"].get("method", "hanley"))}
    cap = cfg["stats"].get("head_max_train")
    target = float(cfg["stats"]["operating_sensitivity"])

    done, perf_rows, stat_rows = R.load_partial(out_dir, tag)
    if done:
        print(f"[e10] {tag}: resuming, {len(done)}/{len(FINDINGS)} findings done.")
    try:
        X, man = load_encoder_pool_frame(cfg_path, encoder, "cxr_pool",
                                         cfg["cxr"]["pool_manifest_csv"])
    except FileNotFoundError as e:
        raise R.MissingInput(f"[e10] no embeddings for {encoder}: {e}")
    if X.shape[0] > 0 and not np.isfinite(X).all(axis=1).any():
        raise R.MissingInput(f"[e10] {encoder}: cxr_pool embeddings are all non-finite.")
    pat_all = (man["subject_id"] if "subject_id" in man.columns else man["case_id"]).astype(str).values

    for finding in tqdm([f for f in FINDINGS if f not in done],
                        desc=f"[e10] {tag}", unit="finding"):
        cell = _fit_cell(X, man, finding, cap, seed, target)
        if cell is None:
            print(f"[e10] {tag}/{finding}: not evaluable; skipping.")
            done.add(finding)
            R.save_partial(out_dir, tag, done, perf_rows, stat_rows)
            continue
        te, s_te, y, thr = cell["te"], cell["s_te"], cell["y"], cell["thr"]
        yte, pte = y[te], pat_all[te]
        base = {"experiment": "e10", "modality": "cxr", "dataset": "cxr_pool",
                "eval_dataset": "cxr_pool", "encoder": encoder,
                "encoder_objective": encoder, "finding": finding,
                "mitigation": "none", "head_type": "linear",
                "operating_point": f"sens{target:.2f}"}

        for tech in a["technical"]:
            if tech not in man.columns:
                continue
            print(f"[e10] {encoder}/{finding}: {tech} as attribute...")
            _panel(perf_rows, {**base, "attribute": tech,
                               "data_composition": "stratum:all"},
                   yte, s_te, man[tech].values[te], pte, thr, a, ref_cfg)

        for attr in a["attributes"]:
            if attr not in man.columns:
                continue
            g_all = man[attr].values[te]
            print(f"[e10] {encoder}/{finding}: {attr} pooled...")
            _panel(perf_rows, {**base, "attribute": attr,
                               "data_composition": "stratum:all"},
                   yte, s_te, g_all, pte, thr, a, ref_cfg)
            for strat in a["stratifiers"]:
                if strat not in man.columns:
                    continue
                lv = pd.Series(man[strat].values[te]).astype(str)
                for level in sorted(lv.dropna().unique()):
                    if level.lower() in a["drop_levels"] or level.lower() == "nan":
                        continue
                    m = (lv.values == level)
                    if int(m.sum()) < a["min_stratum_n"]:
                        continue
                    print(f"[e10] {encoder}/{finding}: {attr} within {strat}={level}"
                          f" ({int(m.sum())} rows)...")
                    _panel(perf_rows,
                           {**base, "attribute": attr,
                            "data_composition": f"stratum:{strat}={level}"},
                           yte[m], s_te[m], g_all[m], pte[m], thr, a, ref_cfg)
        done.add(finding)
        R.save_partial(out_dir, tag, done, perf_rows, stat_rows)
        R.heartbeat_claim(out_dir, tag)
    print(f"[e10] encoder done: {tag}")
    return perf_rows, stat_rows


def main_e10_encoder(encoder: str, global_config_path: str, force: bool = False):
    cfg = read_config(global_config_path)["BiasOrigin"]
    a = _cfg(cfg)
    out_dir = a["out_dir"]
    tag = encoder
    if not force and R.shard_exists(out_dir, tag):
        print(f"[e10] shard for {tag} exists; skipping (force=True to redo).")
        return
    stale = float(cfg["stats"].get("claim_stale_after_s", 21600))
    if not force and not R.claim_unit(out_dir, tag, stale):
        print(f"[e10] {tag} is claimed by another running job; skipping to the next.")
        return
    try:
        perf_rows, stat_rows = run_e10_encoder(cfg, global_config_path, encoder,
                                               out_dir, tag)
    except R.MissingInput as e:
        print(f"{e} SKIP (no shard written; re-runs once the input exists).")
        R.release_claim(out_dir, tag)
        return
    if not perf_rows and not stat_rows:
        print(f"[e10] {tag}: produced no rows; not writing a shard.")
        R.release_claim(out_dir, tag)
        return
    R.write_shard(out_dir, tag, perf_rows, stat_rows)
    R.clear_partial(out_dir, tag)
    R.release_claim(out_dir, tag)


def _paired_wilcoxon(rows, ctx, a, b, family):
    from scipy.stats import wilcoxon
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    if d.size < 6 or np.allclose(d, 0):
        return
    try:
        w, p = wilcoxon(d)
    except ValueError:
        return
    R.report_wilcoxon(rows, ctx, float(w), float(p), family, n_units=int(d.size))


def _stratum_stats(cfg, perf: pd.DataFrame) -> List[Dict]:
    rows: List[Dict] = []
    need = {"metric_name", "data_composition", "attribute", "encoder", "finding"}
    if perf.empty or not need.issubset(perf.columns):
        return rows
    key = ["encoder", "finding"]
    ctx0 = {"experiment": "e10", "modality": "cxr", "dataset": "cxr_pool",
            "encoder": "panel", "encoder_objective": "panel", "mitigation": "none"}

    g = perf[perf["metric_name"] == "auroc_gap"]
    pooled = g[g["data_composition"].astype(str) == "stratum:all"]
    for attr, sub in g.groupby("attribute"):
        base = pooled[pooled["attribute"] == attr].set_index(key)["value_raw"]
        for comp, s in sub.groupby("data_composition"):
            if str(comp) == "stratum:all":
                continue
            j = base.to_frame("pooled").join(
                s.set_index(key)["value_raw"].to_frame("strat"), how="inner").dropna()
            _paired_wilcoxon(rows, {**ctx0, "attribute": attr,
                                    "finding": f"pooled_vs_{comp}",
                                    "data_composition": str(comp)},
                             j["pooled"].values, j["strat"].values,
                             f"e10_stratum::{attr}")

    ref = perf[perf["metric_name"] == "gap_fair_reference"]
    for (attr, comp), s in g.groupby(["attribute", "data_composition"]):
        r = ref[(ref["attribute"] == attr) & (ref["data_composition"] == comp)]
        j = s.set_index(key)["value_raw"].to_frame("obs").join(
            r.set_index(key)["value_raw"].to_frame("ref"), how="inner").dropna()
        _paired_wilcoxon(rows, {**ctx0, "attribute": attr,
                                "finding": f"observed_vs_fair_reference::{comp}",
                                "data_composition": str(comp)},
                         j["obs"].values, j["ref"].values,
                         f"e10_vs_reference::{attr}")
    return rows


def main_e10_merge(global_config_path: str) -> Tuple[str, str]:
    cfg = read_config(global_config_path)["BiasOrigin"]
    return R.merge_shards(
        _cfg(cfg)["out_dir"], "e10", float(cfg["stats"]["fdr_alpha"]),
        extra_stat_from_perf=lambda perf: _stratum_stats(cfg, perf))


def main_e10(global_config_path: str) -> Tuple[str, str]:
    cfg = read_config(global_config_path)["BiasOrigin"]
    encoders = _cfg(cfg)["encoders"] or _e2_encoders(cfg)
    for e in encoders:
        main_e10_encoder(e, global_config_path)
    return main_e10_merge(global_config_path)

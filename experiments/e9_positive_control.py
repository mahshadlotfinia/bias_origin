"""
experiments/e9_positive_control.py
Created on July 26, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config.serde import read_config
from analysis.embedding_io import load_encoder_pool_frame
from analysis.heads import fit_head, fit_standardizer, score_head, to_float64
from analysis.fairness_metrics import as_group_strings, auroc, subgroup_auroc
from Inference.stats_utils import spearman_with_perm
from Inference import report_utils as R
from experiments.e2_ceiling import _masks, run_encoder_finding
from mechanism.entanglement import entanglement_for

import warnings
warnings.filterwarnings("ignore")

MODES = ("decodable_only", "entangled")


def _dose_token(mode: str, strength: float) -> str:
    if mode == "entangled":
        return f"k{int(strength)}"
    return f"s{float(strength):.2f}"


def _check_dose(mode: str, strength: float) -> float:
    if mode == "entangled":
        if float(strength) < 0 or float(strength) != int(strength):
            raise ValueError(
                f"[e9] entangled dose must be a non-negative INTEGER k (the number of "
                f"disease-subspace dimensions removed), got {strength!r}. The old grid "
                f"of fractions ([0.0, 0.70, 0.85, 0.95, 1.0]) belongs to the "
                f"single-direction design and was retired on 2026-08-11.")
        return float(int(strength))
    return float(strength)


def _signed_attribute(man: pd.DataFrame, attr: str) -> np.ndarray:
    if attr not in man.columns:
        return np.zeros(len(man), float)
    g = man[attr].astype(str)
    named = g[g.str.lower() != "nan"]
    if named.empty:
        return np.zeros(len(man), float)
    major = named.value_counts().idxmax()
    a = np.where(g.values == major, 1.0, -1.0)
    a[g.str.lower().values == "nan"] = 0.0
    return a


def _disease_direction(X: np.ndarray, y: np.ndarray, tr: np.ndarray, seed: int) -> np.ndarray:
    ok = tr[np.isfinite(y[tr])]
    sc, Xs = fit_standardizer(to_float64(X, ok))
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    clf.fit(Xs, y[ok])
    del Xs
    w = clf.coef_.ravel() / np.where(sc.scale_ == 0, 1.0, sc.scale_)
    n = np.linalg.norm(w)
    return w / n if n > 0 else w


def _disease_subspace(X: np.ndarray, y: np.ndarray, tr: np.ndarray, seed: int,
                      k: int, chunk: int = 50_000) -> np.ndarray:
    if int(k) <= 0:
        return np.zeros((X.shape[1], 0), dtype=np.float64)
    ok = tr[np.isfinite(y[tr])]
    sc, B = fit_standardizer(to_float64(X, ok))
    scale = np.where(sc.scale_ == 0, 1.0, sc.scale_)
    yv = y[ok]
    cols: List[np.ndarray] = []
    for _ in tqdm(range(int(k)), desc=f"[e9] disease subspace (k={int(k)}, "
                                      f"{len(ok)} rows, {X.shape[1]}d)", unit="dir"):
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        clf.fit(B, yv)
        c = clf.coef_.ravel()
        w = c / scale
        nw = np.linalg.norm(w)
        if nw <= 0:
            break
        cols.append(w / nw)
        nc = np.linalg.norm(c)
        if nc <= 0:
            break
        v = c / nc
        for i in range(0, B.shape[0], chunk):
            blk = B[i:i + chunk]
            blk -= np.outer(blk @ v, v)
    del B
    if not cols:
        return np.zeros((X.shape[1], 0), dtype=np.float64)
    Q, _ = np.linalg.qr(np.stack(cols, axis=1))
    return np.ascontiguousarray(Q)


def inject(X: np.ndarray, a_signed: np.ndarray, direction: np.ndarray,
           strength: float, sigma: float, mode: str, chunk: int = 50_000) -> np.ndarray:
    if strength == 0.0:
        return X
    if mode not in MODES:
        raise ValueError(f"[e9] unknown injection mode '{mode}'.")
    out = np.array(X, dtype=np.float32, copy=True)
    n = out.shape[0]
    if mode == "decodable_only":
        u = np.asarray(direction, dtype=np.float32).ravel()
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            coef = (np.float32(strength * sigma)
                    * a_signed[i:j].astype(np.float32))
            out[i:j] += coef[:, None] * u[None, :]
        return out
    U = np.ascontiguousarray(np.asarray(direction, dtype=np.float32))
    if U.ndim == 1:
        U = U[:, None]
    if U.shape[1] == 0:
        return out
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        minor = (a_signed[i:j] < 0).astype(np.float32)
        blk = out[i:j]
        blk -= minor[:, None] * ((blk @ U) @ U.T)
    return out


def build_injection(X, man, finding, attr, mode, seed, k=1) -> Tuple[np.ndarray, float]:
    M = _masks(X, man, finding, attr)
    if M is None:
        return None, float("nan")
    sigma = float(np.mean(np.std(X[np.isfinite(X).all(axis=1)], axis=0)))
    if mode == "entangled":
        return _disease_subspace(X, M["y"], M["tr"], seed, int(k)), sigma
    w = _disease_direction(X, M["y"], M["tr"], seed)
    rng = np.random.RandomState(seed + 17)
    u = rng.randn(X.shape[1])
    u -= float(u @ w) * w
    n = np.linalg.norm(u)
    return (u / n if n > 0 else u), sigma


def _geometry_rows(perf_rows, X, man, finding, attr, ctx, seed):
    ent = entanglement_for(X, man, finding, attr, seed, with_decodability=False)
    if not ent:
        return
    for name, as_percent in (("geometric_overlap", False),
                             ("leace_collateral", True)):
        if name not in ent or not np.isfinite(ent[name]):
            continue
        perf_rows.append(R.pack_performance(
            name, {"point": float(ent[name]), "std": np.nan, "ci_lower": np.nan,
                   "ci_upper": np.nan, "n": 0},
            "lower", ctx, n_patients=None, as_percent=as_percent))


def run_e9_cell(cfg, cfg_path, encoder, mode, strength, out_dir, tag):
    pc = cfg["positive_control"]
    attr = pc.get("attribute", "race_grp")
    findings = list(pc["findings"])
    methods = list(pc.get("methods", []))
    seed = int(cfg["stats"]["boot_seed"])

    done, perf_rows, stat_rows = R.load_partial(out_dir, tag)
    if done:
        print(f"[e9] {tag}: resuming, {len(done)}/{len(findings)} findings done.")
    pool_csv = cfg["cxr"]["pool_manifest_csv"]
    try:
        X0, man = load_encoder_pool_frame(cfg_path, encoder, "cxr_pool", pool_csv)
    except FileNotFoundError as e:
        raise R.MissingInput(f"[e9] no embeddings for {encoder}: {e}")
    if X0.shape[0] > 0 and not np.isfinite(X0).all(axis=1).any():
        raise R.MissingInput(
            f"[e9] {encoder}: cxr_pool embeddings are all non-finite (corrupt cache); "
            f"re-extract S9 before re-running E9.")

    a_signed = _signed_attribute(man, attr)
    comp = f"inject:{mode}:{_dose_token(mode, strength)}"

    for finding in tqdm([f for f in findings if f not in done],
                        desc=f"[e9] {tag}", unit="finding"):
        direction, sigma = build_injection(X0, man, finding, attr, mode, seed,
                                           k=strength)
        if direction is None:
            print(f"[e9] {tag}/{finding}: not evaluable (too few usable rows); skipping.")
            done.add(finding)
            R.save_partial(out_dir, tag, done, perf_rows, stat_rows)
            continue
        X = inject(X0, a_signed, direction, float(strength), sigma, mode)
        ctx_extra = {"data_composition": comp, "head_type": "linear"}
        run_encoder_finding(perf_rows, stat_rows, encoder, X, man, finding, cfg,
                            attr=attr, modality="cxr", dataset="cxr_pool",
                            experiment="e9", methods=methods, extra_ctx=ctx_extra)
        _geometry_rows(perf_rows, X, man, finding, attr,
                       {"experiment": "e9", "modality": "cxr", "dataset": "cxr_pool",
                        "eval_dataset": "cxr_pool", "encoder": encoder,
                        "encoder_objective": encoder, "attribute": attr,
                        "finding": finding, "mitigation": "none", **ctx_extra},
                       seed)
        del X
        done.add(finding)
        R.save_partial(out_dir, tag, done, perf_rows, stat_rows)
        R.heartbeat_claim(out_dir, tag)
    print(f"[e9] cell done: {tag}")
    return perf_rows, stat_rows


def main_e9_cell(encoder: str, mode: str, strength: float,
                 global_config_path: str, force: bool = False):
    cfg = read_config(global_config_path)["BiasOrigin"]
    pc = cfg["positive_control"]
    if not pc.get("enabled", True):
        print("[e9] positive control disabled in config; skipping.")
        return
    if mode not in MODES:
        raise ValueError(f"[e9] unknown mode '{mode}'; expected one of {MODES}.")
    strength = _check_dose(mode, strength)
    out_dir = pc["results_e9_dir"]
    tag = f"{encoder}__{mode}__{_dose_token(mode, strength)}"
    if not force and R.shard_exists(out_dir, tag):
        print(f"[e9] shard for {tag} exists; skipping (force=True to redo).")
        return
    stale = float(cfg["stats"].get("claim_stale_after_s", 21600))
    if not force and not R.claim_unit(out_dir, tag, stale):
        print(f"[e9] {tag} is claimed by another running job; skipping to the next.")
        return
    try:
        perf_rows, stat_rows = run_e9_cell(cfg, global_config_path, encoder, mode,
                                           float(strength), out_dir, tag)
    except R.MissingInput as e:
        print(f"{e} SKIP (no shard written; re-runs once the input exists).")
        R.release_claim(out_dir, tag)
        return
    if not perf_rows and not stat_rows:
        print(f"[e9] {tag}: produced no rows; not writing a shard.")
        R.release_claim(out_dir, tag)
        return
    R.write_shard(out_dir, tag, perf_rows, stat_rows)
    R.clear_partial(out_dir, tag)
    R.release_claim(out_dir, tag)


def _parse_injection(s: str) -> Tuple[str, float]:
    try:
        _, mode, tok = str(s).split(":")
        return mode, float(tok.lstrip("sk"))
    except (ValueError, AttributeError):
        return "", float("nan")


def _dose_stats(cfg, perf: pd.DataFrame) -> List[Dict]:
    rows: List[Dict] = []
    if perf.empty:
        return rows
    df = perf.copy()
    parsed = df["data_composition"].map(_parse_injection)
    df["mode"] = [m for m, _ in parsed]
    df["strength"] = [s for _, s in parsed]
    df = df[df["mode"].isin(MODES) & np.isfinite(df["strength"])]
    if df.empty:
        return rows
    seed = int(cfg["stats"]["boot_seed"]); n_perm = int(cfg["stats"]["n_perm"])
    targets = [("auroc_gap", "none"), ("decodability", "none"),
               ("decodability_nonlinear", "none"), ("geometric_overlap", "none"),
               ("leace_collateral", "none"), ("ceiling_gap", None),
               ("gap_reduction_at_ceiling", None)]
    for (encoder, mode), sub in df.groupby(["encoder", "mode"]):
        for metric, mitig in targets:
            m = sub[sub["metric_name"] == metric]
            if mitig is not None:
                m = m[m["mitigation"].astype(str) == mitig]
            m = m.dropna(subset=["value_raw", "strength"])
            if m["strength"].nunique() < 3 or len(m) < 4:
                continue
            rho, p = spearman_with_perm(m["strength"].values.astype(float),
                                        m["value_raw"].values.astype(float),
                                        n_perm=n_perm, seed=seed)
            R.report_spearman(
                rows,
                {"experiment": "e9", "modality": "cxr", "dataset": "cxr_pool",
                 "encoder": encoder, "encoder_objective": encoder,
                 "attribute": cfg["positive_control"].get("attribute", "race_grp"),
                 "finding": f"strength_vs_{metric}", "mitigation": mode,
                 "data_composition": f"inject:{mode}"},
                rho, p, fdr_family=f"e9_dose::{mode}", n_units=int(len(m)))
    return rows


def main_e9_merge(global_config_path: str) -> Tuple[str, str]:
    cfg = read_config(global_config_path)["BiasOrigin"]
    return R.merge_shards(
        cfg["positive_control"]["results_e9_dir"], "e9",
        float(cfg["stats"]["fdr_alpha"]),
        extra_stat_from_perf=lambda perf: _dose_stats(cfg, perf))


def strengths_for(cfg, mode: str) -> List[float]:
    s = cfg["positive_control"]["strengths"]
    if isinstance(s, dict):
        return [float(v) for v in s.get(mode, [])]
    return [float(v) for v in s]


def main_e9_probe(encoder: str, global_config_path: str, mode: str = "entangled",
                  finding: Optional[str] = None):
    cfg = read_config(global_config_path)["BiasOrigin"]
    pc = cfg["positive_control"]
    attr = pc.get("attribute", "race_grp")
    finding = finding or list(pc["findings"])[0]
    seed = int(cfg["stats"]["boot_seed"])
    cap = cfg["stats"].get("head_max_train", None)
    doses = [_check_dose(mode, s) for s in strengths_for(cfg, mode)]
    if not doses:
        print(f"[e9 probe] no dose grid configured for mode '{mode}'.")
        return

    X0, man = load_encoder_pool_frame(global_config_path, encoder, "cxr_pool",
                                      cfg["cxr"]["pool_manifest_csv"])
    M = _masks(X0, man, finding, attr)
    if M is None:
        print(f"[e9 probe] {encoder}/{finding}: not evaluable.")
        return
    a_signed = _signed_attribute(man, attr)
    y, tr, te = M["y"], M["tr"], M["te"]
    tr_ok = tr[np.isfinite(y[tr])]
    if cap:
        rng = np.random.RandomState(seed)
        if len(tr_ok) > int(cap):
            tr_ok = np.sort(rng.choice(tr_ok, int(cap), replace=False))
    te_ok = te[np.isfinite(y[te])]
    minor_te = a_signed[te_ok] < 0
    major_te = a_signed[te_ok] > 0
    sigma = float(np.mean(np.std(X0[np.isfinite(X0).all(axis=1)], axis=0)))

    print(f"[e9 probe] {encoder} / {finding} / {attr} / mode={mode}: "
          f"{len(tr_ok)} train rows, {int(major_te.sum())} majority and "
          f"{int(minor_te.sum())} minority test rows.")
    if mode == "entangled":
        basis = _disease_subspace(X0, y, tr, seed, int(max(doses)))
        print(f"[e9 probe] disease subspace built at k={basis.shape[1]} "
              f"of {X0.shape[1]} dimensions.")
    else:
        basis, _ = build_injection(X0, man, finding, attr, mode, seed)
    print(f"{'dose':>8}  {'majority':>9}  {'minority':>9}  {'overall':>8}  {'gap':>7}")
    for d in doses:
        direction = basis[:, :int(d)] if mode == "entangled" else basis
        X = inject(X0, a_signed, direction, float(d), sigma, mode)
        model, sc = fit_head("linear", X[tr_ok], y[tr_ok])
        p = score_head(model, sc, X[te_ok])
        a_maj = auroc(y[te_ok][major_te], p[major_te])
        a_min = auroc(y[te_ok][minor_te], p[minor_te])
        sub = subgroup_auroc(y[te_ok], p, as_group_strings(man[attr].values[te_ok]))
        vals = [v for v in sub.values() if np.isfinite(v)]
        gap = (max(vals) - min(vals)) if len(vals) > 1 else float("nan")
        print(f"{d:>8.2f}  {100*a_maj:>9.2f}  {100*a_min:>9.2f}  "
              f"{100*auroc(y[te_ok], p):>8.2f}  {100*gap:>7.2f}")
        del X


def main_e9(global_config_path: str) -> Tuple[str, str]:
    cfg = read_config(global_config_path)["BiasOrigin"]
    pc = cfg["positive_control"]
    if not pc.get("enabled", True):
        print("[e9] positive control disabled in config; skipping.")
        return "", ""
    for encoder in pc["encoders"]:
        for mode in pc["modes"]:
            for s in strengths_for(cfg, mode):
                main_e9_cell(encoder, mode, s, global_config_path)
    return main_e9_merge(global_config_path)

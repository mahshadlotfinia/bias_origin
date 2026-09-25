"""
experiments/e2_ceiling.py
Created on June 29, 2026

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
from analysis.fairness_metrics import compute_fairness_point
from Inference.stats_utils import (
    cluster_bootstrap, cluster_bootstrap_paired_diff, cluster_bootstrap_arrays,
    cluster_bootstrap_paired_diff_arrays, spearman_with_perm, resolve_n_jobs,
)
from Inference import report_utils as R
from mitigation import preprocessing as PRE
from mitigation import inprocessing as INP
from mitigation import postprocessing as POST
from mitigation import erasure as ERA
from mitigation.ceiling import compute_ceiling, tradeoff_table

import warnings
warnings.filterwarnings("ignore")

FINDINGS = CANONICAL_PATHOLOGY_FINDINGS
ATTR = "race_grp"
SUMMARY = ["auroc_overall", "auroc_gap", "es_auc", "tpr_gap", "fpr_gap",
           "fnr_gap", "ece_gap"]

SCORE_METHODS = {"reweigh": PRE.reweighted, "resample": PRE.resampled,
                 "group_dro": INP.group_dro, "adversarial": INP.adversarial,
                 "reduction": INP.fairlearn_reduction}
POST_METHODS = {"platt_recal": POST.per_group_platt, "eo_shift": POST.per_group_threshold_shift}
ERASE_METHODS = {"leace": ERA.leace, "inlp": ERA.inlp, "rlace": ERA.rlace}


def select_methods(cfg, only=None):
    enabled = list(cfg["mitigation"]["methods_all_encoders"])
    known = set(SCORE_METHODS) | set(POST_METHODS) | set(ERASE_METHODS)
    unknown = [m for m in enabled if m not in known]
    if unknown:
        raise KeyError(
            f"[e2] mitigation.methods_all_encoders names {unknown}, which no method "
            f"table implements. Known: {sorted(known)}. Remove the name or wire it up; "
            f"a listed-but-missing method would otherwise be silently skipped.")
    if only is not None:
        enabled = [m for m in enabled if m in set(only)]
    keep = set(enabled)
    return ({k: v for k, v in SCORE_METHODS.items() if k in keep},
            {k: v for k, v in POST_METHODS.items() if k in keep},
            {k: v for k, v in ERASE_METHODS.items() if k in keep})


def e2_shard_tag(encoder: str, attr: str = ATTR) -> str:
    return encoder if attr == ATTR else f"{encoder}__{attr}"


def _direction(metric: str) -> str:
    return "higher" if metric in ("auroc_overall", "es_auc") else "lower"


def _point_boot(value: float, n: int) -> dict:
    return {"point": float(value) if np.isfinite(value) else np.nan,
            "std": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "n": int(n)}


def _eval_frame(y, score, g, pat) -> pd.DataFrame:
    df = pd.DataFrame({"label": np.asarray(y, float), "score": np.asarray(score, float),
                       "group": [str(v) for v in g], "patient": [str(v) for v in pat]})
    return df[df["group"].str.lower() != "nan"].copy()


def _panel(df, thr):
    return compute_fairness_point(df["label"].values, df["score"].values,
                                  df["group"].values, thr)


class _Eval:

    __slots__ = ("y", "g", "pat", "names", "n_patients", "keep")

    def __init__(self, y, score, g, pat):
        gs = FM.as_group_strings(g)
        self.keep = pd.Series(gs).str.lower().values != "nan"
        self.y = np.asarray(y, float)[self.keep]
        gc = FM.make_group_codes(gs[self.keep])
        self.g = gc.codes
        self.names = gc.names
        self.pat = np.asarray([str(v) for v in pat])[self.keep]
        self.n_patients = int(len(np.unique(self.pat))) if self.pat.size else 0

    def score(self, s) -> np.ndarray:
        return np.asarray(s, float)[self.keep]

    @property
    def empty(self) -> bool:
        return self.y.size == 0

    def boot(self, s, thr, n_boot, seed, n_jobs):
        return cluster_bootstrap_arrays(
            self.pat, FM.panel_point_fn(self.names, thr),
            (self.y, self.score(s), self.g), n_boot, seed, n_jobs=n_jobs)

    def boot_paired(self, s_a, s_b, thr, n_boot, seed, n_jobs):
        fn = FM.panel_point_fn(self.names, thr)
        return cluster_bootstrap_paired_diff_arrays(
            self.pat,
            lambda y, sa, sb, c: fn(y, sa, c),
            lambda y, sa, sb, c: fn(y, sb, c),
            (self.y, self.score(s_a), self.score(s_b), self.g),
            n_boot, seed, n_jobs=n_jobs)


def _emit(perf_rows, ctx, boot, n_pat):
    for k in SUMMARY:
        if k in boot:
            R.report_metric(perf_rows, ctx, k, boot[k], _direction(k), n_pat)


def _masks(X, man, finding, attr=ATTR, seed=0):
    if finding not in man.columns:
        return None
    y = pd.to_numeric(man[finding], errors="coerce").values
    race = man[attr].astype(str).values if attr in man.columns else np.array(["nan"] * len(man))
    finite = np.isfinite(X).all(axis=1)
    has = np.isfinite(y) & finite & (pd.Series(race).astype(str).str.lower().values != "nan")
    tr_m, va_m, te_m = split_masks(man)
    tr = np.where(tr_m & has)[0]
    va = np.where(va_m & has)[0]
    te = np.where(te_m & has)[0]
    if va.size == 0 and tr.size > 0:
        cut = max(1, int(0.1 * tr.size))
        tr_shuffled = np.random.RandomState(seed).permutation(tr)
        va = np.sort(tr_shuffled[:cut])
        tr = np.sort(tr_shuffled[cut:])
    if min(tr.size, te.size) < 30 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        return None
    pat = man["subject_id"].astype(str).values if "subject_id" in man.columns else man["case_id"].astype(str).values
    return dict(y=y, race=race, pat=pat, tr=tr, va=va, te=te)


def run_encoder_finding(perf_rows, stat_rows, encoder, X, man, finding, cfg,
                        attr=ATTR, modality="cxr", dataset="cxr_pool",
                        experiment="e2", methods=None, extra_ctx=None):
    score_methods, post_methods, erase_methods = select_methods(cfg, only=methods)
    extra_ctx = dict(extra_ctx or {})
    M = _masks(X, man, finding, attr)
    if M is None:
        return
    y, race, pat = M["y"], M["race"], M["pat"]
    tr, va, te = M["tr"], M["va"], M["te"]
    Xtr, Xva, Xte = X[tr], X[va], X[te]
    ytr, yva, yte = y[tr], y[va], y[te]
    gtr, gva, gte = race[tr], race[va], race[te]
    pte = pat[te]
    n_boot = int(cfg["stats"]["n_boot"]); seed = int(cfg["stats"]["boot_seed"])
    n_jobs = resolve_n_jobs(cfg["stats"].get("bootstrap_n_jobs", 1))
    target = float(cfg["stats"]["operating_sensitivity"])
    tol = float(cfg["mitigation"].get("auroc_tolerance_pts", 1.0))

    max_train = cfg["stats"].get("head_max_train")
    rng_cap = np.random.RandomState(seed)
    if max_train and len(Xtr) > int(max_train):
        cap_idx = np.sort(rng_cap.choice(len(Xtr), int(max_train), replace=False))
    else:
        cap_idx = np.arange(len(Xtr))

    base = {"experiment": experiment, "modality": modality, "dataset": dataset,
            "eval_dataset": dataset, "encoder": encoder, "attribute": attr,
            "finding": finding, "operating_point": f"sens{target:.2f}", **extra_ctx}

    print(f"[e2] {encoder}/{finding}: baseline...")
    clf, sc = fit_head("linear", Xtr[cap_idx], ytr[cap_idx])
    base_va = score_head(clf, sc, Xva); base_te = score_head(clf, sc, Xte)
    thr0 = threshold_at_sensitivity(yva, base_va, target)
    ev = _Eval(yte, base_te, gte, pte)
    if ev.empty or len(ev.names) < 2:
        return
    boot0 = ev.boot(base_te, thr0, n_boot, seed, n_jobs)
    _emit(perf_rows, {**base, "encoder_objective": encoder, "mitigation": "none"}, boot0,
          ev.n_patients)
    unmit_gap = boot0["auroc_gap"]["point"]; unmit_auroc = boot0["auroc_overall"]["point"]
    dec0 = ERA.decodability(Xtr[cap_idx], gtr[cap_idx], Xte, gte, seed)
    R.report_metric(perf_rows, {**base, "encoder_objective": encoder, "mitigation": "none"},
                    "decodability", _point_boot(dec0, len(te)), "lower", len(te))
    dec0_nl = ERA.decodability_nonlinear(Xtr[cap_idx], gtr[cap_idx], Xte, gte, seed)
    R.report_metric(perf_rows, {**base, "encoder_objective": encoder, "mitigation": "none"},
                    "decodability_nonlinear", _point_boot(dec0_nl, len(te)), "lower", len(te))

    method_points: List[Dict] = []
    te_scores_by_method: Dict[str, np.ndarray] = {}

    for name, fn in score_methods.items():
        print(f"[e2] {encoder}/{finding}: {name}...")
        try:
            mva, mte = fn(Xtr, ytr, gtr, Xva, yva, gva, Xte, gte, seed)
        except Exception as e:
            print(f"[e2] {encoder}/{finding}/{name} failed: {e}"); continue
        thr = threshold_at_sensitivity(yva, mva, target)
        boot = ev.boot(mte, thr, n_boot, seed, n_jobs)
        _emit(perf_rows, {**base, "encoder_objective": encoder, "mitigation": name}, boot,
              ev.n_patients)
        method_points.append({"name": name, "disease_auroc": boot["auroc_overall"]["point"],
                              "gap": boot["auroc_gap"]["point"]})
        te_scores_by_method[name] = mte

    for name, fn in post_methods.items():
        print(f"[e2] {encoder}/{finding}: {name}...")
        try:
            mva, mte = fn(base_va, yva, gva, base_te, gte, target, seed)
        except Exception as e:
            print(f"[e2] {encoder}/{finding}/{name} failed: {e}"); continue
        thr = threshold_at_sensitivity(yva, mva, target)
        boot = ev.boot(mte, thr, n_boot, seed, n_jobs)
        _emit(perf_rows, {**base, "encoder_objective": encoder, "mitigation": name}, boot,
              ev.n_patients)
        method_points.append({"name": name, "disease_auroc": boot["auroc_overall"]["point"],
                              "gap": boot["auroc_gap"]["point"]})
        te_scores_by_method[name] = mte

    for name, fn in erase_methods.items():
        print(f"[e2] {encoder}/{finding}: {name}...")
        Xtr_e = Xte_e = Xva_e = None
        try:
            Xtr_e, Xte_e, Xva_e = fn(Xtr, gtr, Xte, seed, X_more=Xva)
        except Exception as e:
            print(f"[e2] {encoder}/{finding}/{name} failed: {e}"); continue
        clf_e, sc_e = fit_head("linear", Xtr_e[cap_idx], ytr[cap_idx])
        mva = score_head(clf_e, sc_e, Xva_e); mte = score_head(clf_e, sc_e, Xte_e)
        thr = threshold_at_sensitivity(yva, mva, target)
        boot = ev.boot(mte, thr, n_boot, seed, n_jobs)
        _emit(perf_rows, {**base, "encoder_objective": encoder, "mitigation": name}, boot,
              ev.n_patients)
        dec_after = ERA.decodability(Xtr_e[cap_idx], gtr[cap_idx], Xte_e, gte, seed)
        R.report_metric(perf_rows, {**base, "encoder_objective": encoder, "mitigation": name},
                        "decodability", _point_boot(dec_after, len(te)), "lower", len(te))
        dec_after_nl = ERA.decodability_nonlinear(Xtr_e[cap_idx], gtr[cap_idx], Xte_e, gte, seed)
        R.report_metric(perf_rows, {**base, "encoder_objective": encoder, "mitigation": name},
                        "decodability_nonlinear", _point_boot(dec_after_nl, len(te)), "lower", len(te))
        method_points.append({"name": name, "disease_auroc": boot["auroc_overall"]["point"],
                              "gap": boot["auroc_gap"]["point"]})
        te_scores_by_method[name] = mte
    Xtr_e = Xte_e = Xva_e = None

    ceil = compute_ceiling(method_points, unmit_gap, unmit_auroc, tol)
    R.report_metric(perf_rows, {**base, "encoder_objective": encoder,
                                "mitigation": f"ceiling:{ceil['ceiling_method']}"},
                    "ceiling_gap", _point_boot(ceil["ceiling_gap"], len(te)), "lower", len(te))
    R.report_metric(perf_rows, {**base, "encoder_objective": encoder,
                                "mitigation": f"ceiling:{ceil['ceiling_method']}"},
                    "gap_reduction_at_ceiling",
                    _point_boot(ceil["gap_reduction_at_ceiling"], len(te)), "higher", len(te))

    best = ceil["ceiling_method"]
    if best in te_scores_by_method:
        diff = ev.boot_paired(base_te, te_scores_by_method[best], thr0,
                              n_boot, seed, n_jobs)
        if "auroc_gap" in diff:
            R.report_paired_diff(perf_rows, stat_rows,
                                 {**base, "encoder_objective": encoder, "mitigation": f"best:{best}"},
                                 "auroc_gap", diff["auroc_gap"],
                                 fdr_family="e2_ceiling_reduction",
                                 n_patients=ev.n_patients)

    cost, reduction = tradeoff_table(method_points, unmit_gap, unmit_auroc)
    if np.isfinite(cost).sum() >= 4:
        rho, p = spearman_with_perm(cost, reduction)
        R.report_spearman(stat_rows, {**base, "encoder_objective": encoder, "mitigation": "battery"},
                          rho, p, fdr_family="e2_tradeoff", n_units=int(np.isfinite(cost).sum()))


def _print_resample_budget(X, man, attr, tag):
    if attr not in man.columns:
        return
    finite = np.isfinite(X).all(axis=1)
    tr_m, _, _ = split_masks(man)
    g = np.asarray([str(v) for v in man[attr].values])
    ok = tr_m & finite & (pd.Series(g).astype(str).str.lower().values != "nan")
    d = int(X.shape[1])
    worst_gb, worst_f, worst_rows = 0.0, None, 0
    for f in FINDINGS:
        if f not in man.columns:
            continue
        y = pd.to_numeric(man[f], errors="coerce").values
        m = ok & np.isfinite(y)
        if m.sum() < 30:
            continue
        gg, yy = g[m], y[m]
        counts = [int(((gg == gv) & (yy == yv)).sum())
                  for gv in np.unique(gg) for yv in np.unique(yy)]
        counts = [c for c in counts if c > 0]
        if not counts:
            continue
        rows = len(counts) * max(counts)
        gb = rows * d * 8 / 1024 ** 3
        if gb > worst_gb:
            worst_gb, worst_f, worst_rows = gb, f, rows
    if worst_f is None:
        return
    resident = 2 * X.nbytes / 1024 ** 3
    need = 2.13 * worst_gb + resident + 2.0
    print(f"[e2] {tag}: memory budget. The largest array this cell builds is the "
          f"oversampled training block for {worst_f}, {worst_rows} rows x {d} dims "
          f"in float64 = {worst_gb:.1f} GB, and standardizing it briefly holds a "
          f"second copy, so that step peaks near {2.13 * worst_gb:.1f} GB. Up to "
          f"{resident:.1f} GB more is held throughout by the cached embeddings and "
          f"their per-split copies. Give this job at least {need:.0f} GB.")


def _e2_encoders(cfg) -> List[str]:
    panel = cfg["encoder_panel"]["image"]
    return [name for name, spec in panel.items()
            if spec.get("modality", "general") in ("general", "cxr")]


def run_e2_encoder(cfg, cfg_path, encoder, pool_csv, out_dir, attr=ATTR):
    tag = e2_shard_tag(encoder, attr)
    done, perf_rows, stat_rows = R.load_partial(out_dir, tag)
    if done:
        print(f"[e2] {tag}: resuming, {len(done)}/{len(FINDINGS)} findings "
             f"already done.")
    try:
        X, man = load_encoder_pool_frame(cfg_path, encoder, "cxr_pool", pool_csv)
    except FileNotFoundError as e:
        raise R.MissingInput(f"[e2] no embeddings for {encoder}: {e}")
    if X.shape[0] > 0 and not np.isfinite(X).all(axis=1).any():
        raise R.MissingInput(
            f"[e2] {encoder}: cxr_pool embeddings are all non-finite (corrupt cache); "
            f"re-extract the embeddings of this encoder before re-running E2.")
    _print_resample_budget(X, man, attr, tag)
    remaining = [f for f in FINDINGS if f not in done]
    pbar = tqdm(remaining, desc=f"[e2] {tag}", unit="finding")
    for finding in pbar:
        run_encoder_finding(perf_rows, stat_rows, encoder, X, man, finding, cfg, attr=attr)
        done.add(finding)
        R.save_partial(out_dir, tag, done, perf_rows, stat_rows)
        R.heartbeat_claim(out_dir, tag)
    print(f"[e2] battery done: {tag}")
    return perf_rows, stat_rows


def main_e2_encoder(encoder: str, global_config_path: str, attribute: str = ATTR,
                    force: bool = False):
    cfg = read_config(global_config_path)["BiasOrigin"]
    out_dir = cfg["mitigation"]["results_e2_dir"]
    tag = e2_shard_tag(encoder, attribute)
    if not force and R.shard_exists(out_dir, tag):
        print(f"[e2] shard for {tag} exists; skipping (force=True to redo).")
        return
    stale = float(cfg["stats"].get("claim_stale_after_s", 21600))
    if not force and not R.claim_unit(out_dir, tag, stale):
        print(f"[e2] {tag} is claimed by another running job; skipping to the next.")
        return
    pool_csv = cfg["cxr"]["pool_manifest_csv"]
    try:
        perf_rows, stat_rows = run_e2_encoder(cfg, global_config_path, encoder,
                                              pool_csv, out_dir, attr=attribute)
    except R.MissingInput as e:
        print(f"{e} SKIP (no shard written; re-runs once the input exists).")
        R.release_claim(out_dir, tag)
        return
    if not perf_rows and not stat_rows:
        print(f"[e2] {tag}: produced no rows; not writing a shard.")
        R.release_claim(out_dir, tag)
        return
    R.write_shard(out_dir, tag, perf_rows, stat_rows)
    R.clear_partial(out_dir, tag)
    R.release_claim(out_dir, tag)


def main_e2_merge(global_config_path: str) -> Tuple[str, str]:
    cfg = read_config(global_config_path)["BiasOrigin"]
    return R.merge_shards(cfg["mitigation"]["results_e2_dir"], "e2",
                          float(cfg["stats"]["fdr_alpha"]))


def main_e2(global_config_path: str) -> Tuple[str, str]:
    cfg = read_config(global_config_path)["BiasOrigin"]
    for attr in cfg["mitigation"].get("attributes", [ATTR]):
        for encoder in _e2_encoders(cfg):
            main_e2_encoder(encoder, global_config_path, attribute=attr)
    return main_e2_merge(global_config_path)

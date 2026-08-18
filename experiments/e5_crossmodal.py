"""
experiments/e5_crossmodal.py
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
from analysis.heads import fit_head, score_head, threshold_at_sensitivity
from Inference.stats_utils import (
    cluster_bootstrap, cluster_bootstrap_paired_diff, resolve_n_jobs,
)
from Inference import report_utils as R
from mitigation.optimal_transport import fairclip_ot_with_groups
from mechanism.entanglement import entanglement_for, prepare_pool, transfer_r2
from experiments.e2_ceiling import _masks, _eval_frame, _panel, _emit
from experiments.e3_mechanism import _rows_from_perf

import warnings
warnings.filterwarnings("ignore")

FINDINGS = CANONICAL_PATHOLOGY_FINDINGS
PRIMARY = {"cxr": "race_grp", "derm": "fst_grp", "fundus": "race_grp"}


def _image_text_encoders(cfg) -> List[str]:
    panel = cfg["encoder_panel"]["image"]
    return [name for name, spec in panel.items()
            if spec.get("objective") == "image_text"
            and spec.get("modality", "general") in ("general", "cxr")]


def _part1_fairclip_ot(perf_rows, stat_rows, cfg, cfg_path, out_dir, done):
    pool_csv = cfg["cxr"]["pool_manifest_csv"]
    attr = PRIMARY["cxr"]
    n_boot = int(cfg["stats"]["n_boot"]); seed = int(cfg["stats"]["boot_seed"])
    n_jobs = resolve_n_jobs(cfg["stats"].get("bootstrap_n_jobs", 1))
    target = float(cfg["stats"]["operating_sensitivity"])

    encoders = _image_text_encoders(cfg)
    remaining = [e for e in encoders if f"part1::{e}" not in done]
    pbar = tqdm(remaining, desc="[e5] FairCLIP-OT", unit="encoder")
    for encoder in pbar:
        try:
            X, man = load_encoder_pool_frame(cfg_path, encoder, "cxr_pool", pool_csv)
        except FileNotFoundError:
            print(f"[e5] no embeddings for {encoder}; skipping (not marked done).")
            continue
        for finding in FINDINGS:
            M = _masks(X, man, finding, attr)
            if M is None:
                continue
            y, g, pat = M["y"], M["race"], M["pat"]
            tr, va, te = M["tr"], M["va"], M["te"]
            Xtr, Xva, Xte = X[tr], X[va], X[te]
            ytr, yva, yte = y[tr], y[va], y[te]
            gtr, gva, gte = g[tr], g[va], g[te]
            pte = pat[te]
            base = {"experiment": "e5", "modality": "cxr", "dataset": "cxr_pool",
                    "eval_dataset": "cxr_pool", "encoder": encoder,
                    "encoder_objective": "image_text", "attribute": attr,
                    "finding": finding, "operating_point": f"sens{target:.2f}"}

            clf, sc = fit_head("linear", Xtr, ytr)
            base_va = score_head(clf, sc, Xva); base_te = score_head(clf, sc, Xte)
            thr0 = threshold_at_sensitivity(yva, base_va, target)
            df0 = _eval_frame(yte, base_te, gte, pte)
            if df0.empty or len(df0["group"].unique()) < 2:
                continue
            boot0 = cluster_bootstrap(df0, "patient", lambda d: _panel(d, thr0), n_boot, seed, n_jobs=n_jobs)
            _emit(perf_rows, {**base, "mitigation": "none"}, boot0, int(df0["patient"].nunique()))

            try:
                Xtr_o, Xte_o = fairclip_ot_with_groups(Xtr, gtr, Xte, gte, seed)
                Xva_o = fairclip_ot_with_groups(Xtr, gtr, Xva, gva, seed)[1]
            except Exception as e:
                print(f"[e5] {encoder}/{finding} OT failed: {e}"); continue
            clf2, sc2 = fit_head("linear", Xtr_o, ytr)
            va_o = score_head(clf2, sc2, Xva_o); te_o = score_head(clf2, sc2, Xte_o)
            thr_o = threshold_at_sensitivity(yva, va_o, target)
            df_o = _eval_frame(yte, te_o, gte, pte)
            boot_o = cluster_bootstrap(df_o, "patient", lambda d: _panel(d, thr_o), n_boot, seed, n_jobs=n_jobs)
            _emit(perf_rows, {**base, "mitigation": "fairclip_ot"}, boot_o, int(df_o["patient"].nunique()))

            dfp = pd.DataFrame({"label": yte, "score_base": base_te, "score_ot": te_o,
                                "group": [str(v) for v in gte], "patient": [str(v) for v in pte]})
            dfp = dfp[dfp["group"].str.lower() != "nan"]
            diff = cluster_bootstrap_paired_diff(
                dfp, "patient",
                lambda d: _panel(d.rename(columns={"score_base": "score"}), thr0),
                lambda d: _panel(d.rename(columns={"score_ot": "score"}), thr0),
                n_boot, seed, n_jobs=n_jobs)
            if "auroc_gap" in diff:
                R.report_paired_diff(perf_rows, stat_rows, {**base, "mitigation": "fairclip_ot"},
                                     "auroc_gap", diff["auroc_gap"],
                                     fdr_family="e5_fairclip_ot",
                                     n_patients=int(dfp["patient"].nunique()))
        print(f"[e5] FairCLIP-OT done: {encoder}")
        done.add(f"part1::{encoder}")
        R.save_partial(out_dir, "e5_main", done, perf_rows, stat_rows)


def _ceiling_map(perf_csv, attr) -> Dict[Tuple[str, str], float]:
    if not os.path.exists(perf_csv):
        return {}
    df = pd.read_csv(perf_csv, float_precision="round_trip")
    cg = df[(df["metric_name"] == "ceiling_gap") & (df["attribute"] == attr)]
    return {(str(r["encoder"]), str(r["finding"])): float(r["value_raw"])
            for _, r in cg.iterrows()}


def _ent_table(cfg, cfg_path, encoders, pool, manifest, attr, findings, ceiling, seed, desc=""):
    rows = []
    for enc in tqdm(encoders, desc=f"[e5] entanglement {desc}", unit="encoder"):
        try:
            X, man = load_encoder_pool_frame(cfg_path, enc, pool, manifest)
        except FileNotFoundError:
            continue
        prep = prepare_pool(X, man, attr)
        for f in findings:
            ent = entanglement_for(X, man, f, attr, seed, prep=prep,
                                   with_decodability=False)
            if ent is None:
                continue
            tgt = ceiling.get((enc, f), np.nan)
            if not np.isfinite(tgt):
                continue
            rows.append({"leace_collateral": ent["leace_collateral"],
                         "geometric_overlap": ent["geometric_overlap"],
                         "ceiling_gap": tgt})
    return pd.DataFrame(rows)


def _encoders_for(cfg, modality_tag):
    panel = cfg["encoder_panel"]["image"]
    return [n for n, s in panel.items()
            if s.get("modality", "general") in ("general", modality_tag)]


def _load_e3_cxr_table(e3_csv: str) -> pd.DataFrame:
    df = pd.read_csv(e3_csv, float_precision="round_trip")
    rows = _rows_from_perf(df.to_dict("records"))
    return pd.DataFrame(rows)


def _part2_transfer(perf_rows, stat_rows, cfg, cfg_path, out_dir, done):
    if "part2::done" in done:
        print("[e5] part 2 (cross-modal transfer) already completed; skipping.")
        return
    gen = cfg["generalization"]
    e3_csv = os.path.join(cfg["mechanism"]["results_e3_dir"], "results_performance_e3.csv")
    e4_csv = os.path.join(gen["results_e4_dir"], "results_performance_e4.csv")
    if not os.path.exists(e3_csv) or not os.path.exists(e4_csv):
        print("[e5] part 2 needs E3 and E4 CSVs; skipping transfer.")
        return
    seed = int(cfg["stats"]["boot_seed"]); n_perm = int(cfg["mechanism"]["n_perm"])

    cxr = _load_e3_cxr_table(e3_csv)
    if len(cxr) < 6:
        print("[e5] too few CXR rows for a transfer source; skipping.")
        return
    F_tr = cxr[["leace_collateral", "geometric_overlap"]].values
    y_tr = cxr["ceiling_gap"].values

    targets = [("derm", "derm_pool", cfg.get("derm", {}).get("pool_manifest_csv"),
                PRIMARY["derm"], gen["derm_findings"]),
               ("fundus", "fundus_pool", cfg.get("fundus", {}).get("pool_manifest_csv"),
                PRIMARY["fundus"], gen["fundus_findings"])]
    rng = np.random.RandomState(seed)
    for name, pool, manifest, attr, findings in targets:
        if not manifest or not os.path.exists(manifest):
            continue
        tab = _ent_table(cfg, cfg_path, _encoders_for(cfg, name), pool, manifest,
                         attr, findings, _ceiling_map(e4_csv, attr), seed, desc=f"{name}_target")
        if len(tab) < 3:
            print(f"[e5] too few {name} rows for transfer target; skipping.")
            continue
        F_te = tab[["leace_collateral", "geometric_overlap"]].values
        y_te = tab["ceiling_gap"].values
        r2 = transfer_r2(F_tr, y_tr, F_te, y_te)
        if np.isfinite(r2):
            cnt = sum(1 for _ in range(n_perm)
                      if transfer_r2(F_tr, y_tr, F_te, rng.permutation(y_te)) >= r2)
            p = (cnt + 1) / (n_perm + 1)
        else:
            p = float("nan")
        stat_rows.append(R.pack_statistic(
            estimate=r2, estimate_name=f"transfer_r2_cxr_to_{name}",
            test_name="transfer_permutation", p_raw=p, fdr_family="e5_transfer",
            context={"experiment": "e5", "modality": name, "dataset": pool,
                     "attribute": attr, "mitigation": "battery"},
            n_units=len(tab)))
        print(f"[e5] transfer cxr->{name}: R2={r2:.3f} p={p:.4f} (n={len(tab)})")
    done.add("part2::done")
    R.save_partial(out_dir, "e5_main", done, perf_rows, stat_rows)


def main_e5(global_config_path: str) -> Tuple[str, str]:
    cfg = read_config(global_config_path)["BiasOrigin"]
    out_dir = cfg["generalization"]["results_e5_dir"]
    done, perf_rows, stat_rows = R.load_partial(out_dir, "e5_main")
    if done:
        print(f"[e5] resuming with {len(done)} units already done.")

    _part1_fairclip_ot(perf_rows, stat_rows, cfg, global_config_path, out_dir, done)
    _part2_transfer(perf_rows, stat_rows, cfg, global_config_path, out_dir, done)

    R.add_fdr(stat_rows, alpha=float(cfg["stats"]["fdr_alpha"]))
    os.makedirs(out_dir, exist_ok=True)
    perf_csv = os.path.join(out_dir, "results_performance_e5.csv")
    stat_csv = os.path.join(out_dir, "results_statistics_e5.csv")
    R.write_frame_atomic(R.perf_frame(perf_rows), perf_csv)
    R.write_frame_atomic(R.stat_frame(stat_rows), stat_csv)
    R.clear_partial(out_dir, "e5_main")
    print(f"\n[e5] performance rows: {len(perf_rows)} -> {perf_csv}")
    print(f"[e5] statistics rows:  {len(stat_rows)} -> {stat_csv}")
    return perf_csv, stat_csv

"""
experiments/e3_mechanism.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.serde import read_config
from data_loader.cxr_harmonization import CANONICAL_PATHOLOGY_FINDINGS
from data_loader.build_utils import read_csv_defensively
from analysis.embedding_io import load_encoder_pool_frame, load_embeddings, join_with_manifest
from Inference.stats_utils import spearman_with_perm
from Inference import report_utils as R
from mechanism.entanglement import (
    entanglement_for, prepare_pool, format_timings, cross_fit_r2, logo_r2,
    permutation_r2_null, transfer_r2,
)
from experiments.e2_ceiling import _e2_encoders

import warnings
warnings.filterwarnings("ignore")

FINDINGS = CANONICAL_PATHOLOGY_FINDINGS
ATTR = "race_grp"

E3_MANIFEST_COLUMNS = ["split", ATTR] + list(FINDINGS)


def _point(value, n) -> dict:
    return {"point": float(value) if np.isfinite(value) else np.nan,
            "std": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "n": int(n)}


def _emit_point(perf_rows, ctx, metric, value, direction, n, as_percent=True):
    perf_rows.append(R.pack_performance(metric, _point(value, n), direction, ctx,
                                        n_patients=n, as_percent=as_percent))


def _load_ceiling_target(e2_perf_csv: str, attr: str = ATTR) -> Tuple[Dict, Dict]:
    df = pd.read_csv(e2_perf_csv, float_precision="round_trip")
    if "attribute" in df.columns:
        df = df[df["attribute"].astype(str) == attr]
    if df.empty:
        raise ValueError(
            f"[e3] no E2 rows for attribute '{attr}' in {e2_perf_csv}. E3's "
            f"mechanism analysis is defined against one attribute at a time.")
    ceil, unmit = {}, {}
    cg = df[df["metric_name"] == "ceiling_gap"]
    for _, r in cg.iterrows():
        ceil[(str(r["encoder"]), str(r["finding"]))] = float(r["value_raw"])
    ug = df[(df["metric_name"] == "auroc_gap") & (df["mitigation"] == "none")]
    for _, r in ug.iterrows():
        unmit[(str(r["encoder"]), str(r["finding"]))] = float(r["value_raw"])
    return ceil, unmit


def _controlled_run_ids(cfg) -> List[str]:
    c = cfg["controlled"]
    inits = c["inits"]
    runs = []
    for obj in c["objectives"]:
        for bk in c["backbones"]:
            for dc in c["data_compositions"]:
                runs.append(f"cxr__{obj}__{bk}__{inits[bk]}__{dc}__seed0")
    for bk in c["backbones"]:
        runs.append(f"cxr__{c['scrubbed_objective']}__{bk}__{inits[bk]}__scrubbed__seed0")
    if c.get("run_amplified", True):
        for bk in c["backbones"]:
            runs.append(f"cxr__{c['amplified_objective']}__{bk}__{inits[bk]}__amplified__seed0")
    sv = c["seed_variance"]
    for obj in sv["objectives"]:
        for sd in sv["seeds"]:
            runs.append(f"cxr__{obj}__{sv['backbone']}__{inits[sv['backbone']]}__{sv['data_comp']}__seed{sd}")
    return runs


def _parse_rid(rid: str) -> Dict:
    _, obj, bk, _init, dc, sd = rid.split("__")
    return {"objective": obj, "backbone": bk, "data_comp": dc,
            "seed": int(sd.replace("seed", ""))}


def _e1_unmit_gaps(cfg, e1_csv: str, attr: str) -> Dict:
    if not os.path.exists(e1_csv):
        return {}
    valid_dc = {_parse_rid(r)["data_comp"] for r in _controlled_run_ids(cfg)}
    df = pd.read_csv(e1_csv, float_precision="round_trip")
    sub = df[(df["metric_name"] == "auroc_gap") & (df["mitigation"] == "none")
             & (df["attribute"] == attr) & (df["data_composition"].isin(valid_dc))]
    out = {}
    for _, r in sub.iterrows():
        try:
            out[(str(r["encoder_objective"]), str(r["backbone"]),
                 str(r["data_composition"]), int(r["seed"]), str(r["finding"]))] = float(r["value_raw"])
        except (ValueError, TypeError):
            continue
    return out


BRIDGE_TAG = "e3_scale_bridge"


def _bridge_cell(rid: str, finding: str) -> str:
    return f"{rid}::{finding}"


def _bridge_tag(rid: str) -> str:
    return f"{BRIDGE_TAG}__{rid}"


def _load_bridge_cells(out_dir: str, run_ids: List[str]) -> Dict[str, Dict]:
    by_cell: Dict[str, Dict] = {}
    for tag in [BRIDGE_TAG] + [_bridge_tag(r) for r in run_ids]:
        for c in R.load_partial(out_dir, tag)[1]:
            by_cell[c["cell"]] = c
    return by_cell


def _bridge_todo(rid: str, gaps: Dict, by_cell: Dict[str, Dict]) -> List[str]:
    meta = _parse_rid(rid)
    return [f for f in FINDINGS
            if _bridge_cell(rid, f) not in by_cell
            and np.isfinite(gaps.get((meta["objective"], meta["backbone"],
                                      meta["data_comp"], meta["seed"], f), np.nan))]


def _bridge_run_cells(rid: str, gaps: Dict, man_pool: pd.DataFrame, emb_root: str,
                      attr: str, seed: int, out_dir: str, pbar=None) -> int:
    done, cells, _ = R.load_partial(out_dir, _bridge_tag(rid))
    by_cell = {c["cell"]: c for c in cells}
    todo = _bridge_todo(rid, gaps, by_cell)
    if not todo:
        return 0
    npz = os.path.join(emb_root, rid, "cxr_pool.npz")
    if not os.path.exists(npz):
        return 0
    try:
        emb, ids = load_embeddings(npz)
        X, man = join_with_manifest(emb, ids, man_pool)
    except Exception as e:
        print(f"[e3] scale bridge: {rid} embeddings unreadable ({e}); skipping.")
        return 0
    meta = _parse_rid(rid)
    prep = prepare_pool(X, man, attr)
    written = 0
    for finding in todo:
        print(f"[e3] scale bridge: {rid}/{finding}...")
        g = gaps[(meta["objective"], meta["backbone"], meta["data_comp"],
                  meta["seed"], finding)]
        timings: Dict[str, float] = {}
        ent = entanglement_for(X, man, finding, attr, seed, prep=prep,
                               with_decodability=False, timings=timings)
        print(f"[e3] scale bridge: {rid}/{finding} "
              + (format_timings(timings) if ent is not None
                 else "no usable split, nothing fitted"))
        usable = (ent is not None
                  and np.isfinite(ent["leace_collateral"])
                  and np.isfinite(ent["geometric_overlap"]))
        by_cell[_bridge_cell(rid, finding)] = {
            "cell": _bridge_cell(rid, finding), "run_id": rid, "finding": finding,
            "usable": bool(usable),
            "leace_collateral": float(ent["leace_collateral"]) if usable else np.nan,
            "geometric_overlap": float(ent["geometric_overlap"]) if usable else np.nan,
            "gap": float(g)}
        done.add(_bridge_cell(rid, finding))
        R.save_partial(out_dir, _bridge_tag(rid), done, list(by_cell.values()), [])
        R.heartbeat_claim(out_dir, _bridge_tag(rid))
        written += 1
        if pbar is not None:
            pbar.update(1)
    del X, man, emb, ids
    return written


def _bridge_features(cells: List[Dict]) -> Tuple[List[List[float]], List[float]]:
    F, y = [], []
    for c in cells:
        if not c.get("usable"):
            continue
        F.append([c["leace_collateral"], c["geometric_overlap"]])
        y.append(c["gap"])
    return F, y


def _scale_bridge(cfg, cfg_path, pool_csv, foundation_rows, unmit_foundation,
                  attr, seed, n_perm, out_dir):
    Ff, yf = [], []
    for r in foundation_rows:
        g = unmit_foundation.get((r["encoder"], r["finding"]), np.nan)
        if np.isfinite(g) and np.isfinite(r["leace_collateral"]) and np.isfinite(r["geometric_overlap"]):
            Ff.append([r["leace_collateral"], r["geometric_overlap"]])
            yf.append(g)
    if len(Ff) < 2:
        return None

    e1_csv = os.path.join(cfg["controlled"]["results_e1_dir"], "results_performance_e1.csv")
    gaps = _e1_unmit_gaps(cfg, e1_csv, attr)
    if not gaps:
        print("[e3] scale bridge: no E1 unmitigated gaps found; skipping.")
        return None

    emb_root = os.path.join(cfg["embeddings"]["output_dir"], "controlled")
    run_ids = _controlled_run_ids(cfg)
    by_cell = _load_bridge_cells(out_dir, run_ids)
    todo = {rid: _bridge_todo(rid, gaps, by_cell) for rid in run_ids}
    n_todo = sum(len(v) for v in todo.values())
    if by_cell:
        print(f"[e3] scale bridge: resuming, {len(by_cell)} cells already done, "
              f"{n_todo} to go.")
    man_pool = None
    if n_todo:
        man_pool = read_csv_defensively(
            pool_csv, usecols=lambda c: c == "case_id" or c in E3_MANIFEST_COLUMNS)
    stale = float(cfg["stats"].get("claim_stale_after_s", 21600))
    held_elsewhere = []
    pbar = tqdm(total=n_todo, desc="[e3] scale bridge", unit="cell")
    for rid in run_ids:
        if not todo[rid]:
            continue
        if not R.claim_unit(out_dir, _bridge_tag(rid), stale):
            print(f"[e3] scale bridge: {rid} is claimed by another running job; "
                  f"skipping to the next run.")
            held_elsewhere.append(rid)
            continue
        try:
            _bridge_run_cells(rid, gaps, man_pool, emb_root, attr, seed, out_dir, pbar)
        finally:
            R.release_claim(out_dir, _bridge_tag(rid))
    pbar.close()

    by_cell = _load_bridge_cells(out_dir, run_ids)
    if held_elsewhere:
        still = {rid for rid in held_elsewhere if _bridge_todo(rid, gaps, by_cell)}
        if still:
            raise RuntimeError(
                f"[e3] scale bridge: {len(still)} of {len(run_ids)} controlled runs are "
                f"still being computed by another job ({sorted(still)[:3]}). Fitting now "
                f"would fit on part of the matrix, so this run stops instead. Re-run "
                f"main_e3 once every bridge job has ended; it picks up every cell they "
                f"wrote and goes straight to the fit.")

    ordered = [by_cell[_bridge_cell(rid, f)] for rid in run_ids for f in FINDINGS
               if _bridge_cell(rid, f) in by_cell]
    Fc, yc = _bridge_features(ordered)
    if len(Fc) < 4:
        print(f"[e3] scale bridge: only {len(Fc)} controlled cells; skipping.")
        return None

    Fc, yc, Ff, yf = np.array(Fc), np.array(yc), np.array(Ff), np.array(yf)
    r2 = transfer_r2(Fc, yc, Ff, yf)
    if np.isfinite(r2):
        rng = np.random.RandomState(seed)
        cnt = sum(1 for _ in range(n_perm)
                  if transfer_r2(Fc, yc, Ff, rng.permutation(yf)) >= r2)
        p = (cnt + 1) / (n_perm + 1)
    else:
        p = float("nan")
    print(f"[e3] scale bridge controlled->foundation: R2={r2:.3f} p={p:.4f} "
          f"(train n={len(Fc)} controlled, test n={len(Ff)} foundation)")
    return r2, p, len(Fc), len(Ff)


def _rows_from_perf(perf_rows: List[Dict]) -> List[Dict]:
    by_key: Dict[Tuple[str, str], Dict] = {}
    for r in perf_rows:
        key = (r.get("encoder"), r.get("finding"))
        by_key.setdefault(key, {})[r.get("metric_name")] = r.get("value_raw")
    rows = []
    for (encoder, finding), metrics in by_key.items():
        if "ceiling_gap_target" not in metrics:
            continue
        rows.append({"encoder": encoder, "finding": finding,
                     "decodability": metrics.get("decodability"),
                     "decodability_nonlinear": metrics.get("decodability_nonlinear"),
                     "geometric_overlap": metrics.get("geometric_overlap"),
                     "leace_collateral": metrics.get("leace_collateral"),
                     "ceiling_gap": metrics.get("ceiling_gap_target")})
    return rows


ENTANGLEMENT_METRICS = ("decodability", "decodability_nonlinear",
                        "leace_collateral", "geometric_overlap")


def seed_entanglement_partial(global_config_path: str, force: bool = False) -> int:
    cfg = read_config(global_config_path)["BiasOrigin"]
    out_dir = cfg["mechanism"]["results_e3_dir"]
    csv = os.path.join(out_dir, "results_performance_e3.csv")
    if not os.path.exists(csv):
        raise FileNotFoundError(f"[e3] no {csv} to seed from; run the panel.")
    if R.load_partial(out_dir, "e3_entanglement")[0] and not force:
        print("[e3] an e3_entanglement partial already exists; not overwriting "
              "(force=True to replace it).")
        return 0
    df = pd.read_csv(csv, float_precision="round_trip")
    want_enc = set(_e2_encoders(cfg))
    want_metrics = set(ENTANGLEMENT_METRICS) | {"ceiling_gap_target"}
    missing = []
    for enc in sorted(want_enc):
        sub = df[df["encoder"].astype(str) == enc]
        for finding in FINDINGS:
            have = set(sub[sub["finding"].astype(str) == finding]["metric_name"].astype(str))
            gone = want_metrics - have
            if gone:
                missing.append(f"{enc}/{finding}: {sorted(gone)}")
    if missing:
        raise RuntimeError(
            f"[e3] refusing to seed the partial: {len(missing)} incomplete cells, "
            f"first few {missing[:3]}. The CSV does not hold a finished panel, so "
            f"seeding would mark unfinished work as done.")
    rows = df[df["encoder"].astype(str).isin(want_enc)].to_dict("records")
    R.save_partial(out_dir, "e3_entanglement", sorted(want_enc), rows, [])
    print(f"[e3] seeded e3_entanglement from {csv}: {len(want_enc)} encoders, "
          f"{len(FINDINGS)} findings, {len(rows)} rows. A re-run of main_e3 now "
          f"skips the foundation panel and goes straight to the scale bridge.")
    return len(rows)


def main_e3_bridge_run(run_id: str, global_config_path: str) -> int:
    cfg = read_config(global_config_path)["BiasOrigin"]
    out_dir = cfg["mechanism"]["results_e3_dir"]
    seed = int(cfg["stats"]["boot_seed"])
    run_ids = _controlled_run_ids(cfg)
    if run_id not in run_ids:
        raise ValueError(f"[e3] '{run_id}' is not in the controlled matrix. "
                         f"Expected one of {run_ids}.")
    e1_csv = os.path.join(cfg["controlled"]["results_e1_dir"], "results_performance_e1.csv")
    gaps = _e1_unmit_gaps(cfg, e1_csv, ATTR)
    if not gaps:
        raise FileNotFoundError(
            f"[e3] no E1 unmitigated gaps in {e1_csv}; the bridge's target comes from "
            f"E1, so run E1 before any bridge cell.")
    by_cell = {c["cell"]: c for c in R.load_partial(out_dir, _bridge_tag(run_id))[1]}
    todo = _bridge_todo(run_id, gaps, by_cell)
    if not todo:
        print(f"[e3] scale bridge: {run_id} already has all {len(by_cell)} of its "
              f"cells; skipping.")
        return 0
    stale = float(cfg["stats"].get("claim_stale_after_s", 21600))
    if not R.claim_unit(out_dir, _bridge_tag(run_id), stale):
        print(f"[e3] scale bridge: {run_id} is claimed by another running job; "
              f"skipping to the next line.")
        return 0
    man_pool = read_csv_defensively(
        cfg["cxr"]["pool_manifest_csv"],
        usecols=lambda c: c == "case_id" or c in E3_MANIFEST_COLUMNS)
    emb_root = os.path.join(cfg["embeddings"]["output_dir"], "controlled")
    pbar = tqdm(total=len(todo), desc=f"[e3] bridge {run_id}", unit="cell")
    try:
        n = _bridge_run_cells(run_id, gaps, man_pool, emb_root, ATTR, seed, out_dir, pbar)
    finally:
        pbar.close()
        R.release_claim(out_dir, _bridge_tag(run_id))
    if n == 0:
        print(f"[e3] scale bridge: {run_id} produced nothing (missing or unreadable "
              f"embeddings at {emb_root}/{run_id}/cxr_pool.npz); not marked done.")
    else:
        print(f"[e3] scale bridge: {run_id} wrote {n} cells.")
    return n


def main_e3(global_config_path: str) -> Tuple[str, str]:
    cfg = read_config(global_config_path)["BiasOrigin"]
    pool_csv = cfg["cxr"]["pool_manifest_csv"]
    e2_perf = os.path.join(cfg["mitigation"]["results_e2_dir"], "results_performance_e2.csv")
    if not os.path.exists(e2_perf):
        raise FileNotFoundError(f"[e3] E2 performance CSV not found at {e2_perf}. Run E2 first.")
    ceil, unmit = _load_ceiling_target(e2_perf)

    seed = int(cfg["stats"]["boot_seed"])
    n_splits = int(cfg["mechanism"]["n_splits"])
    n_perm = int(cfg["mechanism"]["n_perm"])
    encoders = _e2_encoders(cfg)
    out_dir = cfg["mechanism"]["results_e3_dir"]

    done, perf_rows, stat_rows = R.load_partial(out_dir, "e3_entanglement")
    if done:
        print(f"[e3] resuming, {len(done)}/{len(encoders)} encoders already done.")
    rows = _rows_from_perf(perf_rows)

    remaining = [e for e in encoders if e not in done]
    pbar = tqdm(remaining, desc="[e3] entanglement", unit="encoder")
    for encoder in pbar:
        try:
            X, man = load_encoder_pool_frame(global_config_path, encoder, "cxr_pool",
                                             pool_csv, columns=E3_MANIFEST_COLUMNS)
        except FileNotFoundError:
            print(f"[e3] no embeddings for {encoder}; skipping (not marked done).")
            continue
        prep = prepare_pool(X, man, ATTR)
        for finding in FINDINGS:
            ent = entanglement_for(X, man, finding, ATTR, seed, prep=prep)
            if ent is None:
                continue
            key = (encoder, finding)
            target = ceil.get(key, np.nan)
            ctx = {"experiment": "e3", "modality": "cxr", "dataset": "cxr_pool",
                   "encoder": encoder, "encoder_objective": encoder,
                   "attribute": ATTR, "finding": finding, "mitigation": "none"}
            _emit_point(perf_rows, ctx, "decodability", ent["decodability"], "lower", ent["n_test"])
            _emit_point(perf_rows, ctx, "decodability_nonlinear", ent["decodability_nonlinear"], "lower", ent["n_test"])
            _emit_point(perf_rows, ctx, "leace_collateral", ent["leace_collateral"], "lower", ent["n_test"])
            _emit_point(perf_rows, ctx, "geometric_overlap", ent["geometric_overlap"], "lower", ent["n_test"], as_percent=False)
            _emit_point(perf_rows, ctx, "disease_auroc", ent["disease_auroc"], "higher", ent["n_test"])
            if np.isfinite(target):
                _emit_point(perf_rows, ctx, "ceiling_gap_target", target, "lower", ent["n_test"])
                rows.append({"encoder": encoder, "finding": finding,
                             "decodability": ent["decodability"],
                             "decodability_nonlinear": ent["decodability_nonlinear"],
                             "geometric_overlap": ent["geometric_overlap"],
                             "leace_collateral": ent["leace_collateral"],
                             "ceiling_gap": target})
        done.add(encoder)
        R.save_partial(out_dir, "e3_entanglement", done, perf_rows, stat_rows)
        print(f"[e3] entanglement done: {encoder}")

    tab = pd.DataFrame(rows)
    if tab.empty:
        raise RuntimeError("[e3] no encoder x finding rows with a ceiling target.")

    y = tab["ceiling_gap"].values
    ctx_all = {"experiment": "e3", "modality": "cxr", "dataset": "cxr_pool",
               "attribute": ATTR, "mitigation": "battery"}

    for q, direction in [("decodability", "decodability"),
                         ("leace_collateral", "leace_collateral"),
                         ("geometric_overlap", "geometric_overlap")]:
        rho, p = spearman_with_perm(tab[q].values, y, n_perm=n_perm, seed=seed)
        R.report_spearman(stat_rows, {**ctx_all, "finding": f"vs_ceiling::{q}"},
                          rho, p, fdr_family="e3_univariate", n_units=len(tab))

    F_ent = tab[["leace_collateral", "geometric_overlap"]].values
    F_dec = tab[["decodability"]].values
    F_dec_nl = tab[["decodability_nonlinear"]].values
    for name, F in [("entanglement", F_ent), ("decodability_only", F_dec),
                    ("decodability_nonlinear_only", F_dec_nl)]:
        r2, p = permutation_r2_null(F, y, n_perm=n_perm, n_splits=n_splits, seed=seed)
        stat_rows.append(R.pack_statistic(
            estimate=r2, estimate_name=f"cross_fit_r2_{name}",
            test_name="r2_target_permutation", p_raw=p,
            fdr_family="e3_predictor", context={**ctx_all, "finding": name},
            n_units=len(tab)))

    r2_logo = logo_r2(F_ent, y, tab["encoder"].values, seed)
    rng = np.random.RandomState(seed)
    if np.isfinite(r2_logo):
        cnt = sum(1 for _ in range(n_perm)
                  if logo_r2(F_ent, rng.permutation(y), tab["encoder"].values, seed) >= r2_logo)
        p_logo = (cnt + 1) / (n_perm + 1)
    else:
        p_logo = float("nan")
    stat_rows.append(R.pack_statistic(
        estimate=r2_logo, estimate_name="logo_r2_entanglement",
        test_name="r2_target_permutation_logo", p_raw=p_logo,
        fdr_family="e3_predictor_logo", context={**ctx_all, "finding": "leave_one_encoder_out"},
        n_units=len(tab)))

    bridge = _scale_bridge(cfg, global_config_path, pool_csv, rows, unmit, ATTR,
                           seed, n_perm, out_dir)
    if bridge is not None:
        r2_b, p_b, n_ctrl, n_found = bridge
        stat_rows.append(R.pack_statistic(
            estimate=r2_b, estimate_name="scale_bridge_r2_controlled_to_foundation",
            test_name="transfer_permutation", p_raw=p_b,
            fdr_family="e3_scale_bridge",
            context={**ctx_all, "finding": "controlled_to_foundation"},
            n_units=n_ctrl + n_found))

    R.add_fdr(stat_rows, alpha=float(cfg["stats"]["fdr_alpha"]))
    os.makedirs(out_dir, exist_ok=True)
    perf_csv = os.path.join(out_dir, "results_performance_e3.csv")
    stat_csv = os.path.join(out_dir, "results_statistics_e3.csv")
    R.write_frame_atomic(R.perf_frame(perf_rows), perf_csv)
    R.write_frame_atomic(R.stat_frame(stat_rows), stat_csv)
    R.clear_partial(out_dir, "e3_entanglement")
    if bridge is not None:
        for tag in [BRIDGE_TAG] + [_bridge_tag(r) for r in _controlled_run_ids(cfg)]:
            R.clear_partial(out_dir, tag)
    print(f"\n[e3] entanglement table: {len(tab)} encoder x finding rows")
    print(f"[e3] performance rows: {len(perf_rows)} -> {perf_csv}")
    print(f"[e3] statistics rows:  {len(stat_rows)} -> {stat_csv}")
    return perf_csv, stat_csv

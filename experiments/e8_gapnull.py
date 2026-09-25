"""
experiments/e8_gapnull.py
Created on July 26, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm

from config.serde import read_config
from Inference.stats_utils import spearman_with_perm
from Inference import report_utils as R

import warnings
warnings.filterwarnings("ignore")

_ARM_DIR_KEY = {
    "e1": ("controlled", "results_e1_dir"),
    "e2": ("mitigation", "results_e2_dir"),
    "e4": ("generalization", "results_e4_dir"),
    "e6": ("finetuning", "results_e6_dir"),
}


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    if not np.isfinite(auc) or n_pos < 1 or n_neg < 1:
        return float("nan")
    a = float(min(max(auc, 1e-6), 1 - 1e-6))
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    var = (a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) \
        / (float(n_pos) * float(n_neg))
    return float(np.sqrt(var)) if var > 0 else 0.0


def _sim_subgroup_hanley(auc, n_pos, n_neg, n_sim, rng) -> np.ndarray:
    se = hanley_mcneil_se(auc, n_pos, n_neg)
    if not np.isfinite(se):
        return np.full(n_sim, np.nan)
    return np.clip(rng.normal(auc, se, n_sim), 0.0, 1.0)


def _sim_subgroup_exact(auc, n_pos, n_neg, n_sim, rng, chunk=64) -> np.ndarray:
    a = float(min(max(auc, 1e-6), 1 - 1e-6))
    mu = np.sqrt(2.0) * norm.ppf(a)
    out = np.empty(n_sim, float)
    done = 0
    while done < n_sim:
        k = int(min(chunk, n_sim - done))
        pos = rng.normal(mu, 1.0, (k, n_pos))
        neg = np.sort(rng.normal(0.0, 1.0, (k, n_neg)), axis=1)
        for j in range(k):
            out[done + j] = np.searchsorted(neg[j], pos[j]).mean() / float(n_neg)
        done += k
    return out


def _sim_subgroup_rate(rate, n_den, _unused, n_sim, rng) -> np.ndarray:
    if not np.isfinite(rate) or n_den < 1:
        return np.full(n_sim, np.nan)
    r = float(min(max(rate, 0.0), 1.0))
    return rng.binomial(int(n_den), r, n_sim) / float(n_den)


def simulate_null_gap(value: float, counts: List[Tuple[int, int]], n_sim: int,
                      rng, method: str = "hanley") -> np.ndarray:
    if method in ("rate_tpr", "rate_fpr"):
        k = 0 if method == "rate_tpr" else 1
        draws = np.vstack([_sim_subgroup_rate(value, c[k], None, n_sim, rng)
                           for c in counts])
    else:
        sim = _sim_subgroup_exact if method == "exact" else _sim_subgroup_hanley
        draws = np.vstack([sim(value, npos, nneg, n_sim, rng) for npos, nneg in counts])
    return np.nanmax(draws, axis=0) - np.nanmin(draws, axis=0)


def _agg(samples, n) -> dict:
    a = np.asarray(samples, float); a = a[np.isfinite(a)]
    if a.size == 0:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "n": int(n)}
    return {"point": float(a.mean()),
            "std": float(a.std(ddof=1)) if a.size > 1 else np.nan,
            "ci_lower": float(np.percentile(a, 2.5)),
            "ci_upper": float(np.percentile(a, 97.5)), "n": int(n)}


def _subgroup_counts_index(counts_csv: str, min_n: int) -> Dict:
    df = pd.read_csv(counts_csv, low_memory=False, float_precision="round_trip")
    df = df[df["split"].astype(str).str.lower() == "test"]
    keep = ["modality", "attribute", "finding", "subgroup"]
    g = df.groupby(keep, dropna=True)[["n_images", "n_positive"]].sum().reset_index()
    g["n_neg"] = g["n_images"] - g["n_positive"]
    out: Dict = {}
    for (mod, attr, find), sub in g.groupby(["modality", "attribute", "finding"]):
        cells = {}
        for _, r in sub.iterrows():
            npos, nneg = int(r["n_positive"]), int(r["n_neg"])
            if npos >= 1 and nneg >= 1 and (npos + nneg) >= min_n:
                cells[str(r["subgroup"])] = (npos, nneg)
        if cells:
            out[(str(mod), str(attr), str(find))] = cells
    return out


def _observed_cells(cfg, arms: List[str]) -> pd.DataFrame:
    rows = []
    for arm in arms:
        if arm not in _ARM_DIR_KEY:
            print(f"[e8] unknown source arm '{arm}'; skipping.")
            continue
        blk, key = _ARM_DIR_KEY[arm]
        path = os.path.join(cfg[blk][key], f"results_performance_{arm}.csv")
        if not os.path.exists(path):
            print(f"[e8] (absent) {path}")
            continue
        df = pd.read_csv(path, low_memory=False, float_precision="round_trip")
        df = df[df["metric_name"].isin(["auroc_gap", "auroc_overall"])]
        df = df[df["mitigation"].astype(str).str.startswith(("none", "finetune:"))]
        keys = ["experiment", "modality", "dataset", "encoder", "attribute",
                "finding", "mitigation"]
        piv = df.pivot_table(index=keys, columns="metric_name",
                             values="value_raw", aggfunc="first").reset_index()
        if "auroc_gap" not in piv.columns or "auroc_overall" not in piv.columns:
            continue
        piv = piv.dropna(subset=["auroc_gap", "auroc_overall"])
        print(f"[e8] + {arm}: {len(piv)} observed cells")
        rows.append(piv)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _tracked_from_rows(perf_rows: List[Dict]) -> Dict[Tuple[str, str], List[Tuple[float, float]]]:
    by_cell: Dict[Tuple, Dict[str, float]] = {}
    for r in perf_rows:
        if r.get("metric_name") not in ("gap_null", "gap_observed"):
            continue
        k = (str(r.get("data_composition")), str(r.get("attribute")),
             str(r.get("encoder")), str(r.get("finding")), str(r.get("mitigation")))
        by_cell.setdefault(k, {})[r["metric_name"]] = r.get("value_raw")
    out: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    for (comp, attr, _enc, _find, _mit), vals in by_cell.items():
        o, n = vals.get("gap_observed"), vals.get("gap_null")
        if o is None or n is None or not (np.isfinite(o) and np.isfinite(n)):
            continue
        exp = comp[len("source_"):] if comp.startswith("source_") else comp
        out.setdefault((exp, attr), []).append((float(o), float(n)))
    return out


def _verify_hanley(cells, index, n_sim, seed, n_verify) -> None:
    if n_verify <= 0 or cells.empty:
        return
    rng = np.random.RandomState(seed + 991)
    pick = cells.sample(n=int(min(n_verify, len(cells))), random_state=seed)
    rows = []
    n_exact_sim = int(min(n_sim, 400))
    for _, c in tqdm(list(pick.iterrows()), desc="[e8] verify hanley vs exact",
                     unit="cell"):
        counts = index.get((c["modality"], c["attribute"], c["finding"]))
        if not counts:
            continue
        cl = list(counts.values())
        h = simulate_null_gap(c["auroc_overall"], cl, n_exact_sim,
                              np.random.RandomState(seed), "hanley")
        e = simulate_null_gap(c["auroc_overall"], cl, n_exact_sim, rng, "exact")
        rows.append((float(np.mean(h)), float(np.mean(e))))
    if not rows:
        return
    h = np.array([r[0] for r in rows]); e = np.array([r[1] for r in rows])
    d = 100.0 * (h - e)
    print(f"[e8] hanley-vs-exact on {len(rows)} sampled cells "
          f"({n_exact_sim} draws each): mean null gap "
          f"{100*h.mean():.2f} vs {100*e.mean():.2f} points, "
          f"mean signed difference {d.mean():+.2f}, max |difference| "
          f"{np.abs(d).max():.2f} points.")


def main_e8(global_config_path: str) -> Tuple[str, str]:
    cfg = read_config(global_config_path)["BiasOrigin"]
    gn = cfg["gap_null"]
    out_dir = gn["results_e8_dir"]
    n_sim = int(gn.get("n_sim", 2000))
    seed = int(gn.get("seed", 0))
    method = str(gn.get("method", "hanley"))
    min_n = int(gn.get("min_subgroup_n", 20))
    min_groups = int(gn.get("min_subgroups", 2))

    counts_csv = cfg["subgroup_counts"]["out_csv"]
    if not os.path.exists(counts_csv):
        raise R.MissingInput(
            f"[e8] subgroup counts not found: {counts_csv}. Run "
            f"main_build_subgroup_counts before E8.")
    index = _subgroup_counts_index(counts_csv, min_n)
    print(f"[e8] subgroup-count index: {len(index)} (modality, attribute, finding) keys.")

    cells = _observed_cells(cfg, list(gn.get("source_arms", ["e2", "e4", "e6"])))
    if cells.empty:
        raise R.MissingInput(
            "[e8] no observed cells found. E8 reads the MERGED per-arm "
            "performance CSVs, so run those arms and their merges first.")
    print(f"[e8] {len(cells)} observed cells total; simulating {n_sim} fair "
          f"models each with method='{method}'.")

    if method != "exact":
        _verify_hanley(cells, index, n_sim, seed, int(gn.get("verify_exact_cells", 12)))

    done, perf_rows, stat_rows = R.load_partial(out_dir, "e8_main")
    if done:
        print(f"[e8] resuming, {len(done)}/{len(cells)} cells already done.")
    n_skipped = 0

    def _cell_key(c) -> str:
        return "::".join(str(c[k]) for k in
                         ("experiment", "encoder", "attribute", "finding", "mitigation"))

    for _, c in tqdm(list(cells.iterrows()), desc="[e8] null gap", unit="cell"):
        ck = _cell_key(c)
        if ck in done:
            continue
        key = (c["modality"], c["attribute"], c["finding"])
        counts = index.get(key)
        if not counts or len(counts) < min_groups:
            n_skipped += 1
            done.add(ck)
            continue
        rng = np.random.RandomState(seed)
        null = simulate_null_gap(float(c["auroc_overall"]), list(counts.values()),
                                 n_sim, rng, method)
        null = null[np.isfinite(null)]
        if null.size == 0:
            n_skipped += 1
            done.add(ck)
            continue
        obs = float(c["auroc_gap"])

        ctx = {"experiment": "e8", "modality": c["modality"], "dataset": c["dataset"],
               "eval_dataset": c["dataset"], "encoder": c["encoder"],
               "encoder_objective": c["encoder"], "attribute": c["attribute"],
               "finding": c["finding"], "mitigation": c["mitigation"],
               "data_composition": f"source_{c['experiment']}"}
        R.report_metric(perf_rows, ctx, "gap_null", _agg(null, null.size),
                        "lower", int(sum(n for n, _ in counts.values())))
        R.report_metric(perf_rows, ctx, "gap_observed",
                        {"point": obs, "std": np.nan, "ci_lower": np.nan,
                         "ci_upper": np.nan, "n": null.size}, "lower",
                        int(sum(n for n, _ in counts.values())))
        excess = obs - null
        p_exc = (1.0 + float(np.sum(null >= obs))) / (null.size + 1.0)
        R.report_paired_diff(
            perf_rows, stat_rows, ctx, "gap_excess",
            {**_agg(excess, null.size), "p_value": p_exc},
            fdr_family=f"e8_gap_vs_null::{c['attribute']}",
            test_name="fair_model_simulation", metric_direction="lower",
            n_patients=int(sum(n for n, _ in counts.values())))
        done.add(ck)
        R.save_partial(out_dir, "e8_main", done, perf_rows, stat_rows)

    tracked = _tracked_from_rows(perf_rows)

    if n_skipped:
        print(f"[e8] {n_skipped} cells skipped: no subgroup-count row, or fewer "
              f"than {min_groups} subgroups at n >= {min_n}.")

    n_perm = int(cfg["stats"]["n_perm"])
    for (exp, attr), pairs in sorted(tracked.items()):
        if len(pairs) < 6:
            continue
        o = np.array([p[0] for p in pairs]); nl = np.array([p[1] for p in pairs])
        rho, p = spearman_with_perm(nl, o, n_perm, seed)
        R.report_spearman(
            stat_rows,
            {"experiment": "e8", "modality": np.nan, "attribute": attr,
             "finding": f"null_vs_observed::{exp}", "mitigation": "gap_null"},
            rho, p, fdr_family="e8_null_tracks_observed", n_units=len(pairs))

    R.add_fdr(stat_rows, alpha=float(cfg["stats"]["fdr_alpha"]))
    os.makedirs(out_dir, exist_ok=True)
    perf_csv = os.path.join(out_dir, "results_performance_e8.csv")
    stat_csv = os.path.join(out_dir, "results_statistics_e8.csv")
    R.write_frame_atomic(R.perf_frame(perf_rows), perf_csv)
    R.write_frame_atomic(R.stat_frame(stat_rows), stat_csv)
    R.clear_partial(out_dir, "e8_main")

    sig = [s for s in stat_rows
           if str(s.get("fdr_family", "")).startswith("e8_gap_vs_null")
           and bool(s.get("significant_fdr05"))]
    n_tests = len([s for s in stat_rows
                   if str(s.get("fdr_family", "")).startswith("e8_gap_vs_null")])
    print(f"\n[e8] {len(sig)}/{n_tests} cells have a gap larger than a perfectly "
          f"fair model produces at the same subgroup sizes (BH-FDR).")
    print(f"[e8] performance rows: {len(perf_rows)} -> {perf_csv}")
    print(f"[e8] statistics rows:  {len(stat_rows)} -> {stat_csv}")
    return perf_csv, stat_csv

"""
experiments/validation.py
Created on November 20, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from tqdm import tqdm

from config.serde import read_config
from data_loader.cxr_harmonization import CANONICAL_PATHOLOGY_FINDINGS
from analysis import patient_bootstrap as PB
from Inference import report_utils as R
from Inference.stats_utils import BOOT_SEED, N_BOOT
from analysis.count_chance import simulate_null_gap
from experiments.scores import read_scores, score_file, unit_params
from experiments.chance import FORMULA_METHOD, pool_cases


VALIDATION_VERSION = 1
N_SHUFFLES = 5
NOISE_LEVELS = (0.0, 0.5, 1.0)
LEVEL_ENCODERS = ("rad_dino", "dinov2_large")
N_SHUFFLES_LEVEL = 100


def chest_encoders(cfg) -> List[str]:
    panel = cfg["encoder_panel"]["image"]
    return [name for name, spec in panel.items()
            if spec.get("modality", "general") in ("general", "cxr")]


def _units_dir(cfg: Dict) -> str:
    return os.path.join(cfg["subgroups"]["results_chance_dir"], "validation", "units")


def shuffled_groups(g_patient: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    out = g_patient.copy()
    has = np.array([v is not None for v in g_patient])
    out[has] = rng.permutation(g_patient[has])
    return out


def test_head(y, s, cl, thr, g, weights, n_sim, seed) -> List[Dict]:
    names = PB.eligible_groups(y, g, 20)
    if len(names) < 2:
        return []
    point = {m: [] for m in PB.MEASURES}
    boot = {m: [] for m in PB.MEASURES}
    counts = []
    for name in names:
        msk = g == name
        gr = PB.GroupRows(y[msk], s[msk], cl[msk], thr)
        a, b = gr.stats(), gr.boot(weights)
        n_pos = int((y[msk] == 1).sum())
        counts.append((n_pos, int(msk.sum()) - n_pos))
        for m in PB.MEASURES:
            point[m].append(a[m][0])
            boot[m].append(b[m])
    pooled = PB.GroupRows(y[np.isin(g, names)], s[np.isin(g, names)], cl[np.isin(g, names)], thr).stats()
    out = []
    for m in PB.MEASURES:
        d = PB.difference_against_chance(np.array(point[m]), np.column_stack(boot[m]))
        sim = simulate_null_gap(float(pooled[m][0]), counts, n_sim, np.random.default_rng(seed), FORMULA_METHOD[m])
        out.append({"measure": m, "difference": d["difference"], "p_patient": d["p_raw"], "p_count": float((1 + (sim >= d["difference"] - 1e-12).sum()) / (n_sim + 1)),
                    "group_values": json.dumps([round(float(v), 6) for v in point[m]])})
    return out


def run_encoder(cfg: Dict, encoder: str, level_finding: Optional[str] = None) -> str:
    level = level_finding is not None
    n_shuffles, noise_levels = (N_SHUFFLES_LEVEL, (0.0,)) if level else (N_SHUFFLES, NOISE_LEVELS)
    findings = [level_finding] if level else list(CANONICAL_PATHOLOGY_FINDINGS)
    name = f"level__{encoder}__{level_finding}" if level else encoder
    path = os.path.join(_units_dir(cfg), f"{name}.csv")
    params = {"version": VALIDATION_VERSION, "shuffles": n_shuffles, "noise": list(noise_levels), "scores": unit_params(cfg, encoder, "cxr_pool")}
    if level:
        params["kind"] = "level"
    try:
        with open(path + ".params.json") as fh:
            if json.load(fh) == json.loads(json.dumps(params)) and os.path.exists(path):
                return "skipped"
    except (OSError, ValueError):
        pass
    tag = f"validation__{name}"
    if not R.claim_unit(_units_dir(cfg), tag):
        return "held"
    try:
        cases = pool_cases(cfg, "cxr_pool")
        g_all = next(gr for a, st, gr in cases["analyses"] if a == "race_grp" and st == "all")
        n_sim, seed = int(cfg["gap_null"]["n_sim"]), int(cfg["stats"]["boot_seed"])
        rows = []
        for f in tqdm(findings, desc=f"[validation] {name}", unit="finding"):
            sc = read_scores(score_file(cfg, encoder, "cxr_pool", f), params["scores"])
            if sc is None:
                raise R.MissingInput(f"[validation] {encoder}/{f}: no current score file; run main_scores_sweep.")
            if sc["status"] != "scored":
                continue
            rows_m = sc["manifest_row"].astype(np.int64)
            y = pd.to_numeric(cases["man"][f].iloc[rows_m], errors="coerce").values
            s0, thr, cl, g = sc["y_score"].astype(np.float64), float(sc["threshold"]), cases["cluster"][rows_m], g_all[rows_m]
            keep = np.array([v is not None for v in g])
            y, s0, cl, g = y[keep], s0[keep], cl[keep], g[keep]
            uniq, inv = np.unique(cl, return_inverse=True)
            g_patient = np.empty(uniq.size, dtype=object)
            g_patient[inv] = g
            sd = float(np.std(s0))
            for r in range(n_shuffles):
                rng = np.random.RandomState(seed + 1000 * r + 1)
                gs = shuffled_groups(g_patient, rng)[inv]
                target = min(PB.eligible_groups(y, gs, 20), key=lambda gname: (gs == gname).sum())
                for noise in noise_levels:
                    s = s0.copy()
                    if noise > 0:
                        msk = gs == target
                        s[msk] = s[msk] + rng.normal(0, noise * sd, msk.sum())
                    for row in test_head(y, s, cl, thr, gs, cases["weights"], n_sim, seed):
                        rows.append({"encoder": encoder, "finding": f, "shuffle": r, "noise": noise, "noisy_group": target if noise > 0 else None, **row})
            R.heartbeat_claim(_units_dir(cfg), tag)
        os.makedirs(_units_dir(cfg), exist_ok=True)
        if os.path.exists(path + ".params.json"):
            os.remove(path + ".params.json")
        R.write_frame_atomic(pd.DataFrame(rows), path)
        with open(path + ".params.json.tmp", "w") as fh:
            json.dump(params, fh)
        os.replace(path + ".params.json.tmp", path + ".params.json")
    finally:
        R.release_claim(_units_dir(cfg), tag)
    return "done"


def _family_scores(family: str, a: float, n_pos: int, n_neg: int, n_data: int, rng) -> Tuple[np.ndarray, np.ndarray]:
    if family == "binormal":
        mu = np.sqrt(2.0) * norm.ppf(a)
        return rng.normal(mu, 1.0, (n_data, n_pos)), rng.normal(0.0, 1.0, (n_data, n_neg))
    if family == "binormal_unequal":
        mu = np.sqrt(5.0) * norm.ppf(a)
        return rng.normal(mu, 2.0, (n_data, n_pos)), rng.normal(0.0, 1.0, (n_data, n_neg))
    pos = np.where(rng.rand(n_data, n_pos) < a, 10.0, -10.0) + rng.rand(n_data, n_pos)
    return pos, rng.rand(n_data, n_neg)


def _auroc_rows(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    n_pos, n_neg = pos.shape[1], neg.shape[1]
    allv = np.concatenate([pos, neg], axis=1)
    ranks = rankdata(allv, axis=1)
    return (ranks[:, :n_pos].sum(axis=1) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def simulate_setting(family: str, a: float, delta: float, n_pos: int, ratio: int, k: int, n_data: int, n_sim: int, seed: int) -> Dict:
    rng = np.random.RandomState(seed)
    n_neg = n_pos * ratio
    vals, pooled_pos, pooled_neg = [], [], []
    for j in range(k):
        pos, neg = _family_scores(family, a - (delta if j == 0 else 0.0), n_pos, n_neg, n_data, rng)
        vals.append(_auroc_rows(pos, neg))
        pooled_pos.append(pos)
        pooled_neg.append(neg)
    vals = np.column_stack(vals)
    obs = vals.max(axis=1) - vals.min(axis=1)
    overall = _auroc_rows(np.concatenate(pooled_pos, axis=1)[:, :2000], np.concatenate(pooled_neg, axis=1)[:, :4000])
    out = {"family": family, "auroc": a, "delta": delta, "n_pos": n_pos, "neg_per_pos": ratio, "k_groups": k, "n_datasets": n_data}
    for name in ("hanley", "bound"):
        p = np.empty(n_data)
        for i in range(n_data):
            if name == "hanley":
                g = simulate_null_gap(float(overall[i]), [(n_pos, n_neg)] * k, n_sim, np.random.default_rng(seed + i), "hanley")
            else:
                sd = np.sqrt(overall[i] * (1 - overall[i]) / n_pos)
                d = np.clip(np.random.default_rng(seed + i).normal(overall[i], sd, (n_sim, k)), 0, 1)
                g = d.max(axis=1) - d.min(axis=1)
            p[i] = (1 + (g >= obs[i] - 1e-12).sum()) / (n_sim + 1)
        rate = float((p < 0.05).mean())
        out.update({f"rejection_{name}": rate, f"rejection_{name}_ci_low": float(max(0.0, rate - 1.96 * np.sqrt(rate * (1 - rate) / n_data))),
                    f"rejection_{name}_ci_high": float(min(1.0, rate + 1.96 * np.sqrt(rate * (1 - rate) / n_data)))})
    return out


def main_validation(global_config_path: str) -> None:
    cfg = read_config(global_config_path)["BiasOrigin"]
    out_dir = os.path.join(cfg["subgroups"]["results_chance_dir"], "validation")
    os.makedirs(out_dir, exist_ok=True)
    status = {}
    missing = []
    for e in tqdm(chest_encoders(cfg), desc="[validation] encoders", unit="encoder"):
        try:
            status[e] = run_encoder(cfg, e)
        except R.MissingInput:
            missing.append(e)
    level_units = [(e, f) for e in LEVEL_ENCODERS for f in CANONICAL_PATHOLOGY_FINDINGS]
    for e, f in tqdm(level_units, desc="[validation] level units", unit="unit"):
        try:
            status[f"level__{e}__{f}"] = run_encoder(cfg, e, level_finding=f)
        except R.MissingInput:
            missing.append(f"level__{e}__{f}")
    sim_path = os.path.join(out_dir, "simulation.csv")
    sim_params = {"version": VALIDATION_VERSION, "n_data": 1000, "n_sim": 2000}
    try:
        with open(sim_path + ".params.json") as fh:
            sim_current = json.load(fh) == sim_params and os.path.exists(sim_path)
    except (OSError, ValueError):
        sim_current = False
    if not sim_current and R.claim_unit(out_dir, "validation_simulation"):
        try:
            grid = [(fam, a, d, n, r, k) for fam in ("binormal", "binormal_unequal", "worst_case") for a in (0.75, 0.90) for d in (0.0, 0.05, 0.10)
                    for n in (25, 100, 400) for r in (1, 9) for k in (2, 4)]
            rows = [simulate_setting(*g, n_data=1000, n_sim=2000, seed=i) for i, g in enumerate(tqdm(grid, desc="[validation] simulation", unit="setting"))]
            R.write_frame_atomic(pd.DataFrame(rows), sim_path)
            with open(sim_path + ".params.json.tmp", "w") as fh:
                json.dump(sim_params, fh)
            os.replace(sim_path + ".params.json.tmp", sim_path + ".params.json")
        finally:
            R.release_claim(out_dir, "validation_simulation")
    if missing or any(v == "held" for v in status.values()):
        raise R.MissingInput(f"[validation] units not finished: {missing + [e for e, v in status.items() if v == 'held']}")
    real = pd.concat([pd.read_csv(os.path.join(_units_dir(cfg), f"{e}.csv")) for e in chest_encoders(cfg)], ignore_index=True)
    summary = level_rates(real, ["noise", "measure"]).merge(real.groupby(["noise", "measure"])["difference"].median().rename("median_difference").reset_index(), on=["noise", "measure"])
    R.write_frame_atomic(real, os.path.join(out_dir, "shuffled_groups.csv"))
    R.write_frame_atomic(summary, os.path.join(out_dir, "shuffled_groups_summary.csv"))
    level = pd.concat([_read_unit(os.path.join(_units_dir(cfg), f"level__{e}__{f}.csv")) for e, f in level_units], ignore_index=True)
    level_summary = pd.concat([level_rates(level, ["measure"]).assign(encoder="all"), level_rates(level, ["encoder", "measure"])], ignore_index=True)
    R.write_frame_atomic(level, os.path.join(out_dir, "shuffled_groups_level.csv"))
    R.write_frame_atomic(level_summary, os.path.join(out_dir, "shuffled_groups_level_summary.csv"))


def _read_unit(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def level_rates(level: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    rows = []
    for key, h in level.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        row["n_tests"] = int(len(h))
        codes, uniq = pd.factorize(h["shuffle"].astype(str) + "|" + h["finding"].astype(str))
        row["n_clusters"] = int(len(uniq))
        idx = np.random.default_rng(BOOT_SEED).integers(0, len(uniq), size=(N_BOOT, len(uniq)))
        for form in ("patient", "count"):
            p = h[f"p_{form}"].to_numpy(dtype=float)
            ok = ~np.isnan(p)
            k = np.bincount(codes[ok], weights=(p[ok] < 0.05).astype(float), minlength=len(uniq))
            n = np.bincount(codes[ok], minlength=len(uniq)).astype(float)
            nb = n[idx].sum(axis=1)
            b = np.where(nb > 0, k[idx].sum(axis=1) / np.maximum(nb, 1.0), np.nan)
            row.update({f"rejection_{form}": float(k.sum() / n.sum()) if n.sum() else np.nan, f"rejection_{form}_std": float(np.nanstd(b, ddof=1)),
                        f"rejection_{form}_ci_low": float(np.nanpercentile(b, 2.5)), f"rejection_{form}_ci_high": float(np.nanpercentile(b, 97.5))})
        rows.append(row)
    return pd.DataFrame(rows)

"""
experiments/contrasts.py
Created on November 20, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.serde import read_config
from analysis.patient_bootstrap import MEASURES
from Inference import report_utils as R
from Inference.stats_utils import bh_fdr
from experiments.scores import controlled_run_ids
from experiments.chance import boot_file, unit_current, unit_file


CONTRAST_VERSION = 1
ATTRIBUTES = ("race_grp", "age_grp", "sex_grp", "insurance_grp")


def parse_run(run_id: str) -> Dict:
    _, obj, bk, _init, dc, seed = run_id.split("__")
    return {"run": run_id, "objective": obj, "backbone": bk, "data": dc, "seed": int(seed.replace("seed", ""))}


def run_pairs(cfg: Dict) -> List[Tuple[str, str, str]]:
    runs = [parse_run(r) for r in controlled_run_ids(cfg)]
    by = {(r["objective"], r["backbone"], r["data"], r["seed"]): r["run"] for r in runs}
    out = []
    for a, b in (("image_text", "ssl"), ("image_text", "supervised"), ("ssl", "supervised")):
        for (obj, bk, dc, seed), run in by.items():
            if obj == a and dc in ("natural", "balanced") and (b, bk, dc, seed) in by:
                out.append((f"objective:{a}_minus_{b}:{dc}", run, by[(b, bk, dc, seed)]))
    for (obj, bk, dc, seed), run in by.items():
        if dc == "natural" and seed == 0:
            for other in ("balanced", "scrubbed", "amplified"):
                if (obj, bk, other, 0) in by:
                    out.append((f"data:{other}_minus_natural", by[(obj, bk, other, 0)], run))
    return sorted(out)


class RunStats:

    def __init__(self, cfg: Dict, run_id: str):
        key = f"controlled/{run_id}"
        if not unit_current(cfg, key, "cxr_pool"):
            raise R.MissingInput(f"[contrasts] {run_id}: its chance unit is not current; run main_chance_sweep.")
        df = pd.read_csv(unit_file(cfg, key, "cxr_pool"), low_memory=False, float_precision="round_trip")
        self.groups = df[(df["table"] == "group") & (df["stratum"] == "all")]
        self.overall = df[df["table"] == "overall"].set_index("finding")
        self.boot = np.load(boot_file(cfg, key, "cxr_pool"))

    def stat(self, finding: str, attribute: str, statistic: str) -> Tuple[float, np.ndarray]:
        if statistic == "overall_auroc":
            return float(self.overall.loc[finding, "auroc"]), self.boot[f"{finding}|_all|all|_all|auroc"].astype(np.float64)
        meas = "auroc" if statistic == "worst_auroc" else statistic.split("_", 1)[1]
        g = self.groups[(self.groups["finding"] == finding) & (self.groups["attribute"] == attribute)]
        if len(g) < 2:
            raise KeyError(f"{finding}/{attribute}")
        pt = g[meas].values.astype(np.float64)
        bt = np.column_stack([self.boot[f"{finding}|{attribute}|all|{name}|{meas}"].astype(np.float64) for name in g["group"]])
        if statistic == "worst_auroc":
            return float(pt.min()), bt.min(axis=1)
        return float(pt.max() - pt.min()), bt.max(axis=1) - bt.min(axis=1)


def paired_delta(a: Tuple[float, np.ndarray], b: Tuple[float, np.ndarray]) -> Dict[str, float]:
    delta = a[0] - b[0]
    d = (a[1] - b[1])
    d = d[np.isfinite(d)]
    p = float((1 + (np.abs(d - delta) >= abs(delta) - 1e-15).sum()) / (d.size + 1))
    return {"value_a": a[0], "value_b": b[0], "delta": delta, "delta_std": float(d.std(ddof=1)), "delta_ci_low": float(np.percentile(d, 2.5)),
            "delta_ci_high": float(np.percentile(d, 97.5)), "p_raw": p}


def _input_signature(cfg: Dict) -> Dict:
    sig = {"version": CONTRAST_VERSION}
    for r in controlled_run_ids(cfg):
        p = unit_file(cfg, f"controlled/{r}", "cxr_pool")
        sig[r] = [int(os.stat(p).st_size), int(os.stat(p).st_mtime_ns)] if os.path.exists(p) else None
    return sig


def main_contrasts(global_config_path: str) -> None:
    cfg = read_config(global_config_path)["BiasOrigin"]
    out_dir = cfg["subgroups"]["results_chance_dir"]
    out_c, out_s = os.path.join(out_dir, "contrasts.csv"), os.path.join(out_dir, "seed_spread.csv")
    sig = _input_signature(cfg)
    record = out_c + ".params.json"
    try:
        with open(record) as f:
            if json.load(f) == json.loads(json.dumps(sig)) and os.path.exists(out_c) and os.path.exists(out_s):
                return
    except (OSError, ValueError):
        pass
    if not R.claim_unit(out_dir, "contrasts"):
        return
    try:
        stats: Dict[str, RunStats] = {}

        def get(run_id: str) -> RunStats:
            if run_id not in stats:
                stats[run_id] = RunStats(cfg, run_id)
            return stats[run_id]
        statistics = [f"difference_{m}" for m in MEASURES] + ["worst_auroc", "overall_auroc"]
        rows = []
        for contrast, ra, rb in tqdm(run_pairs(cfg), desc="[contrasts] run pairs", unit="pair"):
            a, b = get(ra), get(rb)
            for finding in a.overall.index:
                for att in ATTRIBUTES:
                    for st in statistics:
                        if st == "overall_auroc" and att != ATTRIBUTES[0]:
                            continue
                        try:
                            d = paired_delta(a.stat(finding, att, st), b.stat(finding, att, st))
                        except KeyError:
                            continue
                        rows.append({"contrast": contrast, "run_a": ra, "run_b": rb, "backbone": parse_run(ra)["backbone"], "seed": parse_run(ra)["seed"],
                                     "finding": finding, "attribute": "all" if st == "overall_auroc" else att, "statistic": st, **d})
        con = pd.DataFrame(rows)
        con["p_fdr"] = np.nan
        for _, idx in con.groupby(["contrast", "attribute", "statistic"]).groups.items():
            con.loc[idx, "p_fdr"] = bh_fdr(con.loc[idx, "p_raw"].values)
        con["significant"] = con["p_fdr"] < float(cfg["stats"]["fdr_alpha"])
        seeds = [parse_run(r) for r in controlled_run_ids(cfg)]
        spread = []
        for obj in sorted({r["objective"] for r in seeds}):
            trio = sorted((r for r in seeds if r["objective"] == obj and r["backbone"] == "vit_s" and r["data"] == "natural"), key=lambda r: r["seed"])
            if len(trio) < 3:
                continue
            for finding in get(trio[0]["run"]).overall.index:
                for att in ATTRIBUTES:
                    for st in statistics[:-1]:
                        try:
                            vals = [get(r["run"]).stat(finding, att, st)[0] for r in trio]
                        except KeyError:
                            continue
                        spread.append({"objective": obj, "finding": finding, "attribute": att, "statistic": st, "n_seeds": len(vals),
                                       "spread": float(max(vals) - min(vals)), "median_abs_change_from_seed0": float(np.median(np.abs(np.array(vals[1:]) - vals[0])))})
        if os.path.exists(record):
            os.remove(record)
        R.write_frame_atomic(con, out_c)
        R.write_frame_atomic(pd.DataFrame(spread), out_s)
        with open(record + ".tmp", "w") as f:
            json.dump(sig, f)
        os.replace(record + ".tmp", record)
    finally:
        R.release_claim(out_dir, "contrasts")

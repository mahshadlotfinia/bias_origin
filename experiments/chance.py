"""
experiments/chance.py
Created on November 20, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import json
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.serde import read_config
from data_loader.build_utils import read_csv_defensively
from analysis import patient_bootstrap as PB
from Inference import report_utils as R
from Inference.stats_utils import bh_fdr
from analysis.count_chance import simulate_null_gap
from experiments.scores import params_match, pool_findings, read_scores, score_file, score_params_difference, score_units, unit_params


CHANCE_VERSION = 2
BEST_IS_LOW = {"auroc": False, "tpr": False, "fpr": True}
FORMULA_METHOD = {"auroc": "hanley", "tpr": "rate_tpr", "fpr": "rate_fpr"}
_CASES: Dict[str, Dict] = {}


def pool_cases(cfg: Dict, pool: str) -> Dict:
    if pool in _CASES:
        return _CASES[pool]
    man = read_csv_defensively(cfg["embeddings"]["pools"][pool]["manifest"])
    test = man["split"].astype(str).str.lower().eq("test").values
    who = man["subject_id"] if "subject_id" in man.columns else man["case_id"]
    who = who.where(who.notna(), man["case_id"]).astype(str)
    key = (man["dataset"].astype(str) + "::" + who).values
    cluster = np.full(len(man), -1, dtype=np.int64)
    codes, uniq = pd.factorize(key[test])
    cluster[test] = codes
    groups = {}
    for att in cfg["subgroups"]["attributes"][pool]:
        g = man[att].astype(str).values.astype(object)
        drop = {"nan", "", "None"} | {str(v) for v in cfg["subgroups"]["unrecorded_levels"].get(att, [])}
        g[np.isin(g, list(drop))] = None
        groups[att] = g
    analyses = [(att, "all", g) for att, g in groups.items()]
    st = cfg["subgroups"].get("strata", {}).get(pool, {})
    for by in st.get("by", []):
        for level in sorted({v for v in groups[by] if v is not None}):
            for att in st.get("attributes", []):
                g = groups[att].copy()
                g[groups[by] != level] = None
                analyses.append((att, f"{by}={level}", g))
    weights = PB.boot_weights(len(uniq), int(cfg["stats"]["n_boot"]), int(cfg["stats"]["boot_seed"]))
    _CASES[pool] = {"man": man, "cluster": cluster, "analyses": analyses, "weights": weights, "n_patients": len(uniq)}
    return _CASES[pool]


def unit_file(cfg: Dict, encoder_key: str, pool: str) -> str:
    return os.path.join(cfg["subgroups"]["results_chance_dir"], "units", f"{encoder_key.replace('/', '__')}__{pool}.csv")


def boot_file(cfg: Dict, encoder_key: str, pool: str) -> str:
    return unit_file(cfg, encoder_key, pool)[:-4] + ".boot.npz"


def chance_params(cfg: Dict, encoder_key: str, pool: str) -> Dict:
    sig = {}
    for f in pool_findings(cfg, pool):
        p = score_file(cfg, encoder_key, pool, f)
        sig[f] = [int(os.stat(p).st_size), int(os.stat(p).st_mtime_ns)] if os.path.exists(p) else None
    return {"version": CHANCE_VERSION, "n_boot": int(cfg["stats"]["n_boot"]), "seed": int(cfg["stats"]["boot_seed"]),
            "min_subgroup_n": int(cfg["gap_null"]["min_subgroup_n"]), "n_sim": int(cfg["gap_null"]["n_sim"]),
            "attributes": list(cfg["subgroups"]["attributes"][pool]), "unrecorded_levels": cfg["subgroups"]["unrecorded_levels"],
            "strata": cfg["subgroups"].get("strata", {}).get(pool, {}), "scores": sig}


def unit_current(cfg: Dict, encoder_key: str, pool: str) -> bool:
    path = unit_file(cfg, encoder_key, pool)
    try:
        with open(path + ".params.json") as f:
            stored = json.load(f)
    except (OSError, ValueError):
        return False
    return os.path.exists(path) and os.path.exists(boot_file(cfg, encoder_key, pool)) and params_match(stored, chance_params(cfg, encoder_key, pool))


def _formula(measure: str, obs: float, overall: float, counts: List, n_sim: int, seed: int) -> Dict[str, float]:
    sim = simulate_null_gap(overall, counts, n_sim, np.random.default_rng(seed), FORMULA_METHOD[measure])
    out = {"formula_chance": float(np.nanmean(sim)), "formula_p": float((1 + (sim >= obs - 1e-12).sum()) / (n_sim + 1))}
    if measure == "auroc":
        rng = np.random.default_rng(seed)
        draws = np.vstack([np.clip(rng.normal(overall, np.sqrt(overall * (1 - overall) / min(p, n)), n_sim), 0, 1) for p, n in counts])
        bound = draws.max(axis=0) - draws.min(axis=0)
        out.update({"bound_chance": float(bound.mean()), "bound_p": float((1 + (bound >= obs - 1e-12).sum()) / (n_sim + 1))})
    return out


def analyze_finding(cfg: Dict, cases: Dict, base: Dict, finding: str, sc: Dict, keep_boot: Dict) -> List[Dict]:
    rows_m = sc["manifest_row"].astype(np.int64)
    y = pd.to_numeric(cases["man"][finding].iloc[rows_m], errors="coerce").values
    s, thr = sc["y_score"].astype(np.float64), float(sc["threshold"])
    cl = cases["cluster"][rows_m]
    if (cl < 0).any() or not np.isfinite(y).all():
        raise ValueError(f"[chance] {base['encoder']}/{finding}: a scored row is not a labeled test row of the manifest.")
    min_n, n_sim, seed = int(cfg["gap_null"]["min_subgroup_n"]), int(cfg["gap_null"]["n_sim"]), int(cfg["stats"]["boot_seed"])
    out = []
    everyone = PB.GroupRows(y, s, cl, thr)
    st_all, bt_all = everyone.stats(), everyone.boot(cases["weights"])
    row = {"table": "overall", **base, "finding": finding, "threshold": thr, "n_images": int(y.size), "n_patients": int(np.unique(cl).size), "n_positive": int((y == 1).sum())}
    for meas in PB.MEASURES:
        v, sd, lo, hi = PB.summarize(st_all[meas][0], bt_all[meas])
        row.update({meas: v, f"{meas}_std": sd, f"{meas}_ci_low": lo, f"{meas}_ci_high": hi})
        keep_boot[f"{finding}|_all|all|_all|{meas}"] = bt_all[meas].astype(np.float32)
    out.append(row)
    for att, stratum, g_all in cases["analyses"]:
        g = g_all[rows_m]
        names = PB.eligible_groups(y, g, min_n)
        if len(names) < 2:
            continue
        ctx = {**base, "finding": finding, "attribute": att, "stratum": stratum, "threshold": thr}
        point = {m: [] for m in PB.MEASURES}
        boot = {m: [] for m in PB.MEASURES}
        counts = []
        for name in names:
            m_ = g == name
            gr = PB.GroupRows(y[m_], s[m_], cl[m_], thr)
            st, bt = gr.stats(), gr.boot(cases["weights"])
            n_pos = int((y[m_] == 1).sum())
            counts.append((n_pos, int(m_.sum()) - n_pos))
            row = {"table": "group", **ctx, "group": name, "n_images": int(m_.sum()), "n_patients": int(np.unique(cl[m_]).size), "n_positive": n_pos}
            for meas in PB.MEASURES:
                v, sd, lo, hi = PB.summarize(st[meas][0], bt[meas])
                row.update({meas: v, f"{meas}_std": sd, f"{meas}_ci_low": lo, f"{meas}_ci_high": hi})
                point[meas].append(st[meas][0])
                boot[meas].append(bt[meas])
                keep_boot[f"{finding}|{att}|{stratum}|{name}|{meas}"] = bt[meas].astype(np.float32)
            out.append(row)
        in_any = np.isin(g, names)
        pooled = PB.GroupRows(y[in_any], s[in_any], cl[in_any], thr).stats()
        for meas in PB.MEASURES:
            pt, bt = np.array(point[meas]), np.column_stack(boot[meas])
            d = PB.difference_against_chance(pt, bt)
            lo, hi = int(np.argmin(pt)), int(np.argmax(pt))
            best, worst = (lo, hi) if BEST_IS_LOW[meas] else (hi, lo)
            overall = float(pooled[meas][0])
            den = [c[0] if meas != "fpr" else c[1] for c in counts]
            out.append({"table": "difference", **ctx, "measure": meas, "k_groups": len(names), "groups": "; ".join(names), "best_group": names[best],
                        "worst_group": names[worst], "overall": overall, "smallest_denominator": int(min(den)) if meas != "auroc" else int(min(min(c) for c in counts)),
                        **d, **_formula(meas, d["difference"], overall, counts, n_sim, seed)})
    return out


def run_unit(cfg: Dict, encoder_key: str, pool: str) -> str:
    if unit_current(cfg, encoder_key, pool):
        return "skipped"
    params = chance_params(cfg, encoder_key, pool)
    if any(v is None for v in params["scores"].values()):
        raise R.MissingInput(f"[chance] {encoder_key}/{pool}: scores missing; run main_scores_sweep.")
    out_dir = cfg["subgroups"]["results_chance_dir"]
    tag = f"chance__{encoder_key.replace('/', '__')}__{pool}"
    if not R.claim_unit(out_dir, tag):
        return "held"
    try:
        cases = pool_cases(cfg, pool)
        base = {"pool": pool, "encoder": encoder_key}
        rows, keep_boot = [], {}
        for f in tqdm(pool_findings(cfg, pool), desc=f"[chance] {encoder_key}/{pool}", unit="finding"):
            sc = read_scores(score_file(cfg, encoder_key, pool, f), unit_params(cfg, encoder_key, pool))
            if sc is None:
                raise R.MissingInput(f"[chance] {encoder_key}/{pool}/{f}: its score file is not current; run main_scores_sweep.")
            if sc["status"] == "scored":
                rows += analyze_finding(cfg, cases, base, f, sc, keep_boot)
            R.heartbeat_claim(out_dir, tag)
        path = unit_file(cfg, encoder_key, pool)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path + ".params.json"):
            os.remove(path + ".params.json")
        R.write_frame_atomic(pd.DataFrame(rows), path)
        tmp_b = f"{boot_file(cfg, encoder_key, pool)}.{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp_b, **keep_boot)
        os.replace(tmp_b, boot_file(cfg, encoder_key, pool))
        tmp = f"{path}.params.json.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(params, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path + ".params.json")
    finally:
        R.release_claim(out_dir, tag)
    return "done"


def main_chance_sweep(global_config_path: str) -> None:
    cfg = read_config(global_config_path)["BiasOrigin"]
    counts, missing = {"done": 0, "skipped": 0, "held": 0}, []
    for encoder_key, pool in tqdm(score_units(cfg), desc="[chance] units", unit="unit"):
        try:
            counts[run_unit(cfg, encoder_key, pool)] += 1
        except R.MissingInput:
            missing.append(f"{encoder_key}/{pool}")
    if missing:
        raise R.MissingInput(f"[chance] {len(missing)} unit(s) wait for their scores: {', '.join(missing[:6])}")


def strata_tests(cfg: Dict, units: List, grp: pd.DataFrame, dif: pd.DataFrame, spec_all: Dict = None, boot_of=None) -> pd.DataFrame:
    spec_all = cfg["subgroups"].get("strata", {}) if spec_all is None else spec_all
    boot_of = boot_file if boot_of is None else boot_of
    rows = []
    for e, p in tqdm(units, desc="[chance] strata tests", unit="unit"):
        by_list = spec_all.get(p, {}).get("by", [])
        g_u = grp[(grp["encoder"] == e) & (grp["pool"] == p) & (grp["stratum"] != "all")]
        if not by_list or g_u.empty:
            continue
        with np.load(boot_of(cfg, e, p)) as z:
            boot = {k: z[k].astype(np.float64) for k in z.files}
        for by in by_list:
            sub = g_u[g_u["stratum"].str.startswith(f"{by}=")]
            for (f, att), h in sub.groupby(["finding", "attribute"]):
                levels = sorted(h["stratum"].unique())
                ref = dif[(dif["encoder"] == e) & (dif["pool"] == p) & (dif["finding"] == f) & (dif["attribute"] == att) & (dif["stratum"] == "all")]
                for meas in PB.MEASURES:
                    t_obs, t_b, per_level = 0.0, 0.0, []
                    for lev in levels:
                        hh = h[h["stratum"] == lev]
                        pts = hh[meas].to_numpy(float)
                        rec = np.column_stack([boot[f"{f}|{att}|{lev}|{name}|{meas}"] for name in hh["group"]]) - pts
                        t_b = t_b + (rec.max(axis=1) - rec.min(axis=1))
                        per_level.append(f"{lev.split('=', 1)[1]}:{pts.max() - pts.min():.6f}")
                        t_obs += float(pts.max() - pts.min())
                    ok = np.isfinite(t_b)
                    n_lev, tb = len(levels), t_b[ok] / len(levels)
                    obs = t_obs / n_lev
                    overall = ref[ref["measure"] == meas]["difference"]
                    rows.append({"pool": p, "encoder": e, "finding": f, "attribute": att, "by": by, "n_levels": n_lev, "levels": "; ".join(per_level), "measure": meas,
                                 "difference_within": obs, "chance_within": float(tb.mean()) if tb.size else np.nan, "chance_within_std": float(tb.std(ddof=1)) if tb.size > 1 else np.nan,
                                 "p_raw": float((1 + (tb >= obs - 1e-12).sum()) / (tb.size + 1)) if tb.size else np.nan, "n_boot_used": int(tb.size),
                                 "difference_all": float(overall.iloc[0]) if len(overall) else np.nan})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_fdr"] = np.nan
    for _, idx in out[out["p_raw"].notna()].groupby(["pool", "by", "attribute", "measure"]).groups.items():
        out.loc[idx, "p_fdr"] = bh_fdr(out.loc[idx, "p_raw"].values)
    out["larger_than_chance"] = out["p_fdr"] < float(cfg["stats"]["fdr_alpha"])
    return out


MERGE_VERSION = 2


def main_chance_merge(global_config_path: str) -> None:
    cfg = read_config(global_config_path)["BiasOrigin"]
    units = score_units(cfg)
    stale = [f"{e}/{p}" for e, p in units if not unit_current(cfg, e, p)]
    if stale:
        raise R.MissingInput(f"[chance] {len(stale)} of {len(units)} unit(s) are not current (first: {stale[0]}); run main_chance_sweep.")
    out_dir = cfg["subgroups"]["results_chance_dir"]
    outputs = [os.path.join(out_dir, n) for n in ("chance_overall.csv", "chance_groups.csv", "chance_differences.csv", "chance_strata.csv")]
    sig = {"version": MERGE_VERSION, "fdr_alpha": float(cfg["stats"]["fdr_alpha"]),
           "units": {f"{e}/{p}": [int(os.stat(unit_file(cfg, e, p)).st_size), int(os.stat(unit_file(cfg, e, p)).st_mtime_ns)] for e, p in units}}
    record = os.path.join(out_dir, "chance_merge.params.json")
    try:
        with open(record) as fh:
            if json.load(fh) == json.loads(json.dumps(sig)) and all(os.path.exists(o) for o in outputs):
                return
    except (OSError, ValueError):
        pass
    df = pd.concat([pd.read_csv(unit_file(cfg, e, p), low_memory=False, float_precision="round_trip") for e, p in units], ignore_index=True)
    grp = df[df["table"] == "group"].dropna(axis=1, how="all").drop(columns=["table"])
    dif = df[df["table"] == "difference"].dropna(axis=1, how="all").drop(columns=["table"]).copy()
    for col, name in (("p_raw", "p_fdr"), ("formula_p", "formula_p_fdr")):
        dif[name] = np.nan
        for _, idx in dif.groupby(["pool", "attribute", "stratum", "measure"]).groups.items():
            dif.loc[idx, name] = bh_fdr(dif.loc[idx, col].values)
    dif["larger_than_chance"] = dif["p_fdr"] < float(cfg["stats"]["fdr_alpha"])
    strata = strata_tests(cfg, units, grp, dif)
    if os.path.exists(record):
        os.remove(record)
    R.write_frame_atomic(df[df["table"] == "overall"].dropna(axis=1, how="all").drop(columns=["table"]), outputs[0])
    R.write_frame_atomic(grp, outputs[1])
    R.write_frame_atomic(dif, outputs[2])
    R.write_frame_atomic(strata, outputs[3])
    tmp = f"{record}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(sig, fh)
    os.replace(tmp, record)


JOINT_VERSION = 1


def joint_spec(cfg: Dict, pool: str) -> Dict:
    return cfg["subgroups"].get("strata_joint", {}).get(pool, {})


def joint_analyses(cfg: Dict, pool: str, cases: Dict) -> List:
    spec = joint_spec(cfg, pool)
    groups = {att: g for att, st, g in cases["analyses"] if st == "all"}
    out = []
    for by in spec.get("by", []):
        parts = by.split("+")
        ok = np.ones(len(cases["man"]), dtype=bool)
        for part in parts:
            ok &= np.array([v is not None for v in groups[part]])
        lab = np.full(len(cases["man"]), None, dtype=object)
        lab[ok] = ["|".join(str(groups[part][i]) for part in parts) for i in np.where(ok)[0]]
        for level in sorted({v for v in lab[ok]}):
            for att in spec.get("attributes", []):
                g = groups[att].copy()
                g[lab != level] = None
                out.append((att, f"{by}={level}", g))
    return out


def joint_unit_file(cfg: Dict, encoder_key: str, pool: str) -> str:
    return unit_file(cfg, encoder_key, pool)[:-4] + ".joint.csv"


def joint_boot_file(cfg: Dict, encoder_key: str, pool: str) -> str:
    return unit_file(cfg, encoder_key, pool)[:-4] + ".joint.boot.npz"


def joint_params(cfg: Dict, encoder_key: str, pool: str) -> Dict:
    return {**chance_params(cfg, encoder_key, pool), "strata": joint_spec(cfg, pool), "joint_version": JOINT_VERSION}


def joint_current(cfg: Dict, encoder_key: str, pool: str) -> bool:
    path = joint_unit_file(cfg, encoder_key, pool)
    try:
        with open(path + ".params.json") as f:
            stored = json.load(f)
    except (OSError, ValueError):
        return False
    return os.path.exists(path) and os.path.exists(joint_boot_file(cfg, encoder_key, pool)) and params_match(stored, joint_params(cfg, encoder_key, pool))


def run_joint_unit(cfg: Dict, encoder_key: str, pool: str) -> str:
    if joint_current(cfg, encoder_key, pool):
        return "skipped"
    params = joint_params(cfg, encoder_key, pool)
    if any(v is None for v in params["scores"].values()):
        raise R.MissingInput(f"[chance_joint] {encoder_key}/{pool}: scores missing; run main_scores_sweep.")
    out_dir = cfg["subgroups"]["results_chance_dir"]
    tag = f"chance_joint__{encoder_key.replace('/', '__')}__{pool}"
    if not R.claim_unit(out_dir, tag):
        return "held"
    try:
        if joint_current(cfg, encoder_key, pool):
            return "skipped"
        cases = pool_cases(cfg, pool)
        cases_j = {**cases, "analyses": joint_analyses(cfg, pool, cases)}
        base = {"pool": pool, "encoder": encoder_key}
        rows, keep_boot = [], {}
        for f in tqdm(pool_findings(cfg, pool), desc=f"[chance_joint] {encoder_key}/{pool}", unit="finding"):
            sp = unit_params(cfg, encoder_key, pool)
            sc = read_scores(score_file(cfg, encoder_key, pool, f), sp)
            if sc is None:
                why = score_params_difference(score_file(cfg, encoder_key, pool, f), sp)
                raise R.MissingInput(f"[chance_joint] {encoder_key}/{pool}/{f}: its score file is not current ({why}); run main_scores_sweep.")
            if sc["status"] == "scored":
                rows += [r for r in analyze_finding(cfg, cases_j, base, f, sc, keep_boot) if r["table"] != "overall"]
            R.heartbeat_claim(out_dir, tag)
        keep_boot = {k: v for k, v in keep_boot.items() if "|_all|" not in k}
        path = joint_unit_file(cfg, encoder_key, pool)
        if os.path.exists(path + ".params.json"):
            os.remove(path + ".params.json")
        R.write_frame_atomic(pd.DataFrame(rows), path)
        tmp_b = f"{joint_boot_file(cfg, encoder_key, pool)}.{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp_b, **keep_boot)
        os.replace(tmp_b, joint_boot_file(cfg, encoder_key, pool))
        tmp = f"{path}.params.json.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(params, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path + ".params.json")
    finally:
        R.release_claim(out_dir, tag)
    return "done"


def main_chance_joint(global_config_path: str) -> None:
    cfg = read_config(global_config_path)["BiasOrigin"]
    units = [(e, p) for e, p in score_units(cfg) if joint_spec(cfg, p)]
    counts, missing, held = {"done": 0, "skipped": 0, "held": 0}, [], []
    for e, p in tqdm(units, desc="[chance_joint] units", unit="unit"):
        try:
            status = run_joint_unit(cfg, e, p)
        except R.MissingInput:
            missing.append(f"{e}/{p}")
            continue
        counts[status] += 1
        if status == "held":
            held.append((e, p))
    if missing:
        raise R.MissingInput(f"[chance_joint] {len(missing)} unit(s) wait for scores (first: {missing[0]}); run main_scores_sweep.")
    last_note, pause = 0.0, 1.0
    while held:
        held = [(e, p) for e, p in held if run_joint_unit(cfg, e, p) == "held"]
        if held:
            if time.time() - last_note > 60.0:
                last_note = time.time()
            time.sleep(pause)
            pause = min(2.0 * pause, 10.0)
    out_dir = cfg["subgroups"]["results_chance_dir"]
    out = os.path.join(out_dir, "chance_strata_joint.csv")
    main_dif = os.path.join(out_dir, "chance_differences.csv")
    if not os.path.exists(main_dif):
        raise R.MissingInput("[chance_joint] chance_differences.csv is missing; run main_chance_merge.")
    sig = {"version": JOINT_VERSION, "fdr_alpha": float(cfg["stats"]["fdr_alpha"]), "differences": [int(os.stat(main_dif).st_size), int(os.stat(main_dif).st_mtime_ns)],
           "units": {f"{e}/{p}": [int(os.stat(joint_unit_file(cfg, e, p)).st_size), int(os.stat(joint_unit_file(cfg, e, p)).st_mtime_ns)] for e, p in units}}
    record = out + ".params.json"

    def merged() -> bool:
        try:
            with open(record) as fh:
                return json.load(fh) == json.loads(json.dumps(sig)) and os.path.exists(out)
        except (OSError, ValueError):
            return False
    if not R.claim_or_wait(out_dir, "chance_joint__merge", merged):
        return
    try:
        df = pd.concat([pd.read_csv(joint_unit_file(cfg, e, p), low_memory=False, float_precision="round_trip") for e, p in units], ignore_index=True)
        grp = df[df["table"] == "group"].dropna(axis=1, how="all").drop(columns=["table"])
        dif = pd.read_csv(main_dif, low_memory=False, float_precision="round_trip")
        spec_all = {p: joint_spec(cfg, p) for _, p in units}
        strata = strata_tests(cfg, units, grp, dif, spec_all=spec_all, boot_of=joint_boot_file)
        if os.path.exists(record):
            os.remove(record)
        R.write_frame_atomic(strata, out)
        tmp = f"{record}.{os.getpid()}.tmp"
        with open(tmp, "w") as fh:
            json.dump(sig, fh)
        os.replace(tmp, record)
    finally:
        R.release_claim(out_dir, "chance_joint__merge")

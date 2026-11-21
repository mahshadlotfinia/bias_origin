"""
experiments/scores.py
Created on November 20, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import json
import os
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.serde import read_config
from data_loader.build_utils import read_csv_defensively
from data_loader.cxr_harmonization import CANONICAL_PATHOLOGY_FINDINGS
from analysis.embedding_io import load_embeddings
from analysis import heads as H
from Inference import report_utils as R
from analysis.heads import fit_head, score_head, split_masks, threshold_at_sensitivity


SCORES_VERSION = 1
POOLS_OF_MODALITY = {"general": ["cxr_pool", "derm_pool", "fundus_pool"], "cxr": ["cxr_pool"], "derm": ["derm_pool"], "fundus": ["fundus_pool"]}


def controlled_run_ids(cfg) -> List[str]:
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


def pool_findings(cfg: Dict, pool: str) -> List[str]:
    if pool == "cxr_pool":
        return list(CANONICAL_PATHOLOGY_FINDINGS)
    return list(cfg["generalization"]["derm_findings" if pool == "derm_pool" else "fundus_findings"])


def score_units(cfg: Dict) -> List[Tuple[str, str]]:
    units = []
    for name, spec in cfg["encoder_panel"]["image"].items():
        units += [(name, pool) for pool in POOLS_OF_MODALITY[spec.get("modality", "general")]]
    units += [(f"controlled/{run_id}", "cxr_pool") for run_id in controlled_run_ids(cfg)]
    return units


def embedding_file(cfg: Dict, encoder_key: str, pool: str) -> str:
    return os.path.join(cfg["embeddings"]["output_dir"], encoder_key, f"{pool}.npz")


def score_file(cfg: Dict, encoder_key: str, pool: str, finding: str) -> str:
    return os.path.join(cfg["subgroups"]["scores_dir"], encoder_key, pool, f"{finding}.npz")


def _signature(path: str) -> List[int]:
    st = os.stat(path)
    return [int(st.st_size), int(st.st_mtime_ns)]


def _portable(value):
    if isinstance(value, dict):
        return {k: _portable(v) for k, v in value.items()}
    if isinstance(value, list) and len(value) == 2 and all(type(v) is int for v in value) and value[1] > 10 ** 17:
        return [value[0], value[1] // 10 ** 9]
    if isinstance(value, list):
        return [_portable(v) for v in value]
    return value


def params_match(stored: Dict, current: Dict) -> bool:
    return _portable(json.loads(json.dumps(stored))) == _portable(json.loads(json.dumps(current)))


def params_difference(stored: Dict, current: Dict) -> str:
    a, b = _portable(json.loads(json.dumps(stored))), _portable(json.loads(json.dumps(current)))
    return "; ".join(f"{k}: stored {a.get(k)}, here {b.get(k)}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)) or "none"


def score_params_difference(path: str, params: Dict) -> str:
    try:
        with np.load(path, allow_pickle=False) as z:
            stored = json.loads(str(z["params"]))
    except (OSError, ValueError, KeyError) as e:
        return f"unreadable ({type(e).__name__})"
    return params_difference(stored, params)


def unit_params(cfg: Dict, encoder_key: str, pool: str) -> Dict:
    return {"version": SCORES_VERSION, "head": "linear", "logreg_c": H.LOGREG_C, "logreg_max_iter": H.LOGREG_MAX_ITER, "head_seed": H.SEED,
            "max_train": cfg["stats"].get("head_max_train"), "seed": int(cfg["stats"]["boot_seed"]),
            "target_sensitivity": float(cfg["stats"]["operating_sensitivity"]),
            "embeddings": _signature(embedding_file(cfg, encoder_key, pool)), "manifest": _signature(cfg["embeddings"]["pools"][pool]["manifest"])}


def read_scores(path: str, params: Optional[Dict] = None) -> Optional[Dict]:
    try:
        with np.load(path, allow_pickle=False) as z:
            out = {k: z[k] for k in z.files}
        stored = json.loads(str(out["params"]))
    except (OSError, ValueError, KeyError):
        return None
    if params is not None and not params_match(stored, params):
        return None
    out["status"] = str(out["status"])
    if out["status"] == "scored" and (out["manifest_row"].shape != out["y_score"].shape or out["y_score"].size == 0):
        return None
    return out


def _write_scores(path: str, params: Dict, status: str, rows=None, scores=None, threshold=np.nan) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp.npz"
    np.savez_compressed(tmp, params=json.dumps(params), status=status,
                        manifest_row=np.asarray(rows if rows is not None else [], dtype=np.int32),
                        y_score=np.asarray(scores if scores is not None else [], dtype=np.float32),
                        threshold=np.float64(threshold))
    os.replace(tmp, path)


def load_aligned(cfg: Dict, encoder_key: str, pool: str) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    emb, ids = load_embeddings(embedding_file(cfg, encoder_key, pool))
    man = read_csv_defensively(cfg["embeddings"]["pools"][pool]["manifest"])
    pos = {str(c): i for i, c in enumerate(ids)}
    hit = np.array([pos.get(str(c), -1) for c in man["case_id"].astype(str)], dtype=np.int64)
    rows = np.where(hit >= 0)[0]
    if rows.size == 0:
        raise ValueError(f"[scores] {encoder_key}/{pool}: no case id of the manifest has an embedding.")
    return emb[hit[rows]], man.iloc[rows].reset_index(drop=True), rows


def run_unit(cfg: Dict, encoder_key: str, pool: str) -> Dict[str, int]:
    if not os.path.exists(embedding_file(cfg, encoder_key, pool)):
        raise R.MissingInput(f"[scores] no embeddings for {encoder_key}/{pool}: {embedding_file(cfg, encoder_key, pool)}")
    params = unit_params(cfg, encoder_key, pool)
    findings = pool_findings(cfg, pool)
    todo = [f for f in findings if read_scores(score_file(cfg, encoder_key, pool, f), params) is None]
    counts = {"skipped": len(findings) - len(todo), "scored": 0, "not_evaluable": 0}
    if not todo:
        return counts
    out_dir = cfg["subgroups"]["scores_dir"]
    tag = f"scores__{encoder_key.replace('/', '__')}__{pool}"
    if not R.claim_unit(out_dir, tag):
        return counts
    try:
        X, man, rows = load_aligned(cfg, encoder_key, pool)
        for f in tqdm(todo, desc=f"[scores] {encoder_key}/{pool}", unit="finding"):
            cell = _fit_cell(X, man, f, params["max_train"], params["seed"], params["target_sensitivity"])
            path = score_file(cfg, encoder_key, pool, f)
            if cell is None:
                _write_scores(path, params, "not_evaluable")
                counts["not_evaluable"] += 1
            else:
                _write_scores(path, params, "scored", rows[cell["te"]], cell["s_te"], cell["thr"])
                counts["scored"] += 1
            R.heartbeat_claim(out_dir, tag)
    finally:
        R.release_claim(out_dir, tag)
    return counts


def main_scores_unit(encoder_key: str, pool: str, global_config_path: str) -> None:
    cfg = read_config(global_config_path)["BiasOrigin"]
    run_unit(cfg, encoder_key, pool)


def main_scores_sweep(global_config_path: str) -> None:
    cfg = read_config(global_config_path)["BiasOrigin"]
    missing, total = [], {"skipped": 0, "scored": 0, "not_evaluable": 0}
    for encoder_key, pool in tqdm(score_units(cfg), desc="[scores] units", unit="unit"):
        try:
            for k, v in run_unit(cfg, encoder_key, pool).items():
                total[k] += v
        except R.MissingInput:
            missing.append(f"{encoder_key}/{pool}")
    if missing:
        raise R.MissingInput(f"[scores] {len(missing)} unit(s) have no embeddings: {', '.join(missing)}")

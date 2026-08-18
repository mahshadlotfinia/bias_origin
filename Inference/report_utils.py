"""
Inference/report_utils.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

from typing import Callable, Dict, List, Optional

import json
import os
import shutil
import socket
import subprocess
import time

import numpy as np
import pandas as pd

from Inference.stats_utils import bh_fdr

PCT_DECIMALS = 1
TEST_PACKAGE = "scipy/statsmodels/sklearn"


class MissingInput(Exception):
    pass


def write_frame_atomic(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

PERF_COLUMNS: List[str] = [
    "experiment", "modality", "dataset", "eval_dataset", "encoder",
    "encoder_objective", "backbone", "data_composition", "seed", "head_type",
    "attribute", "subgroup", "finding", "mitigation", "operating_point",
    "metric_name", "metric_direction",
    "value", "std", "ci_low", "ci_high",
    "value_raw", "std_raw", "ci_low_raw", "ci_high_raw",
    "n_units", "n_patients",
]

STAT_COLUMNS: List[str] = [
    "experiment", "modality", "dataset", "eval_dataset", "encoder",
    "encoder_objective", "backbone", "data_composition", "seed", "head_type",
    "attribute", "finding", "mitigation",
    "estimate", "estimate_name", "test_name",
    "p_raw", "p_fdr", "fdr_family", "significant_fdr05",
    "n_units", "test_package",
]


def fmt_pct(x) -> float:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return np.nan
    return round(100.0 * float(x), PCT_DECIMALS)


def keep_p(p) -> float:
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return np.nan
    return float(p)


def _blank(columns: List[str], context: Dict) -> Dict:
    row = {c: np.nan for c in columns}
    for k, v in (context or {}).items():
        if k in row:
            row[k] = v
    return row


def pack_performance(
    metric_name: str,
    boot: dict,
    metric_direction: str,
    context: Dict,
    n_patients: Optional[int] = None,
    as_percent: bool = True,
) -> Dict:
    row = _blank(PERF_COLUMNS, context)
    row["metric_name"] = metric_name
    row["metric_direction"] = metric_direction
    pt, sd = boot.get("point", np.nan), boot.get("std", np.nan)
    lo, hi = boot.get("ci_lower", np.nan), boot.get("ci_upper", np.nan)
    if as_percent:
        row["value"], row["std"] = fmt_pct(pt), fmt_pct(sd)
        row["ci_low"], row["ci_high"] = fmt_pct(lo), fmt_pct(hi)
    else:
        row["value"], row["std"] = (round(float(pt), 4) if np.isfinite(pt) else np.nan), \
                                   (round(float(sd), 4) if np.isfinite(sd) else np.nan)
        row["ci_low"] = round(float(lo), 4) if np.isfinite(lo) else np.nan
        row["ci_high"] = round(float(hi), 4) if np.isfinite(hi) else np.nan
    row["value_raw"]   = float(pt) if np.isfinite(pt) else np.nan
    row["std_raw"]     = float(sd) if np.isfinite(sd) else np.nan
    row["ci_low_raw"]  = float(lo) if np.isfinite(lo) else np.nan
    row["ci_high_raw"] = float(hi) if np.isfinite(hi) else np.nan
    row["n_units"]     = int(boot.get("n", 0))
    row["n_patients"]  = int(n_patients) if n_patients is not None else np.nan
    return row


def report_metric(perf_rows: List[Dict], context: Dict, metric_name: str,
                  boot: dict, metric_direction: str = "higher",
                  n_patients: Optional[int] = None) -> None:
    perf_rows.append(pack_performance(metric_name, boot, metric_direction,
                                      context, n_patients))


def report_paired_diff(perf_rows: List[Dict], stat_rows: List[Dict], context: Dict,
                       metric_name: str, boot_diff: dict, fdr_family: str,
                       test_name: str = "paired_cluster_bootstrap",
                       metric_direction: str = "lower",
                       n_patients: Optional[int] = None) -> None:
    perf_rows.append(pack_performance(f"{metric_name}__diff", boot_diff,
                                      metric_direction, context, n_patients))
    pack = pack_statistic(
        estimate=boot_diff.get("point", np.nan),
        estimate_name=f"{metric_name}_difference",
        test_name=test_name, p_raw=boot_diff.get("p_value", np.nan),
        fdr_family=fdr_family, context=context, n_units=boot_diff.get("n", 0))
    stat_rows.append(pack)


def pack_statistic(estimate, estimate_name: str, test_name: str, p_raw,
                   fdr_family: str, context: Dict, n_units: int = 0) -> Dict:
    row = _blank(STAT_COLUMNS, context)
    row["estimate"] = float(estimate) if (estimate is not None and np.isfinite(estimate)) else np.nan
    row["estimate_name"] = estimate_name
    row["test_name"] = test_name
    row["p_raw"] = keep_p(p_raw)
    row["p_fdr"] = np.nan
    row["fdr_family"] = fdr_family
    row["significant_fdr05"] = np.nan
    row["n_units"] = int(n_units)
    row["test_package"] = TEST_PACKAGE
    return row


def report_spearman(stat_rows, context, rho, p_raw, fdr_family, n_units=0):
    stat_rows.append(pack_statistic(rho, "spearman_rho", "spearman_permutation",
                                    p_raw, fdr_family, context, n_units))


def report_slope(stat_rows, context, slope, p_raw, fdr_family, n_units=0):
    stat_rows.append(pack_statistic(slope, "ols_slope", "slope_bootstrap",
                                    p_raw, fdr_family, context, n_units))


def report_kappa(stat_rows, context, kappa, p_raw, fdr_family, n_units=0):
    stat_rows.append(pack_statistic(kappa, "cohen_kappa", "kappa_permutation",
                                    p_raw, fdr_family, context, n_units))


def report_icc(stat_rows, context, icc, p_raw, fdr_family, n_units=0):
    stat_rows.append(pack_statistic(icc, "icc2", "icc_two_way",
                                    p_raw, fdr_family, context, n_units))


def report_overlap(stat_rows, context, overlap, p_raw, fdr_family, n_units=0):
    stat_rows.append(pack_statistic(overlap, "subspace_overlap",
                                    "overlap_label_permutation", p_raw,
                                    fdr_family, context, n_units))


def report_wilcoxon(stat_rows, context, statistic, p_raw, fdr_family, n_units=0):
    stat_rows.append(pack_statistic(statistic, "wilcoxon_W", "wilcoxon_signed_rank",
                                    p_raw, fdr_family, context, n_units))


def report_friedman(stat_rows, context, statistic, p_raw, fdr_family, n_units=0):
    stat_rows.append(pack_statistic(statistic, "friedman_chi2", "friedman",
                                    p_raw, fdr_family, context, n_units))


def add_fdr(stat_rows: List[Dict], alpha: float = 0.05) -> List[Dict]:
    if not stat_rows:
        return stat_rows
    df = pd.DataFrame(stat_rows)
    n_missing_family = df["fdr_family"].isna().sum()
    if n_missing_family:
        print(f"[report_utils] WARNING: {n_missing_family} statistic row(s) have no "
              f"fdr_family set. pandas groupby silently excludes these from FDR "
              f"correction, so their p_fdr will stay NaN. This indicates a caller "
              f"forgot to pass fdr_family; every report_* call must supply one.")
    df["p_fdr"] = np.nan
    for fam, grp in df.groupby("fdr_family"):
        idx = grp.index.values
        df.loc[idx, "p_fdr"] = bh_fdr(grp["p_raw"].values)
    df["significant_fdr05"] = df["p_fdr"] < alpha
    updated = df.to_dict("records")
    stat_rows.clear()
    stat_rows.extend(updated)
    return stat_rows


def perf_frame(perf_rows: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame(perf_rows, columns=PERF_COLUMNS)


def stat_frame(stat_rows: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame(stat_rows, columns=STAT_COLUMNS)


def _shards_dir(out_dir: str) -> str:
    return os.path.join(out_dir, "shards")


def write_shard(out_dir: str, tag: str, perf_rows: List[Dict],
                stat_rows: List[Dict]) -> None:
    sd = _shards_dir(out_dir)
    os.makedirs(sd, exist_ok=True)
    perf_frame(perf_rows).to_csv(os.path.join(sd, f"perf__{tag}.csv"), index=False)
    stat_frame(stat_rows).to_csv(os.path.join(sd, f"stat__{tag}.csv"), index=False)
    print(f"[shard] wrote {tag}: {len(perf_rows)} perf, {len(stat_rows)} stat rows.")


def shard_exists(out_dir: str, tag: str) -> bool:
    sd = _shards_dir(out_dir)
    return (os.path.exists(os.path.join(sd, f"perf__{tag}.csv"))
            and os.path.exists(os.path.join(sd, f"stat__{tag}.csv")))


def _read_shard_kind(out_dir: str, kind: str) -> pd.DataFrame:
    sd = _shards_dir(out_dir)
    if not os.path.isdir(sd):
        return pd.DataFrame()
    frames = []
    for fn in sorted(os.listdir(sd)):
        if fn.startswith(f"{kind}__") and fn.endswith(".csv"):
            p = os.path.join(sd, fn)
            try:
                frames.append(pd.read_csv(p, float_precision="round_trip"))
            except Exception as e:
                print(f"[shard] could not read {p}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def merge_shards(out_dir: str, experiment: str, alpha: float,
                 extra_stat_from_perf: Optional[Callable[[pd.DataFrame], List[Dict]]] = None
                 ) -> tuple:
    perf = _read_shard_kind(out_dir, "perf")
    stat = _read_shard_kind(out_dir, "stat")
    stat_recs = stat.to_dict("records") if not stat.empty else []
    if extra_stat_from_perf is not None and not perf.empty:
        stat_recs = stat_recs + list(extra_stat_from_perf(perf))
    add_fdr(stat_recs, alpha=alpha)

    os.makedirs(out_dir, exist_ok=True)
    perf_csv = os.path.join(out_dir, f"results_performance_{experiment}.csv")
    stat_csv = os.path.join(out_dir, f"results_statistics_{experiment}.csv")
    perf_out = (perf if not perf.empty else perf_frame([])).reindex(columns=PERF_COLUMNS)
    stat_out = (pd.DataFrame(stat_recs, columns=STAT_COLUMNS) if stat_recs
                else stat_frame([]))
    write_frame_atomic(perf_out, perf_csv)
    write_frame_atomic(stat_out, stat_csv)
    print(f"[merge/{experiment}] {len(perf)} perf, {len(stat_recs)} stat rows "
          f"from {len(os.listdir(_shards_dir(out_dir)))} shard files.")
    print(f"[merge/{experiment}] -> {perf_csv}")
    print(f"[merge/{experiment}] -> {stat_csv}")
    return perf_csv, stat_csv


def _partial_path(out_dir: str, tag: str) -> str:
    return os.path.join(_shards_dir(out_dir), f"partial__{tag}.json")


def load_partial(out_dir: str, tag: str):
    path = _partial_path(out_dir, tag)
    if not os.path.exists(path):
        return set(), [], []
    try:
        with open(path, "r") as f:
            d = json.load(f)
        return set(d["done_units"]), d["perf_rows"], d["stat_rows"]
    except Exception as e:
        print(f"[partial] in-progress save at {path} unreadable ({e}); "
              f"restarting {tag} from the beginning.")
        return set(), [], []


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def save_partial(out_dir: str, tag: str, done_units, perf_rows: List[Dict],
                 stat_rows: List[Dict]) -> None:
    sd = _shards_dir(out_dir)
    os.makedirs(sd, exist_ok=True)
    path = _partial_path(out_dir, tag)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"done_units": sorted(done_units), "perf_rows": perf_rows,
                  "stat_rows": stat_rows}, f, default=_json_default)
    os.replace(tmp, path)


def clear_partial(out_dir: str, tag: str) -> None:
    path = _partial_path(out_dir, tag)
    if os.path.exists(path):
        os.remove(path)


def _claim_path(out_dir: str, tag: str) -> str:
    return os.path.join(_shards_dir(out_dir), f"claim__{tag}")


def _read_claim(path: str):
    with open(path) as f:
        fields = f.read().strip().split(":")
    if len(fields) == 3:
        return fields[0], fields[1], "-"
    if len(fields) == 4:
        return fields[0], fields[1], fields[3]
    raise ValueError(f"unparseable claim file {path}")


def _slurm_job_is_gone(job_id: str) -> bool:
    if not job_id or job_id == "-" or shutil.which("squeue") is None:
        return False
    try:
        p = subprocess.run(["squeue", "-h", "-j", str(job_id), "-o", "%i"],
                           capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return False
    if p.returncode == 0:
        return not p.stdout.strip()
    return "invalid job id" in (p.stderr or "").lower()


def _claim_owner_is_dead(path: str) -> bool:
    try:
        host, pid, job_id = _read_claim(path)
    except (OSError, ValueError):
        return False
    if host != socket.gethostname():
        return _slurm_job_is_gone(job_id)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return True
    except (OSError, ValueError):
        return False
    return False


def claim_unit(out_dir: str, tag: str, stale_after_s: float = 21600.0) -> bool:
    sd = _shards_dir(out_dir)
    os.makedirs(sd, exist_ok=True)
    path = _claim_path(out_dir, tag)
    job_id = os.environ.get("SLURM_JOB_ID", "-").strip() or "-"
    owner = f"{socket.gethostname()}:{os.getpid()}:{time.time():.0f}:{job_id}"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, owner.encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            age = time.time() - os.path.getmtime(path)
        except OSError:
            return False
        if age <= stale_after_s and not _claim_owner_is_dead(path):
            return False
        try:
            with open(path, "w") as f:
                f.write(owner)
            return True
        except OSError:
            return False


def heartbeat_claim(out_dir: str, tag: str) -> None:
    try:
        os.utime(_claim_path(out_dir, tag), None)
    except OSError:
        pass


def release_claim(out_dir: str, tag: str) -> None:
    try:
        os.remove(_claim_path(out_dir, tag))
    except OSError:
        pass

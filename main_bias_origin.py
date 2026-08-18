"""
main_bias_origin.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import sys
import pdb
from typing import Optional
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(line_buffering=True)
    except AttributeError:
        pass

try:
    import resource
    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    _target = 65536 if _hard == resource.RLIM_INFINITY else _hard
    if _soft != resource.RLIM_INFINITY and _soft < _target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (_target, _hard))
except (ImportError, ValueError, OSError) as _e:
    print(f"[main] could not raise the open-file limit ({_e}); if a stage fails "
          f"with 'Too many open files', raise it with ulimit -n in the job script.")

def _job_memory_limit_gb():
    import os as _os
    _best = None
    try:
        with open("/proc/self/cgroup") as _fh:
            _rel = _fh.read().strip().split("\n")[0].split(":")[-1].lstrip("/")
    except OSError:
        _rel = ""
    _parts = [p for p in _rel.split("/") if p]
    for _i in range(len(_parts), -1, -1):
        _base = _os.path.join("/sys/fs/cgroup", *_parts[:_i])
        for _name in ("memory.max", "memory/memory.limit_in_bytes"):
            try:
                with open(_os.path.join(_base, _name)) as _fh:
                    _v = _fh.read().strip()
            except (OSError, ValueError):
                continue
            if not _v or _v == "max":
                continue
            try:
                _b = int(_v)
            except ValueError:
                continue
            if 0 < _b < (1 << 50) and (_best is None or _b / 1024 ** 3 < _best[0]):
                _best = (_b / 1024 ** 3, _os.path.join(_base, _name))
    if _best is not None:
        return _best
    for _k in ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU"):
        _v = _os.environ.get(_k)
        if _v and _v.isdigit():
            _mb = int(_v)
            if _k == "SLURM_MEM_PER_CPU":
                _mb *= int(_os.environ.get("SLURM_CPUS_PER_TASK", 1))
            return _mb / 1024, _k
    return None, None


try:
    _limit_gb, _src = _job_memory_limit_gb()
    if _limit_gb is not None:
        print(f"[main] this job may use {_limit_gb:.1f} GB of memory (from {_src}). "
              f"Stages print their own projected peak; if a stage asks for more "
              f"than this, stop and resubmit with a larger --mem instead of "
              f"waiting for the OOM kill.")
except Exception as _e:
    print(f"[main] could not read this job's memory limit ({_e}).")

def _job_cpu_count():
    import os as _os
    try:
        _n = len(_os.sched_getaffinity(0))
    except AttributeError:
        _n = _os.cpu_count() or 1
    _env = {_k: _os.environ[_k] for _k in
            ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "MKL_NUM_THREADS") if _k in _os.environ}
    return _n, _env


try:
    _ncpu, _cpu_env = _job_cpu_count()
    _note = "" if _ncpu > 1 else (" Only one core: every scikit-learn fit and every "
                                  "bootstrap replicate in this job runs serially. "
                                  "Submit CPU stages with --cpus-per-task.")
    print(f"[main] this job may use {_ncpu} CPU core(s)"
          + (f", environment {_cpu_env}" if _cpu_env else "") + f".{_note}")
    import os as _os2
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        _os2.environ.setdefault(_var, str(_ncpu))
except Exception as _e:
    print(f"[main] could not read this job's CPU allocation ({_e}).")


GLOBAL_CONFIG_PATH = "/PATH/bias_origin/config/config.yaml"


def main_build_mimic_demographics(cfg_path: str = GLOBAL_CONFIG_PATH):
    from data_loader.build_mimic_demographics import main_build_mimic_demographics
    main_build_mimic_demographics(cfg_path)


def main_build_cxr_pool(cfg_path: str = GLOBAL_CONFIG_PATH):
    from data_loader.build_cxr_pool import main_build_cxr_pool
    main_build_cxr_pool(cfg_path)


def main_build_cxr_paired_reports(cfg_path: str = GLOBAL_CONFIG_PATH):
    from data_loader.build_cxr_paired_reports import main_build_cxr_paired_reports
    main_build_cxr_paired_reports(cfg_path)


def main_build_cxr_scrubbed_reports(cfg_path: str = GLOBAL_CONFIG_PATH):
    from data_loader.build_cxr_paired_reports import main_build_cxr_scrubbed_reports
    main_build_cxr_scrubbed_reports(cfg_path)


def main_build_cxr_amplified_reports(cfg_path: str = GLOBAL_CONFIG_PATH):
    from data_loader.build_cxr_paired_reports import main_build_cxr_amplified_reports
    main_build_cxr_amplified_reports(cfg_path)


def main_build_derm_pool(cfg_path: str = GLOBAL_CONFIG_PATH):
    from data_loader.build_derm_pool import main_build_derm_pool
    main_build_derm_pool(cfg_path)


def main_build_fairvision_pool(cfg_path: str = GLOBAL_CONFIG_PATH):
    from data_loader.build_fairvision_pool import main_build_fairvision_pool
    main_build_fairvision_pool(cfg_path)


def main_preprocess_derm_fundus(cfg_path: str = GLOBAL_CONFIG_PATH):
    from config.serde import read_config
    from data_loader.preprocess_utils import preprocess_manifest
    cfg = read_config(cfg_path)["BiasOrigin"]
    preprocess_manifest(cfg["derm"]["pool_manifest_csv"], cfg["derm"]["image_root"],
                        num_workers=int(cfg["embeddings"].get("num_workers", 8)), tag="/derm")
    preprocess_manifest(cfg["fundus"]["pool_manifest_csv"], cfg["fundus"]["image_root"],
                        num_workers=int(cfg["embeddings"].get("num_workers", 8)), tag="/fundus")


def main_build_subgroup_counts(cfg_path: str = GLOBAL_CONFIG_PATH):
    from data_loader.build_subgroup_counts import main_build_subgroup_counts
    main_build_subgroup_counts(cfg_path)


def main_extract_image_embeddings(cfg_path: str = GLOBAL_CONFIG_PATH, encoder_names=None):
    from encoders.extract_embeddings import main_extract_image_embeddings
    main_extract_image_embeddings(cfg_path, encoder_names)


def main_build_training_mixtures(cfg_path: str = GLOBAL_CONFIG_PATH):
    from controlled.build_training_mixtures import main_build_training_mixtures
    main_build_training_mixtures(cfg_path)


def main_train_controlled_run(run_id: str, cfg_path: str = GLOBAL_CONFIG_PATH):
    from controlled.train_encoder import run_one
    run_one(run_id, cfg_path)


def main_e1(cfg_path: str = GLOBAL_CONFIG_PATH):
    from experiments.e1_driver import main_e1
    main_e1(cfg_path)


def main_e2_encoder(encoder: str, cfg_path: str = GLOBAL_CONFIG_PATH,
                    attribute: str = "race_grp"):
    from experiments.e2_ceiling import main_e2_encoder
    main_e2_encoder(encoder, cfg_path, attribute=attribute)


def main_e2_merge(cfg_path: str = GLOBAL_CONFIG_PATH):
    from experiments.e2_ceiling import main_e2_merge
    main_e2_merge(cfg_path)


def main_e3(cfg_path: str = GLOBAL_CONFIG_PATH):
    from experiments.e3_mechanism import main_e3
    main_e3(cfg_path)


def main_e4_encoder(modality: str, encoder: str, cfg_path: str = GLOBAL_CONFIG_PATH):
    from experiments.e4_generalization import main_e4_encoder
    main_e4_encoder(modality, encoder, cfg_path)


def main_e4_merge(cfg_path: str = GLOBAL_CONFIG_PATH):
    from experiments.e4_generalization import main_e4_merge
    main_e4_merge(cfg_path)


def main_e5(cfg_path: str = GLOBAL_CONFIG_PATH):
    from experiments.e5_crossmodal import main_e5
    main_e5(cfg_path)


def main_e6_cell(encoder: str, level: str, cfg_path: str = GLOBAL_CONFIG_PATH):
    from experiments.e6_finetuning import main_e6_cell
    main_e6_cell(encoder, level, cfg_path)


def main_e6_merge(cfg_path: str = GLOBAL_CONFIG_PATH):
    from experiments.e6_finetuning import main_e6_merge
    main_e6_merge(cfg_path)


def main_e7(cfg_path: str = GLOBAL_CONFIG_PATH):
    from experiments.e7_theory import main_e7
    main_e7(cfg_path)


def main_e8(cfg_path: str = GLOBAL_CONFIG_PATH):
    from experiments.e8_gapnull import main_e8
    main_e8(cfg_path)


def main_e9_cell(encoder: str, mode: str, strength: float,
                 cfg_path: str = GLOBAL_CONFIG_PATH):
    from experiments.e9_positive_control import main_e9_cell
    main_e9_cell(encoder, mode, strength, cfg_path)


def main_e3_seed_partial(cfg_path: str = GLOBAL_CONFIG_PATH, force: bool = False):
    from experiments.e3_mechanism import seed_entanglement_partial
    seed_entanglement_partial(cfg_path, force=force)


def main_e10_encoder(encoder: str, cfg_path: str = GLOBAL_CONFIG_PATH):
    from experiments.e10_acquisition import main_e10_encoder
    main_e10_encoder(encoder, cfg_path)


def main_e10_merge(cfg_path: str = GLOBAL_CONFIG_PATH):
    from experiments.e10_acquisition import main_e10_merge
    main_e10_merge(cfg_path)


def main_e3_bridge_run(run_id: str, cfg_path: str = GLOBAL_CONFIG_PATH):
    from experiments.e3_mechanism import main_e3_bridge_run
    main_e3_bridge_run(run_id, cfg_path)


def main_e9_probe(encoder: str, mode: str = "entangled",
                  finding: Optional[str] = None,
                  cfg_path: str = GLOBAL_CONFIG_PATH):
    from experiments.e9_positive_control import main_e9_probe
    main_e9_probe(encoder, cfg_path, mode=mode, finding=finding)


def main_e9_merge(cfg_path: str = GLOBAL_CONFIG_PATH):
    from experiments.e9_positive_control import main_e9_merge
    main_e9_merge(cfg_path)


def main_build_final_tables(cfg_path: str = GLOBAL_CONFIG_PATH):
    from merge.build_final_tables import main_build_final_tables
    main_build_final_tables(cfg_path)


def main_build_published_claims(cfg_path: str = GLOBAL_CONFIG_PATH):
    from data_loader.build_published_claims import main_build_published_claims as _run
    return _run(cfg_path)


def main_e11(cfg_path: str = GLOBAL_CONFIG_PATH, force: bool = False):
    from experiments.e11_published import main_e11 as _run
    _run(cfg_path, force=force)


def main_e11_merge(cfg_path: str = GLOBAL_CONFIG_PATH):
    from experiments.e11_published import main_e11_merge as _run
    return _run(cfg_path)

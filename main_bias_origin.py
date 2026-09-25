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


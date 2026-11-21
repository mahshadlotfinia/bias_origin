"""
Inference/report_utils.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

from typing import Callable, Dict

import json
import os
import shutil
import socket
import subprocess
import time
import uuid

import pandas as pd


class MissingInput(Exception):
    pass


def write_frame_atomic(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _shards_dir(out_dir: str) -> str:
    return os.path.join(out_dir, "shards")


_CLAIM_STALE_AFTER_S = 21600.0
_TAKEOVER_LOCK_S = 600.0
_HEARTBEAT_EVERY_S = 60.0
_LIVE_CLAIMS: set = set()
_CLAIM_TOKENS: Dict[str, str] = {}
_LAST_BEAT: Dict[str, float] = {}
_LINK_WARNED = False


def _claim_path(out_dir: str, tag: str) -> str:
    return os.path.join(_shards_dir(out_dir), f"claim__{tag}")


def _read_claim(path: str) -> Dict:
    try:
        with open(path) as f:
            text = f.read().strip()
    except OSError:
        return {}
    try:
        out = json.loads(text)
        return out if isinstance(out, dict) else {}
    except ValueError:
        fields = text.split(":")
        if len(fields) in (3, 4):
            job = fields[3] if len(fields) == 4 and fields[3] != "-" else ""
            return {"host": fields[0], "pid": fields[1], "slurm_job_id": job, "legacy": text}
        return {}


def _slurm_job_state(job_id: str) -> str:
    if not job_id or job_id == "-" or shutil.which("squeue") is None:
        return "unknown"
    try:
        p = subprocess.run(["squeue", "-h", "-j", str(job_id), "-o", "%i"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if p.returncode != 0:
        return "gone" if "invalid job id" in (p.stderr or "").lower() else "unknown"
    return "gone" if not p.stdout.strip() else "alive"


def _owner_state(claim: Dict) -> str:
    state = _slurm_job_state(str(claim.get("slurm_job_id", "") or ""))
    if state != "unknown" or claim.get("host") != socket.gethostname():
        return state
    try:
        pid = int(claim.get("pid", 0))
    except (TypeError, ValueError):
        return "unknown"
    if pid <= 0:
        return "unknown"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "gone"
    except OSError:
        return "unknown"
    return "alive"


def _create_claim(path: str, token: str) -> bool:
    global _LINK_WARNED
    tmp = f"{path}.{token}.tmp"
    with open(tmp, "w") as f:
        json.dump({"host": socket.gethostname(), "pid": os.getpid(), "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
                   "claimed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "token": token}, f, ensure_ascii=True)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)
        return True
    except FileExistsError:
        return False
    except OSError:
        if not _LINK_WARNED:
            _LINK_WARNED = True
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as f, open(tmp) as src:
            f.write(src.read())
        return True
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _take_over(path: str, expired: Dict, token: str) -> bool:
    lock = f"{path}.takeover"
    if not _create_claim(lock, token):
        try:
            if time.time() - os.path.getmtime(lock) > _TAKEOVER_LOCK_S:
                os.remove(lock)
        except OSError:
            pass
        return False
    try:
        if _read_claim(path) != expired:
            return False
        try:
            os.remove(path)
        except OSError:
            return False
        return _create_claim(path, token)
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


def claim_unit(out_dir: str, tag: str, stale_after_s: float = _CLAIM_STALE_AFTER_S) -> bool:
    os.makedirs(_shards_dir(out_dir), exist_ok=True)
    path = _claim_path(out_dir, tag)
    token = uuid.uuid4().hex
    if not _create_claim(path, token):
        claim = _read_claim(path)
        try:
            age = time.time() - os.path.getmtime(path)
        except OSError:
            return False
        owner = _owner_state(claim)
        if owner == "alive" or (owner == "unknown" and age < stale_after_s):
            return False
        if not _take_over(path, claim, token):
            return False
    _LIVE_CLAIMS.add(path)
    _CLAIM_TOKENS[path] = token
    _LAST_BEAT[path] = time.time()
    return True


def heartbeat_claim(out_dir: str, tag: str) -> None:
    path = _claim_path(out_dir, tag)
    token = _CLAIM_TOKENS.get(path)
    if token is None or time.time() - _LAST_BEAT.get(path, 0.0) < _HEARTBEAT_EVERY_S:
        return
    if _read_claim(path).get("token") == token:
        try:
            os.utime(path, None)
            _LAST_BEAT[path] = time.time()
        except OSError:
            pass


def _remove_own_claim(path: str) -> None:
    token = _CLAIM_TOKENS.pop(path, None)
    _LIVE_CLAIMS.discard(path)
    _LAST_BEAT.pop(path, None)
    if token is None or not os.path.exists(path):
        return
    if _read_claim(path).get("token") != token:
        return
    try:
        os.remove(path)
    except OSError:
        pass


def release_claim(out_dir: str, tag: str) -> None:
    _remove_own_claim(_claim_path(out_dir, tag))


def claim_or_wait(out_dir: str, tag: str, is_done: Callable[[], bool], stale_after_s: float = 1800.0, poll_s: float = 10.0) -> bool:
    last_note, pause = 0.0, 1.0
    while True:
        if is_done():
            return False
        if claim_unit(out_dir, tag, stale_after_s):
            if is_done():
                release_claim(out_dir, tag)
                return False
            return True
        if time.time() - last_note > 60.0:
            last_note = time.time()
        time.sleep(pause)
        pause = min(2.0 * pause, poll_s)

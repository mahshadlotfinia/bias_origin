"""
controlled/train_encoder.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config.serde import read_config
from data_loader.build_cxr_paired_reports import assert_derived_manifest_complete
from data_loader.build_utils import read_csv_defensively
from data_loader.cxr_harmonization import (
    CANONICAL_PATHOLOGY_FINDINGS, resolve_cxr_image_path,
    reroot_mimic_report_path,
)

import warnings
warnings.filterwarnings("ignore")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_run_id(run_id: str) -> Dict[str, str]:
    parts = run_id.split("__")
    if len(parts) != 6 or parts[0] != "cxr":
        raise ValueError(f"[train] bad run_id '{run_id}'. Expected "
                         f"cxr__<obj>__<backbone>__<init>__<data_comp>__seed<N>.")
    _, objective, backbone, init, data_comp, seed_tok = parts
    return {"objective": objective, "backbone": backbone, "init": init,
            "data_comp": data_comp, "seed": int(seed_tok.replace("seed", ""))}


def _timm_hub(hf_id: str) -> str:
    s = str(hf_id)
    return s if s.startswith("hf-hub:") else f"hf-hub:{s}"


def _with_timeout(fn, timeout_s: float, label: str):
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn)
    try:
        result = fut.result(timeout=timeout_s)
        ex.shutdown(wait=False)
        return result
    except FutTimeout:
        raise TimeoutError(
            f"{label} did not complete within {timeout_s:.0f}s. This is "
            f"almost always a stuck network call (egress to huggingface.co is "
            f"slow, blocked, or rate-limited on this machine). The background "
            f"attempt is abandoned; Python cannot force-kill it, but this call "
            f"is giving up and failing clearly instead of hanging silently.")


def _init_hf_id(init: str, cfg: dict) -> str:
    spec = cfg["encoder_panel"]["image"].get(init, {})
    hf_id = spec.get("hf_id")
    if not hf_id or str(hf_id).endswith("_LOCAL"):
        raise RuntimeError(f"[train] controlled init '{init}' has no usable hf_id.")
    return str(hf_id)


def _maybe_enable_grad_checkpointing(model, cfg: dict):
    if not cfg["controlled"].get("gradient_checkpointing", True):
        return
    if not hasattr(model, "gradient_checkpointing_enable"):
        return
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()


def _load_hf_offline_first(loader_fn, hf_id: str, label: str, timeout_s: float = 90.0):
    t0 = time.time()
    try:
        result = _with_timeout(lambda: loader_fn(hf_id, local_files_only=True),
                               timeout_s, f"offline load of {label} '{hf_id}'")
        print(f"[hf] {label} '{hf_id}': loaded offline in {time.time()-t0:.1f}s.")
        return result
    except Exception as e:
        print(f"[hf] offline load of {label} '{hf_id}' failed after {time.time()-t0:.1f}s "
             f"({e}); falling back to a network-allowed load (timeout {timeout_s:.0f}s).")
        t1 = time.time()
        result = _with_timeout(lambda: loader_fn(hf_id), timeout_s,
                               f"network load of {label} '{hf_id}'")
        print(f"[hf] {label} '{hf_id}': loaded over the network in {time.time()-t1:.1f}s.")
        return result


def build_backbone(backbone: str, init: str, cfg: dict, seed: int):
    from transformers import AutoModel
    torch.manual_seed(seed); np.random.seed(seed)
    init_id = _init_hf_id(init, cfg)
    token = cfg.get("hf_token")
    try:
        model = _load_hf_offline_first(
            lambda hf_id, **kw: AutoModel.from_pretrained(hf_id, token=token,
                                                          trust_remote_code=True, **kw),
            init_id, "controlled backbone")
    except Exception as e:
        raise RuntimeError(
            f"[train] failed to load DINOv3 init '{init_id}' for backbone "
            f"{backbone}: {e}. This must be a loadable HF model; controlled runs "
            f"cannot proceed from a random init.") from e
    _maybe_enable_grad_checkpointing(model, cfg)
    return model, int(model.config.hidden_size)


def get_embed_dim(model) -> int:
    cfg = getattr(model, "config", None)
    if cfg is not None and getattr(cfg, "hidden_size", None):
        return int(cfg.hidden_size)
    return int(model.embed_dim)


def forward_features_cls(model, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "forward_features"):
        feats = model.forward_features(x)
        return feats[:, 0, :] if feats.ndim == 3 else feats
    out = model(pixel_values=x)
    pooled = getattr(out, "pooler_output", None)
    return pooled if pooled is not None else out.last_hidden_state[:, 0]


def _ssl_transform(res: int):
    from torchvision import transforms as T
    return T.Compose([
        T.RandomResizedCrop(res, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4)], p=0.5),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _train_transform(res: int):
    from torchvision import transforms as T
    return T.Compose([
        T.RandomResizedCrop(res, scale=(0.5, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(res: int):
    from torchvision import transforms as T
    return T.Compose([
        T.Resize(int(round(res * 256 / 224))),
        T.CenterCrop(res),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class _ImageMixtureDataset(Dataset):
    def __init__(self, mixture_csv, cfg, transform, with_labels: bool = False,
                 two_views: bool = False):
        self.df = read_csv_defensively(mixture_csv).reset_index(drop=True)
        self.transform = transform
        self.with_labels = with_labels
        self.two_views = two_views
        self.site_root = {s: sc["image_root"] for s, sc in cfg["cxr"]["sites"].items()}
        self.res = int(cfg.get("target_resolution", 224))
        self.findings = [f for f in CANONICAL_PATHOLOGY_FINDINGS if f in self.df.columns]

    def __len__(self):
        return len(self.df)

    def _pil(self, row):
        site = str(row.get("site") or row.get("dataset") or "mimic")
        path = resolve_cxr_image_path(site, self.site_root.get(site, ""),
                                      str(row.get("image_key", "")),
                                      row.get("image_subdir"), self.res)
        return Image.open(path).convert("RGB")

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = self._pil(row)
        if self.two_views:
            return self.transform(img), self.transform(img)
        x = self.transform(img)
        if not self.with_labels:
            return x
        y = np.array([row[f] for f in self.findings], dtype=np.float32)
        return x, torch.from_numpy(y)


class _PairedTextDataset(Dataset):
    def __init__(self, mixture_csv, reports_csv, cfg, transform):
        mix = read_csv_defensively(mixture_csv)
        rep = read_csv_defensively(reports_csv)
        keep = [c for c in ["case_id", "report_path", "report_text"] if c in rep.columns]
        self.df = mix.merge(rep[keep], on="case_id", how="inner").reset_index(drop=True)
        self.transform = transform
        self.site_root = {s: sc["image_root"] for s, sc in cfg["cxr"]["sites"].items()}
        self.mimic_root = self.site_root.get("mimic", "")
        self.res = int(cfg.get("target_resolution", 224))
        if len(self.df) == 0:
            def _srcs(d):
                for c in ("dataset", "site"):
                    if c in d.columns:
                        return dict(d[c].astype(str).value_counts())
                return {}
            raise RuntimeError(
                "[train/clip] no image-report pairs after joining the mixture "
                "with the reports manifest on case_id. "
                f"mixture {os.path.basename(mixture_csv)}: {len(mix)} rows "
                f"{_srcs(mix)}; reports {os.path.basename(reports_csv)}: "
                f"{len(rep)} rows {_srcs(rep)}. If a source the mixture needs is "
                "absent from the reports manifest, rebuild that manifest on THIS "
                "machine; if both hold it, the case_ids disagree.")
        self._assert_text_present()

    def _reroot(self, p: str) -> str:
        return reroot_mimic_report_path(p, self.mimic_root)

    def _read(self, path: str) -> str:
        if path and os.path.exists(path):
            try:
                with open(path, "r", errors="ignore") as f:
                    return f.read()
            except OSError:
                return ""
        return ""

    def _text(self, row) -> str:
        t = row.get("report_text")
        if isinstance(t, str) and t.strip():
            return t
        return self._read(self._reroot(row.get("report_path")))

    def _assert_text_present(self, n_sample: int = 300, min_frac: float = 0.1):
        idx = np.linspace(0, len(self.df) - 1, min(n_sample, len(self.df))).astype(int)
        nonempty = sum(1 for i in idx if self._text(self.df.iloc[int(i)]).strip())
        if nonempty / len(idx) < min_frac:
            inline = int(self.df["report_text"].apply(
                lambda t: isinstance(t, str) and t.strip() != "").sum())
            has_path = self.df["report_path"].apply(lambda p: isinstance(p, str) and p.strip() != "")
            sample = self.df.loc[has_path, "report_path"].head(200)
            raw_exist = int(sum(os.path.exists(p) for p in sample))
            rerooted_exist = int(sum(os.path.exists(self._reroot(p)) for p in sample))
            raise RuntimeError(
                f"[train/clip] report text empty for ~all sampled rows "
                f"({nonempty}/{len(idx)}). Inline report_text: {inline}/{len(self.df)}. "
                f"report_path as-stored exists: {raw_exist}/{len(sample)}; rerooted "
                f"onto mimic image_root '{self.mimic_root}' exists: "
                f"{rerooted_exist}/{len(sample)}. If rerooted is also 0, the MIMIC "
                f"report files are not under that image_root on this machine.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        site = str(row.get("site") or row.get("dataset") or "mimic")
        path = resolve_cxr_image_path(site, self.site_root.get(site, ""),
                                      str(row.get("image_key", "")),
                                      row.get("image_subdir"), self.res)
        img = self.transform(Image.open(path).convert("RGB"))
        return img, self._text(row)


def _optimizer_and_sched(params, cfg, steps_total):
    opt = torch.optim.AdamW(params, lr=float(cfg["controlled"]["lr"]),
                            weight_decay=float(cfg["controlled"]["weight_decay"]))
    warmup = int(float(cfg["controlled"].get("warmup_fraction", 0.1)) * steps_total)

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, steps_total - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return opt, sched


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def _bf16_ctx(cfg, device):
    use = bool(cfg["controlled"].get("bf16", True)) and device == "cuda"
    return torch.autocast("cuda", dtype=torch.bfloat16) if use else _nullctx()


def _progress_ckpt_path(cfg: dict, run_id: str) -> str:
    return os.path.join(cfg["controlled"]["ckpts_dir"], f"{run_id}.inprogress.pt")


def _save_progress(path: str, epoch: int, **states) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"epoch": epoch, **states}, tmp)
    os.replace(tmp, path)


def _load_progress(path: str):
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"[train] in-progress checkpoint at {path} could not be loaded "
              f"({e}); starting this run from epoch 0.")
        return None


def _try_resume(prog: dict, run_id: str, tag: str, **components) -> int:
    try:
        for obj, key in components.values():
            obj.load_state_dict(prog[key])
        print(f"[train/{tag}] resuming {run_id} from epoch {prog['epoch']}.")
        return int(prog["epoch"])
    except (RuntimeError, KeyError) as e:
        print(f"[train/{tag}] in-progress checkpoint for {run_id} does not match "
              f"the current model (likely written before a training-code change); "
              f"discarding it and starting this run from epoch 0. ({e})")
        return 0


def _loaders(dataset, cfg):
    bs = int(cfg["controlled"]["batch_size"])
    eff = int(cfg["controlled"].get("effective_batch_size", bs))
    accum = max(1, eff // bs)
    nw = int(cfg["controlled"].get("num_workers",
                                   cfg["embeddings"].get("num_workers", 8)))
    dl = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True,
                    num_workers=nw, persistent_workers=nw > 0,
                    pin_memory=True, prefetch_factor=4 if nw > 0 else None)
    return dl, accum


def _info_nce(z1: torch.Tensor, z2: torch.Tensor, temp: float) -> torch.Tensor:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.t() / temp
    labels = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def train_ssl(model, dataset, cfg, device, run_id: str) -> nn.Module:
    embed_dim = get_embed_dim(model)
    proj = nn.Sequential(nn.Linear(embed_dim, 2048), nn.ReLU(inplace=True),
                         nn.Linear(2048, 256)).to(device)
    model = model.to(device)
    temp = float(cfg["controlled"].get("ssl_temp", 0.1))
    epochs = int(cfg["controlled"]["epochs"])
    dl, accum = _loaders(dataset, cfg)
    params = list(model.parameters()) + list(proj.parameters())
    opt, sched = _optimizer_and_sched(params, cfg, epochs * len(dl))

    ckpt_path = _progress_ckpt_path(cfg, run_id)
    prog = _load_progress(ckpt_path)
    start_epoch = 0
    if prog is not None:
        start_epoch = _try_resume(prog, run_id, "ssl", model=(model, "model"),
                                  proj=(proj, "proj"), opt=(opt, "opt"),
                                  sched=(sched, "sched"))

    model.train(); proj.train()
    for ep in range(start_epoch, epochs):
        running = 0.0
        pbar = tqdm(dl, desc=f"[ssl] {run_id} ep{ep+1}/{epochs}", unit="batch")
        for it, (v1, v2) in enumerate(pbar):
            v1, v2 = v1.to(device, non_blocking=True), v2.to(device, non_blocking=True)
            with _bf16_ctx(cfg, device):
                z1 = proj(forward_features_cls(model, v1))
                z2 = proj(forward_features_cls(model, v2))
                loss = _info_nce(z1, z2, temp) / accum
            loss.backward()
            if (it + 1) % accum == 0:
                opt.step(); sched.step(); opt.zero_grad()
            running += loss.item() * accum
            pbar.set_postfix(loss=f"{running/(it+1):.4f}")
        print(f"[train/ssl] epoch {ep+1}/{epochs} loss={running/len(dl):.4f}")
        _save_progress(ckpt_path, ep + 1, model=model.state_dict(),
                       proj=proj.state_dict(), opt=opt.state_dict(),
                       sched=sched.state_dict())
    return model


def train_supervised(model, dataset, cfg, device, run_id: str) -> nn.Module:
    n_lab = int(cfg["controlled"].get("cxr_n_labels", len(CANONICAL_PATHOLOGY_FINDINGS)))
    head = nn.Linear(get_embed_dim(model), n_lab).to(device)
    model = model.to(device)
    epochs = int(cfg["controlled"]["epochs"])
    dl, accum = _loaders(dataset, cfg)
    params = list(model.parameters()) + list(head.parameters())
    opt, sched = _optimizer_and_sched(params, cfg, epochs * len(dl))
    bce = nn.BCEWithLogitsLoss(reduction="none")

    ckpt_path = _progress_ckpt_path(cfg, run_id)
    prog = _load_progress(ckpt_path)
    start_epoch = 0
    if prog is not None:
        start_epoch = _try_resume(prog, run_id, "supervised", model=(model, "model"),
                                  head=(head, "head"), opt=(opt, "opt"),
                                  sched=(sched, "sched"))

    model.train(); head.train()
    for ep in range(start_epoch, epochs):
        running = 0.0
        pbar = tqdm(dl, desc=f"[supervised] {run_id} ep{ep+1}/{epochs}", unit="batch")
        for it, (imgs, y) in enumerate(pbar):
            imgs, y = imgs.to(device), y.to(device)
            mask = torch.isfinite(y).float()
            y0 = torch.nan_to_num(y, nan=0.0)
            with _bf16_ctx(cfg, device):
                feats = forward_features_cls(model, imgs)
                logits = head(feats)
                loss = (bce(logits, y0) * mask).sum() / mask.sum().clamp(min=1) / accum
            loss.backward()
            if (it + 1) % accum == 0:
                opt.step(); sched.step(); opt.zero_grad()
            running += loss.item() * accum
            pbar.set_postfix(loss=f"{running/(it+1):.4f}")
        print(f"[train/supervised] epoch {ep+1}/{epochs} loss={running/len(dl):.4f}")
        _save_progress(ckpt_path, ep + 1, model=model.state_dict(),
                       head=head.state_dict(), opt=opt.state_dict(),
                       sched=sched.state_dict())
    return model


def train_clip(model, dataset, cfg, device, run_id: str) -> nn.Module:
    from transformers import AutoModel, AutoTokenizer
    text_id = cfg["controlled"].get("text_encoder", "emilyalsentzer/Bio_ClinicalBERT")
    tok = _load_hf_offline_first(AutoTokenizer.from_pretrained, text_id, "text tokenizer")
    text_model = _load_hf_offline_first(AutoModel.from_pretrained, text_id, "text model").to(device)
    _maybe_enable_grad_checkpointing(text_model, cfg)
    proj_dim = 256
    img_proj = nn.Linear(get_embed_dim(model), proj_dim).to(device)
    txt_proj = nn.Linear(text_model.config.hidden_size, proj_dim).to(device)
    temp = float(cfg["controlled"].get("clip_temp", 0.07))
    model = model.to(device)

    def collate(batch):
        imgs = torch.stack([b[0] for b in batch])
        texts = [b[1] for b in batch]
        return imgs, texts

    bs = int(cfg["controlled"]["batch_size"])
    nw = int(cfg["controlled"].get("num_workers",
                                   cfg["embeddings"].get("num_workers", 8)))
    dl = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True,
                    num_workers=nw, collate_fn=collate,
                    persistent_workers=nw > 0, pin_memory=True,
                    prefetch_factor=4 if nw > 0 else None)
    epochs = int(cfg["controlled"]["epochs"])
    params = list(model.parameters()) + list(text_model.parameters()) + \
             list(img_proj.parameters()) + list(txt_proj.parameters())
    opt, sched = _optimizer_and_sched(params, cfg, epochs * len(dl))

    ckpt_path = _progress_ckpt_path(cfg, run_id)
    prog = _load_progress(ckpt_path)
    start_epoch = 0
    if prog is not None:
        start_epoch = _try_resume(prog, run_id, "clip", model=(model, "model"),
                                  text_model=(text_model, "text_model"),
                                  img_proj=(img_proj, "img_proj"),
                                  txt_proj=(txt_proj, "txt_proj"),
                                  opt=(opt, "opt"), sched=(sched, "sched"))

    model.train(); text_model.train()
    for ep in range(start_epoch, epochs):
        running = 0.0
        pbar = tqdm(dl, desc=f"[clip] {run_id} ep{ep+1}/{epochs}", unit="batch")
        for it, (imgs, texts) in enumerate(pbar):
            imgs = imgs.to(device)
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=256).to(device)
            with _bf16_ctx(cfg, device):
                img_e = F.normalize(img_proj(forward_features_cls(model, imgs)), dim=-1)
                txt_out = text_model(**enc).last_hidden_state[:, 0, :]
                txt_e = F.normalize(txt_proj(txt_out), dim=-1)
                logits = img_e @ txt_e.t() / temp
                labels = torch.arange(len(imgs), device=device)
                loss = 0.5 * (F.cross_entropy(logits, labels) +
                              F.cross_entropy(logits.t(), labels))
            loss.backward()
            opt.step(); sched.step(); opt.zero_grad()
            running += loss.item()
            pbar.set_postfix(loss=f"{running/(it+1):.4f}")
        print(f"[train/clip] epoch {ep+1}/{epochs} loss={running/len(dl):.4f}")
        _save_progress(ckpt_path, ep + 1, model=model.state_dict(),
                       text_model=text_model.state_dict(),
                       img_proj=img_proj.state_dict(), txt_proj=txt_proj.state_dict(),
                       opt=opt.state_dict(), sched=sched.state_dict())
    return model


def _mixture_and_reports(meta: Dict, cfg: dict) -> Tuple[str, Optional[str]]:
    mix_dir = cfg["controlled"]["mixtures_dir"]
    if meta["data_comp"] == "balanced":
        mixture = os.path.join(mix_dir, "cxr_balanced.csv")
    else:
        mixture = os.path.join(mix_dir, "cxr_natural.csv")
    reports = None
    if meta["objective"] == "image_text":
        if meta["data_comp"] == "scrubbed":
            reports = cfg["cxr"]["reports"]["scrubbed_reports_csv"]
        elif meta["data_comp"] == "amplified":
            reports = cfg["cxr"]["reports"]["amplified_reports_csv"]
        else:
            reports = cfg["cxr"]["paired_reports_csv"]
    return mixture, reports


def run_one(run_id: str, global_config_path: str) -> str:
    cfg = read_config(global_config_path)["BiasOrigin"]
    ckpt_dir = cfg["controlled"]["ckpts_dir"]
    out = os.path.join(ckpt_dir, f"{run_id}.pt")
    if os.path.exists(out):
        print(f"[train] {run_id} already complete -> {out}; skipping.")
        return out

    meta = parse_run_id(run_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(meta["seed"]); np.random.seed(meta["seed"])

    model, _embed_dim = build_backbone(meta["backbone"], meta["init"], cfg, meta["seed"])
    res = int(cfg.get("target_resolution", 224))
    mixture, reports = _mixture_and_reports(meta, cfg)
    if not os.path.exists(mixture):
        raise FileNotFoundError(f"[train] mixture not found: {mixture}. Run build_training_mixtures.")

    obj = meta["objective"]
    if obj == "ssl":
        ds = _ImageMixtureDataset(mixture, cfg, _ssl_transform(res), two_views=True)
        model = train_ssl(model, ds, cfg, device, run_id)
    elif obj == "supervised":
        ds = _ImageMixtureDataset(mixture, cfg, _train_transform(res), with_labels=True)
        model = train_supervised(model, ds, cfg, device, run_id)
    elif obj == "image_text":
        if not reports or not os.path.exists(reports):
            raise FileNotFoundError(f"[train] reports manifest not found: {reports}.")
        if meta["data_comp"] in ("scrubbed", "amplified"):
            assert_derived_manifest_complete(
                reports, cfg["cxr"]["paired_reports_csv"], meta["data_comp"])
        ds = _PairedTextDataset(mixture, reports, cfg, _train_transform(res))
        model = train_clip(model, ds, cfg, device, run_id)
    else:
        raise ValueError(f"[train] unknown objective '{obj}'")

    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({"run_id": run_id, "meta": meta,
                "backbone_hf_id": _init_hf_id(meta["init"], cfg),
                "state_dict": model.state_dict()}, out)
    print(f"[train] saved {run_id} -> {out}")

    prog_path = _progress_ckpt_path(cfg, run_id)
    if os.path.exists(prog_path):
        os.remove(prog_path)
    return out


def list_controlled_runs(global_config_path: str) -> List[str]:
    cfg = read_config(global_config_path)["BiasOrigin"]["controlled"]
    inits = cfg["inits"]
    runs = []
    for obj in cfg["objectives"]:
        for bk in cfg["backbones"]:
            for dc in cfg["data_compositions"]:
                runs.append(f"cxr__{obj}__{bk}__{inits[bk]}__{dc}__seed0")
    for bk in cfg["backbones"]:
        runs.append(f"cxr__{cfg['scrubbed_objective']}__{bk}__{inits[bk]}__scrubbed__seed0")
    if cfg.get("run_amplified", True):
        for bk in cfg["backbones"]:
            runs.append(f"cxr__{cfg['amplified_objective']}__{bk}__{inits[bk]}__amplified__seed0")
    sv = cfg["seed_variance"]
    for obj in sv["objectives"]:
        for sd in sv["seeds"]:
            runs.append(f"cxr__{obj}__{sv['backbone']}__{inits[sv['backbone']]}__{sv['data_comp']}__seed{sd}")
    return runs

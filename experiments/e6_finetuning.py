"""
experiments/e6_finetuning.py
Created on July 01, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import math
import os
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
from data_loader.build_utils import read_csv_defensively
from data_loader.cxr_harmonization import (
    CANONICAL_PATHOLOGY_FINDINGS, resolve_cxr_image_path,
)
from analysis.heads import split_masks, threshold_at_sensitivity
from analysis.fairness_metrics import auroc, compute_fairness_point
from Inference.stats_utils import (
    cluster_bootstrap, spearman_with_perm, resolve_n_jobs,
)
from Inference import report_utils as R
from controlled.train_encoder import (
    forward_features_cls, get_embed_dim, eval_transform, _load_hf_offline_first,
    _save_progress, _load_progress, _try_resume,
)

import warnings
warnings.filterwarnings("ignore")

FINDINGS = CANONICAL_PATHOLOGY_FINDINGS
LEVEL_DEPTH = {"linear": 0, "last_block": 1, "lora": 2, "full": 3}
SUMMARY = ["auroc_overall", "auroc_gap", "es_auc", "tpr_gap", "fpr_gap",
           "fnr_gap", "ece_gap"]


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.A = nn.Parameter(torch.zeros(self.rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(x, self.A), self.B)


def _blocks_of(model):
    b = getattr(model, "blocks", None)
    if b is not None:
        return b
    enc = getattr(model, "encoder", None)
    return getattr(enc, "layer", None) if enc is not None else None


def _final_norm_of(model):
    n = getattr(model, "norm", None)
    return n if n is not None else getattr(model, "layernorm", None)


def _attn_linears_of(block) -> List[Tuple[nn.Module, str, nn.Linear]]:
    out: List[Tuple[nn.Module, str, nn.Linear]] = []
    attn = getattr(block, "attn", None)
    if attn is not None:
        for name in ("qkv", "proj"):
            lin = getattr(attn, name, None)
            if isinstance(lin, nn.Linear):
                out.append((attn, name, lin))
        return out
    attention = getattr(block, "attention", None)
    if attention is not None:
        inner = getattr(attention, "attention", None)
        if inner is not None:
            for name in ("query", "key", "value"):
                lin = getattr(inner, name, None)
                if isinstance(lin, nn.Linear):
                    out.append((inner, name, lin))
        output = getattr(attention, "output", None)
        if output is not None and isinstance(getattr(output, "dense", None), nn.Linear):
            out.append((output, "dense", output.dense))
    return out


def _inject_lora(model, rank: int, alpha: float) -> List[nn.Module]:
    injected: List[nn.Module] = []
    blocks = _blocks_of(model)
    if blocks is None:
        raise RuntimeError("[e6] LoRA injection needs a ViT with a block list "
                           "(timm .blocks or HF .encoder.layer); this encoder "
                           "exposes neither.")
    for blk in blocks:
        for parent, name, lin in _attn_linears_of(blk):
            wrapped = LoRALinear(lin, rank, alpha)
            setattr(parent, name, wrapped)
            injected.append(wrapped)
    if not injected:
        raise RuntimeError("[e6] LoRA injection found no attention Linear layers.")
    return injected


def set_trainable(model, head, level: str, lora_modules: Optional[List[nn.Module]]) -> List[nn.Parameter]:
    if level not in LEVEL_DEPTH:
        raise ValueError(f"[e6] unknown level '{level}'; expected one of {list(LEVEL_DEPTH)}.")
    for p in model.parameters():
        p.requires_grad = False
    for p in head.parameters():
        p.requires_grad = True

    if level == "linear":
        pass
    elif level == "last_block":
        blocks = _blocks_of(model)
        if blocks is None or len(blocks) == 0:
            raise RuntimeError("[e6] last_block needs a ViT with a block list "
                               "(timm .blocks or HF .encoder.layer).")
        for p in blocks[-1].parameters():
            p.requires_grad = True
        norm = _final_norm_of(model)
        if norm is not None:
            for p in norm.parameters():
                p.requires_grad = True
    elif level == "lora":
        if not lora_modules:
            raise RuntimeError("[e6] lora level requires injected LoRA modules.")
        for m in lora_modules:
            m.A.requires_grad = True
            m.B.requires_grad = True
    elif level == "full":
        for p in model.parameters():
            p.requires_grad = True

    return [p for p in list(model.parameters()) + list(head.parameters()) if p.requires_grad]


class _CXRFinetuneDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, cfg: dict, transform, row_ids: np.ndarray):
        self.man = manifest
        self.rows = np.asarray(row_ids)
        self.transform = transform
        self.site_root = {s: sc["image_root"] for s, sc in cfg["cxr"]["sites"].items()}
        self.res = int(cfg.get("target_resolution", 224))
        self.findings = [f for f in FINDINGS if f in manifest.columns]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rid = int(self.rows[i])
        row = self.man.iloc[rid]
        site = str(row.get("site") or row.get("dataset") or "mimic")
        path = resolve_cxr_image_path(site, self.site_root.get(site, ""),
                                      str(row.get("image_key", "")),
                                      row.get("image_subdir"), self.res)
        img = self.transform(Image.open(path).convert("RGB"))
        y = np.array([pd.to_numeric(row.get(f), errors="coerce") for f in self.findings],
                     dtype=np.float32)
        return img, torch.from_numpy(y), rid


def _split_row_ids(man: pd.DataFrame, max_train: Optional[int], seed: int):
    tr_m, va_m, te_m = split_masks(man)
    tr = np.where(tr_m)[0]; va = np.where(va_m)[0]; te = np.where(te_m)[0]
    if max_train and tr.size > max_train:
        tr = np.sort(np.random.RandomState(seed).permutation(tr)[:max_train])
    return tr, va, te


def _build_backbone(encoder: str, cfg: dict):
    from transformers import AutoModel
    spec = cfg["encoder_panel"]["image"].get(encoder, {})
    hf_id = spec.get("hf_id")
    if not hf_id or str(hf_id).endswith("_LOCAL"):
        raise RuntimeError(f"[e6] encoder '{encoder}' has no HF-loadable hf_id.")
    token = cfg.get("hf_token")
    try:
        model = _load_hf_offline_first(
            lambda mid, **kw: AutoModel.from_pretrained(mid, token=token,
                                                        trust_remote_code=True, **kw),
            hf_id, f"e6 backbone '{encoder}'")
    except Exception as e:
        raise RuntimeError(
            f"[e6] could not load '{encoder}' ({hf_id}) as an HF ViT: {e}. "
            f"finetuning.encoders must be HF-loadable block-structured ViTs.") from e
    if _blocks_of(model) is None:
        raise RuntimeError(
            f"[e6] '{encoder}' ({hf_id}) loaded but exposes no transformer block "
            f"list (.blocks or .encoder.layer); the unfreezing ladder needs one.")
    if cfg["finetuning"].get("gradient_checkpointing", True) and \
            hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    transform = eval_transform(int(cfg.get("target_resolution", 224)))
    return model, transform, get_embed_dim(model)


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def _amp(cfg, device):
    use = bool(cfg["finetuning"].get("bf16", True)) and device == "cuda"
    return torch.autocast("cuda", dtype=torch.bfloat16) if use else _nullctx()


def _e6_progress_path(cfg: dict, tag: str) -> str:
    return os.path.join(cfg["finetuning"]["results_e6_dir"], "ckpts", f"{tag}.inprogress.pt")


def _optimizer(trainable_backbone, head, cfg, steps_total):
    ft = cfg["finetuning"]
    groups = [{"params": head.parameters(), "lr": float(ft["lr_head"])}]
    if trainable_backbone:
        groups.append({"params": trainable_backbone, "lr": float(ft["lr_backbone"])})
    opt = torch.optim.AdamW(groups, weight_decay=float(ft["weight_decay"]))
    warmup = int(float(ft.get("warmup_fraction", 0.1)) * steps_total)

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, steps_total - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


@torch.no_grad()
def _val_auroc(model, head, ds_val, findings, man, cfg, device) -> float:
    rid, probs = _infer(model, head, ds_val, cfg, device,
                        max_batches=cfg["finetuning"].get("es_max_val_batches"))
    if rid.size == 0:
        return float("nan")
    idx = [int(r) for r in rid]
    aucs = []
    for j, finding in enumerate(findings):
        y = pd.to_numeric(man[finding], errors="coerce").values[idx]
        s = probs[:, j]
        ok = np.isfinite(y) & np.isfinite(s)
        if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
            continue
        aucs.append(auroc(y[ok], s[ok]))
    aucs = [a for a in aucs if np.isfinite(a)]
    return float(np.mean(aucs)) if aucs else float("nan")


def _train(model, head, ds_train, cfg, device, level, lora_modules, tag,
           ds_val=None, findings=None, man=None):
    ft = cfg["finetuning"]
    trainable = set_trainable(model, head, level, lora_modules)
    backbone_trainable = [p for p in model.parameters() if p.requires_grad]
    model = model.to(device); head = head.to(device)
    dl = DataLoader(ds_train, batch_size=int(ft["batch_size"]), shuffle=True,
                    drop_last=True, num_workers=int(ft.get("num_workers", 4)))
    epochs = int(ft["epochs"])
    opt, sched = _optimizer(backbone_trainable, head, cfg, epochs * max(1, len(dl)))
    bce = nn.BCEWithLogitsLoss(reduction="none")
    n_trainable = sum(p.numel() for p in trainable)
    es = bool(ft.get("early_stopping", True)) and ds_val is not None and findings
    min_epochs = int(ft.get("es_min_epochs", 1))
    patience = int(ft.get("es_patience", 3))
    min_delta = float(ft.get("es_min_delta", 0.0005))
    print(f"[e6] training level={level}: {n_trainable:,} trainable params, "
          f"{len(ds_train)} train images, {epochs} epochs"
          f"{f', early stopping on val AUROC (patience {patience})' if es else ''}.")

    ckpt_path = _e6_progress_path(cfg, tag)
    prog = _load_progress(ckpt_path)
    start_epoch = 0
    already_stopped = False
    best = {"auroc": -np.inf, "epoch": 0, "model": None, "head": None}
    if prog is not None:
        start_epoch = _try_resume(prog, tag, "e6", model=(model, "model"),
                                  head=(head, "head"), opt=(opt, "opt"),
                                  sched=(sched, "sched"))
        if start_epoch > 0 and isinstance(prog.get("best"), dict):
            b = prog["best"]
            best.update({"auroc": float(b.get("auroc", -np.inf)),
                         "epoch": int(b.get("epoch", 0)),
                         "model": b.get("model"), "head": b.get("head")})
            already_stopped = bool(prog.get("stopped", False))
            print(f"[e6] {tag}: resumed early-stopping state, best val AUROC "
                  f"{best['auroc']:.4f} at epoch {best['epoch']}"
                  f"{', training already stopped' if already_stopped else ''}.")
    if already_stopped:
        print(f"[e6] {tag}: training was already complete; skipping straight to "
              f"the selected epoch.")
        start_epoch = epochs

    for ep in range(start_epoch, epochs):
        model.train(); head.train()
        running = 0.0
        pbar = tqdm(dl, desc=f"[e6] {tag} ep{ep+1}/{epochs}", unit="batch")
        for it, (imgs, y, _) in enumerate(pbar):
            imgs, y = imgs.to(device), y.to(device)
            mask = torch.isfinite(y).float()
            y0 = torch.nan_to_num(y, nan=0.0)
            with _amp(cfg, device):
                feats = forward_features_cls(model, imgs)
                logits = head(feats)
                loss = (bce(logits, y0) * mask).sum() / mask.sum().clamp(min=1)
            loss.backward()
            opt.step(); sched.step(); opt.zero_grad()
            running += loss.item()
            pbar.set_postfix(loss=f"{running/(it+1):.4f}")
        train_loss = running / max(1, len(dl))

        stop = False
        if es:
            va = _val_auroc(model, head, ds_val, findings, man, cfg, device)
            improved = np.isfinite(va) and va > best["auroc"] + min_delta
            if improved:
                best = {"auroc": float(va), "epoch": ep + 1,
                        "model": {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()},
                        "head": {k: v.detach().cpu().clone()
                                 for k, v in head.state_dict().items()}}
            since = (ep + 1) - best["epoch"]
            print(f"[e6]   epoch {ep+1}/{epochs} loss={train_loss:.4f} "
                  f"val_auroc={va:.4f} best={best['auroc']:.4f}@ep{best['epoch']}"
                  f"{' *' if improved else f' (no gain for {since})'}")
            stop = (ep + 1) >= min_epochs and since >= patience
        else:
            print(f"[e6]   epoch {ep+1}/{epochs} loss={train_loss:.4f}")

        _save_progress(ckpt_path, ep + 1, model=model.state_dict(),
                       head=head.state_dict(), opt=opt.state_dict(),
                       sched=sched.state_dict(), best=best,
                       stopped=bool(stop or (ep + 1) >= epochs))
        R.heartbeat_claim(ft["results_e6_dir"], tag)
        if stop:
            print(f"[e6] {tag}: early stop at epoch {ep+1}; no val-AUROC gain "
                  f"for {patience} epochs.")
            break

    if es and best["model"] is not None:
        model.load_state_dict({k: v.to(device) for k, v in best["model"].items()})
        head.load_state_dict({k: v.to(device) for k, v in best["head"].items()})
        print(f"[e6] {tag}: restored epoch {best['epoch']} "
              f"(val AUROC {best['auroc']:.4f}) for inference.")
    return model, head


@torch.no_grad()
def _infer(model, head, ds, cfg, device, max_batches=None) -> Tuple[np.ndarray, np.ndarray]:
    ft = cfg["finetuning"]
    dl = DataLoader(ds, batch_size=int(ft["batch_size"]), shuffle=False,
                    num_workers=int(ft.get("num_workers", 4)))
    model.eval(); head.eval()
    all_p, all_rid = [], []
    for bi, (imgs, _, rid) in enumerate(tqdm(dl, desc="[e6] infer", unit="batch", leave=False)):
        if max_batches is not None and bi >= int(max_batches):
            break
        imgs = imgs.to(device)
        with _amp(cfg, device):
            feats = forward_features_cls(model, imgs)
            logits = head(feats)
        all_p.append(torch.sigmoid(logits.float()).cpu().numpy())
        all_rid.append(np.asarray(rid).copy())
    if not all_p:
        return np.zeros(0, int), np.zeros((0, len(ds.findings)), np.float32)
    return np.concatenate(all_rid), np.concatenate(all_p, axis=0)


def _patient_series(man: pd.DataFrame) -> np.ndarray:
    col = "subject_id" if "subject_id" in man.columns else "case_id"
    return man[col].astype(str).values


def _eval_level(perf_rows, encoder, level, man, findings, va_rid, va_probs,
                te_rid, te_probs, cfg):
    n_boot = int(cfg["stats"]["n_boot"]); seed = int(cfg["stats"]["boot_seed"])
    n_jobs = resolve_n_jobs(cfg["stats"].get("bootstrap_n_jobs", 1))
    target = float(cfg["stats"]["operating_sensitivity"])
    depth = LEVEL_DEPTH[level]
    va_map = {int(r): i for i, r in enumerate(va_rid)}
    te_map = {int(r): i for i, r in enumerate(te_rid)}
    pat = _patient_series(man)

    for j, finding in enumerate(tqdm(findings, desc=f"[e6] eval {encoder}/{level}", unit="finding")):
        yj = pd.to_numeric(man[finding], errors="coerce").values
        va_ids = [r for r in va_rid if np.isfinite(yj[int(r)])]
        te_ids = [r for r in te_rid if np.isfinite(yj[int(r)])]
        if len(va_ids) < 20 or len(te_ids) < 20:
            continue
        yva = yj[[int(r) for r in va_ids]]
        sva = va_probs[[va_map[int(r)] for r in va_ids], j]
        if len(np.unique(yva[np.isfinite(yva)])) < 2:
            continue
        thr = threshold_at_sensitivity(yva, sva, target)

        yte = yj[[int(r) for r in te_ids]]
        ste = te_probs[[te_map[int(r)] for r in te_ids], j]
        pte = pat[[int(r) for r in te_ids]]

        for attr in cfg["finetuning"]["attributes"]:
            if attr not in man.columns:
                continue
            gte = man[attr].astype(str).values[[int(r) for r in te_ids]]
            df = pd.DataFrame({"label": yte, "score": ste,
                               "group": [str(v) for v in gte],
                               "patient": [str(v) for v in pte]})
            df = df[df["group"].str.lower() != "nan"]
            if df.empty or len(df["group"].unique()) < 2:
                continue
            boot = cluster_bootstrap(df, "patient",
                                     lambda d: compute_fairness_point(
                                         d["label"].values, d["score"].values,
                                         d["group"].values, thr),
                                     n_boot, seed, n_jobs=n_jobs)
            ctx = {"experiment": "e6", "modality": "cxr", "dataset": "cxr_pool",
                   "eval_dataset": "cxr_pool", "encoder": encoder,
                   "encoder_objective": encoder, "attribute": attr,
                   "finding": finding, "mitigation": f"finetune:{level}",
                   "operating_point": f"sens{target:.2f}",
                   "data_composition": f"unfreeze_depth{depth}"}
            for k in SUMMARY:
                if k in boot:
                    direction = "higher" if k in ("auroc_overall", "es_auc") else "lower"
                    R.report_metric(perf_rows, ctx, k, boot[k], direction,
                                    int(df["patient"].nunique()))


def _collect_gap_by_depth(perf_rows) -> pd.DataFrame:
    df = pd.DataFrame(perf_rows)
    if df.empty:
        return df
    g = df[df["metric_name"] == "auroc_gap"].copy()
    g["depth"] = g["data_composition"].str.replace("unfreeze_depth", "", regex=False).astype(int)
    return g[["encoder", "attribute", "finding", "depth", "value_raw"]]


def run_finetuning_cell(encoder, level, cfg, cfg_path, device, man, transform, embed_dim):
    model, _, _ = _build_backbone(encoder, cfg)
    ft = cfg["finetuning"]
    seed = int(cfg["stats"]["boot_seed"])
    torch.manual_seed(seed); np.random.seed(seed)
    findings = [f for f in FINDINGS if f in man.columns]
    head = nn.Linear(embed_dim, len(findings))

    lora_modules = None
    if level == "lora":
        lora_modules = _inject_lora(model, int(ft["lora_rank"]), float(ft["lora_alpha"]))

    tr, va, te = _split_row_ids(man, ft.get("max_train"), seed)
    ds_tr = _CXRFinetuneDataset(man, cfg, transform, tr)
    ds_va = _CXRFinetuneDataset(man, cfg, transform, va)
    ds_te = _CXRFinetuneDataset(man, cfg, transform, te)

    model, head = _train(model, head, ds_tr, cfg, device, level, lora_modules,
                         tag=f"{encoder}__{level}", ds_val=ds_va,
                         findings=findings, man=man)
    va_rid, va_probs = _infer(model, head, ds_va, cfg, device)
    te_rid, te_probs = _infer(model, head, ds_te, cfg, device)
    return findings, va_rid, va_probs, te_rid, te_probs


def _dose_response_stats(cfg, perf: "pd.DataFrame") -> List[Dict]:
    seed = int(cfg["stats"]["boot_seed"]); n_perm = int(cfg["stats"]["n_perm"])
    gaps = _collect_gap_by_depth(perf.to_dict("records"))
    rows: List[Dict] = []
    if gaps.empty:
        return rows
    for (encoder, attr), sub in gaps.groupby(["encoder", "attribute"]):
        if sub["depth"].nunique() < 3 or len(sub) < 4:
            continue
        rho, p = spearman_with_perm(sub["depth"].values.astype(float),
                                    sub["value_raw"].values.astype(float),
                                    n_perm=n_perm, seed=seed)
        R.report_spearman(rows,
                          {"experiment": "e6", "modality": "cxr", "dataset": "cxr_pool",
                           "encoder": encoder, "encoder_objective": encoder,
                           "attribute": attr, "finding": "dose_response",
                           "mitigation": "finetune_ladder"},
                          rho, p, fdr_family=f"e6_dose_response::{attr}",
                          n_units=int(len(sub)))
    return rows


def main_e6_cell(encoder: str, level: str, global_config_path: str,
                 force: bool = False):
    cfg = read_config(global_config_path)["BiasOrigin"]
    ft = cfg["finetuning"]
    if not ft.get("enabled", True):
        print("[e6] finetuning disabled in config; skipping.")
        return
    out_dir = ft["results_e6_dir"]
    tag = f"{encoder}__{level}"
    if not force and R.shard_exists(out_dir, tag):
        print(f"[e6] shard for {tag} exists; skipping (force=True to redo).")
        return
    stale = float(cfg["stats"].get("claim_stale_after_s", 21600))
    if not force and not R.claim_unit(out_dir, tag, stale):
        print(f"[e6] {tag} is claimed by another running job; skipping to the next.")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    man = read_csv_defensively(cfg["cxr"]["pool_manifest_csv"])
    try:
        _, transform, embed_dim = _build_backbone(encoder, cfg)
    except RuntimeError as e:
        print(f"{e} Skipping.")
        R.release_claim(out_dir, tag)
        return
    findings, va_rid, va_probs, te_rid, te_probs = run_finetuning_cell(
        encoder, level, cfg, global_config_path, device, man, transform, embed_dim)
    perf_rows: List[Dict] = []
    _eval_level(perf_rows, encoder, level, man, findings,
                va_rid, va_probs, te_rid, te_probs, cfg)
    if not perf_rows:
        print(f"[e6] {tag}: produced no rows; not writing a shard.")
        R.release_claim(out_dir, tag)
        return
    R.write_shard(out_dir, tag, perf_rows, [])
    ckpt_path = _e6_progress_path(cfg, tag)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    R.release_claim(out_dir, tag)
    print(f"[e6] {encoder} level={level} done.")


def main_e6_merge(global_config_path: str) -> Tuple[str, str]:
    cfg = read_config(global_config_path)["BiasOrigin"]
    return R.merge_shards(
        cfg["finetuning"]["results_e6_dir"], "e6", float(cfg["stats"]["fdr_alpha"]),
        extra_stat_from_perf=lambda perf: _dose_response_stats(cfg, perf))


def main_e6(global_config_path: str) -> Tuple[str, str]:
    cfg = read_config(global_config_path)["BiasOrigin"]
    ft = cfg["finetuning"]
    if not ft.get("enabled", True):
        print("[e6] finetuning disabled in config; skipping.")
        return "", ""
    for encoder in ft["encoders"]:
        for level in ft["levels"]:
            main_e6_cell(encoder, level, global_config_path)
    return main_e6_merge(global_config_path)

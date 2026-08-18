"""
encoders/image_encoders.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

from functools import lru_cache
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config.serde import read_config

import warnings
warnings.filterwarnings("ignore")


def _get_spec(model_name: str, cfg: dict) -> dict:
    panel = cfg["BiasOrigin"]["encoder_panel"]["image"]
    if model_name not in panel:
        raise KeyError(
            f"[image_encoders] '{model_name}' not in encoder_panel.image. "
            f"Available: {sorted(panel.keys())}")
    return panel[model_name]


def list_encoder_names(cfg_path: str) -> List[str]:
    cfg = read_config(cfg_path)
    return sorted(cfg["BiasOrigin"]["encoder_panel"]["image"].keys())


@lru_cache(maxsize=4)
def _load_encoder(model_name: str, cfg_path: str, device: str):
    cfg  = read_config(cfg_path)
    spec = _get_spec(model_name, cfg)
    hf_id = spec.get("hf_id")
    etype = spec["type"]
    token = cfg["BiasOrigin"].get("hf_token")
    fp16  = cfg["BiasOrigin"]["embeddings"].get("fp16", True)
    dtype = torch.float16 if (fp16 and device == "cuda") else torch.float32

    if etype == "dino_cls":
        from transformers import AutoModel, AutoImageProcessor
        model = AutoModel.from_pretrained(hf_id, token=token, trust_remote_code=True)
        proc  = AutoImageProcessor.from_pretrained(hf_id, token=token, trust_remote_code=True)
        model = model.to(device=device, dtype=dtype).eval()
        return etype, (model, proc), spec

    if etype == "clip_pooled":
        if "BiomedCLIP" in str(hf_id) or spec.get("loader") == "open_clip":
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(f"hf-hub:{hf_id}")
            model = model.to(device=device, dtype=dtype).eval()
            return "open_clip_custom", (model, preprocess), spec
        from transformers import AutoModel, AutoProcessor
        model = AutoModel.from_pretrained(hf_id, token=token)
        proc  = AutoProcessor.from_pretrained(hf_id, token=token)
        model = model.to(device=device, dtype=dtype).eval()
        return etype, (model, proc), spec

    if etype == "open_clip_custom":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(f"hf-hub:{hf_id}")
        model = model.to(device=device, dtype=dtype).eval()
        return etype, (model, preprocess), spec

    if etype == "timm_cls":
        import timm
        from timm.data import resolve_data_config, create_transform
        model = timm.create_model(_timm_name(hf_id), pretrained=True, num_classes=0)
        cfg_t = resolve_data_config({}, model=model)
        transform = create_transform(**cfg_t)
        model = model.to(device=device, dtype=dtype).eval()
        return etype, (model, transform), spec

    if etype == "txrv":
        import torchxrayvision as xrv
        weights = spec.get("weights", "densenet121-res224-all")
        _orig_torch_load = torch.load
        def _patched_load(*a, **kw):
            kw.setdefault("weights_only", False)
            return _orig_torch_load(*a, **kw)
        torch.load = _patched_load
        try:
            model = xrv.models.DenseNet(weights=weights)
        finally:
            torch.load = _orig_torch_load
        model = model.to(device=device).eval()
        return etype, (model, None), spec

    if etype == "random_init":
        import timm
        from timm.data import resolve_data_config, create_transform
        arch = spec.get("arch", "vit_small_patch16_224")
        model = timm.create_model(arch, pretrained=False, num_classes=0)
        cfg_t = resolve_data_config({}, model=model)
        transform = create_transform(**cfg_t)
        model = model.to(device=device, dtype=dtype).eval()
        return etype, (model, transform), spec

    if etype == "flair":
        return _load_flair(hf_id, device, dtype, spec)

    raise ValueError(f"[image_encoders] unknown type '{etype}' for {model_name}")


def _timm_name(hf_id: str) -> str:
    s = str(hf_id)
    if s.startswith("hf-hub:") or "/" in s:
        return s if s.startswith("hf-hub:") else f"hf-hub:{s}"
    return s


def _load_flair(hf_id, device, dtype, spec):
    try:
        from flair import FLAIRModel
    except (ImportError, ModuleNotFoundError) as e:
        raise RuntimeError(
            "[flair] requires the FLAIR package (pip install "
            "git+https://github.com/jusiro/FLAIR.git). Import error: "
            f"{e}") from e
    repo = hf_id if (hf_id and str(hf_id) not in ("FLAIR_LOCAL", "")) else "jusiro2/FLAIR"
    try:
        model = FLAIRModel.from_pretrained(repo)
    except Exception as e:
        print(f"[flair] from_pretrained('{repo}') failed ({e}); falling back to "
              f"from_checkpoint=True (the package's own internal download).")
        model = FLAIRModel(from_checkpoint=True)
    try:
        model.to(device)
    except Exception:
        pass
    return "flair", (model, None), spec


def _emb_dino(payload, images, device, dtype):
    model, proc = payload
    inputs = proc(images=images, return_tensors="pt")
    inputs = {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
              for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    hs = out.last_hidden_state
    return hs[:, 0, :]


def _emb_clip_hf(payload, images, device, dtype):
    model, proc = payload
    inputs = proc(images=images, return_tensors="pt")
    inputs = {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
              for k, v in inputs.items()}

    needs_fallback = getattr(model, "_clip_hf_needs_full_forward", None)
    if not needs_fallback:
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
        if isinstance(emb, torch.Tensor):
            model._clip_hf_needs_full_forward = False
            return emb
        print(f"[clip_hf] get_image_features() returned {type(emb).__name__}, not "
              f"a tensor (known transformers>=5.0 regression); switching to the "
              f"full forward pass's image_embeds field for this model.")
        model._clip_hf_needs_full_forward = True

    if "input_ids" not in inputs:
        dummy = proc(text=[""] * len(images), images=images,
                     return_tensors="pt", padding=True)
        inputs = {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
                  for k, v in dummy.items()}
    with torch.no_grad():
        out = model(**inputs)
    if getattr(out, "image_embeds", None) is None:
        raise RuntimeError(
            "[clip_hf] get_image_features() did not return a tensor and the full "
            f"forward pass has no usable image_embeds either (got {type(out).__name__}); "
            "cannot extract an embedding for this model.")
    return out.image_embeds


def _emb_open_clip(payload, images, device, dtype):
    model, preprocess = payload
    px = torch.stack([preprocess(im) for im in images]).to(device=device, dtype=dtype)
    with torch.no_grad():
        emb = model.encode_image(px)
    return emb


def _emb_timm(payload, images, device, dtype):
    model, transform = payload
    px = torch.stack([transform(im) for im in images]).to(device=device, dtype=dtype)
    with torch.no_grad():
        feats = model.forward_features(px)
    if feats.ndim == 3:
        return feats[:, 0, :]
    return feats


def _emb_txrv(payload, images, device, dtype):
    import torchxrayvision as xrv
    model, _ = payload
    arrs = []
    for im in images:
        g = np.asarray(im.convert("L"), dtype=np.float32)
        g = xrv.datasets.normalize(g, 255)
        arrs.append(g[None, ...])
    x = torch.from_numpy(np.stack(arrs)).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        feats = model.features(x)
        emb = F.adaptive_avg_pool2d(F.relu(feats), 1).flatten(1)
    return emb


def _emb_flair(payload, images, device, dtype):
    model, _ = payload
    import numpy as _np
    batch = _np.stack([_np.asarray(im.convert("RGB")) for im in images])
    with torch.no_grad():
        if hasattr(model, "encode_image"):
            emb = model.encode_image(batch)
        elif hasattr(model, "vision_model"):
            px = torch.from_numpy(batch).permute(0, 3, 1, 2).float().to(device) / 255.0
            emb = model.vision_model(px)
        else:
            raise RuntimeError("[flair] could not find an image-encoding method "
                               "on FLAIRModel; check the installed FLAIR version.")
    if isinstance(emb, np.ndarray):
        emb = torch.from_numpy(emb)
    return emb.float().to(device)


_DISPATCH = {
    "dino_cls":         _emb_dino,
    "clip_pooled":      _emb_clip_hf,
    "open_clip_custom": _emb_open_clip,
    "timm_cls":         _emb_timm,
    "random_init":      _emb_timm,
    "txrv":             _emb_txrv,
    "flair":            _emb_flair,
}


_FORCED_DTYPE_ATTR = "_bias_origin_forced_dtype"


def _module_of(payload):
    m = payload[0] if isinstance(payload, tuple) else payload
    return m if isinstance(m, torch.nn.Module) else None


def _param_dtype(module):
    for p in module.parameters():
        return p.dtype
    return None


def _wider_dtypes(current, device):
    out = []
    if current != torch.bfloat16 and device == "cuda" and torch.cuda.is_bf16_supported():
        out.append(torch.bfloat16)
    if current != torch.float32:
        out.append(torch.float32)
    return out


def extract_image_embeddings(
    model_name: str,
    images: List[Image.Image],
    cfg_path: str,
    device: str = "cuda",
) -> np.ndarray:
    if not torch.cuda.is_available() and device == "cuda":
        device = "cpu"
    kind, payload, spec = _load_encoder(model_name, cfg_path, device)
    fp16 = read_config(cfg_path)["BiasOrigin"]["embeddings"].get("fp16", True)
    dtype = torch.float16 if (fp16 and device == "cuda") else torch.float32

    fn = _DISPATCH.get(kind)
    if fn is None:
        raise ValueError(f"[image_encoders] no extraction branch for kind '{kind}'")

    module = _module_of(payload)
    if module is not None:
        forced = getattr(module, _FORCED_DTYPE_ATTR, None)
        if forced is not None:
            dtype = forced

    def _run(dt):
        e = fn(payload, images, device, dt)
        e = F.normalize(e.float(), p=2, dim=-1)
        return e.cpu().numpy().astype(np.float32)

    out = _run(dtype)
    cur = _param_dtype(module) if module is not None else None
    if cur is not None and cur != torch.float32 and not np.isfinite(out).all():
        for wider in _wider_dtypes(cur, device):
            n_bad = int((~np.isfinite(out).all(axis=1)).sum())
            print(f"[image_encoders] '{model_name}': {n_bad}/{out.shape[0]} non-finite "
                  f"embeddings with weights in {cur}; promoting this encoder's weights "
                  f"to {wider} and retrying.")
            module.to(dtype=wider)
            setattr(module, _FORCED_DTYPE_ATTR, wider)
            cur = wider
            out = _run(wider)
            if np.isfinite(out).all():
                break
    if not np.isfinite(out).all():
        n_bad = int((~np.isfinite(out).all(axis=1)).sum())
        raise RuntimeError(
            f"[image_encoders] '{model_name}' produced non-finite embeddings for "
            f"{n_bad}/{out.shape[0]} images with weights in {cur}; refusing to cache a "
            f"corrupt embedding. Inspect the model load for this encoder.")
    return out

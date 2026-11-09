"""Model catalog: built-in Real-ESRGAN models plus any custom model you add.

Custom models are loaded through spandrel, which recognises the architecture of most
open-source image models (ESRGAN, Compact, SPAN, SwinIR, HAT, DAT, OmniSR, ...) from the
weights alone. Weight files are always read with weights-only loading (or safetensors),
so a model file can never execute code on this machine.
"""
import json
import os
import re
import shutil
import threading
import urllib.request
from dataclasses import dataclass

import torch

from .arch import RRDBNet, SRVGGNetCompact

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
CUSTOM_DIR = os.path.join(MODELS_DIR, "custom")
_INDEX = os.path.join(CUSTOM_DIR, "index.json")
_RELEASES = "https://github.com/xinntao/Real-ESRGAN/releases/download"
MODEL_EXTS = (".pth", ".safetensors", ".ckpt", ".bin")
_index_lock = threading.Lock()


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    description: str
    filename: str
    url: str
    scale: int             # native upscale factor of the network
    pad_multiple: int      # input width/height must be a multiple of this
    workers_per_gpu: int   # parallel frames per GPU (more = better GPU use, more VRAM)
    half: bool = True      # run in FP16
    path: str | None = None  # custom model file (None for built-ins)
    arch: str = ""
    params: int = 0

    @property
    def builtin(self) -> bool:
        return self.path is None

    @property
    def file_id(self) -> str:
        """Filesystem-safe id, used for cached ONNX/TensorRT files."""
        return re.sub(r"[^A-Za-z0-9._-]+", "-", self.key)

    def build(self):
        if self.key == "x2plus":
            return RRDBNet(scale=2, num_block=23)
        if self.key == "general-v3":
            return SRVGGNetCompact(num_feat=64, num_conv=32, upscale=4)
        if self.key == "anime-v3":
            return SRVGGNetCompact(num_feat=64, num_conv=16, upscale=4)
        raise ValueError(self.key)


MODELS = [
    ModelSpec(
        "x2plus", "Real-ESRGAN x2 - Best quality",
        "Large network that natively upscales 2x. Sharpest detail for live-action video; slowest.",
        "RealESRGAN_x2plus.pth", f"{_RELEASES}/v0.2.1/RealESRGAN_x2plus.pth",
        scale=2, pad_multiple=2, workers_per_gpu=1, arch="ESRGAN", params=16_700_000,
    ),
    ModelSpec(
        "general-v3", "Real-ESRGAN General v3 - Fast",
        "Compact network (4x, resized to 4K on the GPU). Several times faster; slightly softer.",
        "realesr-general-x4v3.pth", f"{_RELEASES}/v0.2.5.0/realesr-general-x4v3.pth",
        scale=4, pad_multiple=1, workers_per_gpu=2, arch="RealESRGAN Compact", params=1_210_000,
    ),
    ModelSpec(
        "anime-v3", "Real-ESRGAN Anime Video v3 - Animation",
        "Compact network tuned for cartoons and anime. Very fast.",
        "realesr-animevideov3.pth", f"{_RELEASES}/v0.2.5.0/realesr-animevideov3.pth",
        scale=4, pad_multiple=1, workers_per_gpu=2, arch="RealESRGAN Compact", params=620_000,
    ),
]
MODELS_BY_KEY = {m.key: m for m in MODELS}


# --------------------------------------------------------------------------- loading
def read_state_dict(path: str) -> dict:
    """Read weights safely: safetensors, or torch.load with weights_only=True (no code execution)."""
    if path.lower().endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path, device="cpu")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001
        raise ValueError(
            "This file can't be loaded safely as plain weights (it may be a full pickled model or "
            f"TorchScript). Look for a .safetensors or weights-only .pth version.\n({e})") from e


def ensure_downloaded(spec: ModelSpec, log=print) -> str:
    """Return the local weight file, downloading built-in models on first use."""
    if spec.path:
        if not os.path.exists(spec.path):
            raise FileNotFoundError(f"Model file is missing: {spec.path}")
        return spec.path
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, spec.filename)
    if os.path.exists(path):
        return path
    log(f"Downloading model {spec.filename} ...")
    tmp = path + ".part"
    with urllib.request.urlopen(spec.url, timeout=60) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done, last_pct = 0, -10
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                if pct >= last_pct + 10:
                    log(f"  {pct}% of {total / 1e6:.0f} MB")
                    last_pct = pct
    os.replace(tmp, path)
    log("Download complete.")
    return path


def _describe(state: dict):
    from spandrel import ImageModelDescriptor, ModelLoader
    desc = ModelLoader().load_from_state_dict(state)
    if not isinstance(desc, ImageModelDescriptor):
        raise ValueError("This is not an image-to-image model.")
    return desc


def load_model(spec: ModelSpec, device: torch.device, half: bool, log=print) -> torch.nn.Module:
    path = ensure_downloaded(spec, log)
    state = read_state_dict(path)
    if spec.path:
        net = _describe(state).model
    else:
        for k in ("params_ema", "params"):
            if isinstance(state, dict) and k in state:
                state = state[k]
                break
        net = spec.build()
        net.load_state_dict(state, strict=True)
    net.eval().requires_grad_(False).to(device)
    if half and spec.half:
        net = net.half()
    # channels_last lets cuDNN use tensor cores efficiently: ~40% faster on RTX 30/40 cards.
    return net.to(memory_format=torch.channels_last)


# --------------------------------------------------------------------------- custom models
def inspect_model_file(path: str) -> dict:
    """Identify a model file. Raises ValueError with a readable reason if it can't be used."""
    state = read_state_dict(path)
    try:
        desc = _describe(state)
    except ValueError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Unknown or unsupported model architecture ({type(e).__name__}: {e})") from e
    if desc.purpose not in ("SR", "Restoration"):
        raise ValueError(f"{desc.architecture.name} is a {desc.purpose} model; only upscaling and "
                         "restoration models can be used on video.")
    if desc.input_channels != 3 or desc.output_channels != 3:
        raise ValueError(f"The model uses {desc.input_channels}->{desc.output_channels} channels; "
                         "only RGB (3-channel) models are supported.")
    req = desc.size_requirements
    if getattr(req, "square", False):
        raise ValueError("This model only accepts square images, which doesn't work for video frames.")
    params = sum(p.numel() for p in desc.model.parameters())
    return {
        "arch": desc.architecture.name,
        "scale": int(desc.scale),
        "half": bool(desc.supports_half),
        "multiple_of": int(getattr(req, "multiple_of", 1) or 1),
        "purpose": desc.purpose,
        "params": int(params),
    }


def _spec_from_meta(path: str, meta: dict) -> ModelSpec:
    name = os.path.splitext(os.path.basename(path))[0]
    scale = meta["scale"]
    params = meta["params"]
    note = "" if scale == 2 else (f" Output is resized from {scale}× to 2× on the GPU." if scale > 1
                                  else " Restores at 1×, then resizes to 2× on the GPU.")
    heavy = params > 8_000_000 or meta["arch"] in ("HAT", "DAT", "SwinIR", "Swin2SR", "GRL", "RGT", "DRCT", "ATD", "IPT")
    desc = (f"{meta['arch']}  ·  {scale}×  ·  {params / 1e6:.2f}M params  ·  {'FP16' if meta['half'] else 'FP32'}.{note}"
            + (" Large/transformer model: expect it to be slow at 1080p." if heavy else ""))
    return ModelSpec(
        key=f"custom:{os.path.basename(path)}", label=name, description=desc,
        filename=os.path.basename(path), url="", scale=scale, pad_multiple=max(meta["multiple_of"], 1),
        workers_per_gpu=1 if heavy or params >= 3_000_000 else 2, half=meta["half"], path=path,
        arch=meta["arch"], params=params,
    )


def _load_index() -> dict:
    try:
        with open(_INDEX, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_index(idx: dict):
    os.makedirs(CUSTOM_DIR, exist_ok=True)
    tmp = _INDEX + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=1)
    os.replace(tmp, _INDEX)


def scan_custom_models(log=print) -> list[ModelSpec]:
    """All usable models in models/custom (metadata cached in index.json)."""
    os.makedirs(CUSTOM_DIR, exist_ok=True)
    with _index_lock:
        idx = _load_index()
        specs, changed = [], False
        names = sorted(f for f in os.listdir(CUSTOM_DIR) if f.lower().endswith(MODEL_EXTS))
        for fname in names:
            path = os.path.join(CUSTOM_DIR, fname)
            st = os.stat(path)
            stamp = f"{st.st_size}:{int(st.st_mtime)}"
            entry = idx.get(fname)
            if not entry or entry.get("stamp") != stamp:
                try:
                    entry = {"stamp": stamp, "meta": inspect_model_file(path)}
                except Exception as e:  # noqa: BLE001
                    entry = {"stamp": stamp, "error": str(e)}
                idx[fname] = entry
                changed = True
            if "meta" in entry:
                specs.append(_spec_from_meta(path, entry["meta"]))
            else:
                log(f"Skipping custom model {fname}: {entry['error'].splitlines()[0]}")
        for gone in set(idx) - set(names):
            del idx[gone]
            changed = True
        if changed:
            _save_index(idx)
    return specs


def add_custom_model(src: str) -> ModelSpec:
    """Validate a model file and copy it into models/custom. Raises ValueError if unusable."""
    meta = inspect_model_file(src)
    os.makedirs(CUSTOM_DIR, exist_ok=True)
    dst = os.path.join(CUSTOM_DIR, os.path.basename(src))
    if os.path.normcase(os.path.abspath(src)) != os.path.normcase(os.path.abspath(dst)):
        if os.path.exists(dst):
            raise ValueError(f"A model named {os.path.basename(src)} is already installed.")
        shutil.copy2(src, dst)
    st = os.stat(dst)
    with _index_lock:
        idx = _load_index()
        idx[os.path.basename(dst)] = {"stamp": f"{st.st_size}:{int(st.st_mtime)}", "meta": meta}
        _save_index(idx)
    return _spec_from_meta(dst, meta)


def remove_custom_model(spec: ModelSpec):
    if not spec.path:
        raise ValueError("Built-in models can't be removed.")
    if os.path.exists(spec.path):
        os.remove(spec.path)
    with _index_lock:
        idx = _load_index()
        idx.pop(os.path.basename(spec.path), None)
        _save_index(idx)


def all_models(log=print) -> list[ModelSpec]:
    return MODELS + scan_custom_models(log)

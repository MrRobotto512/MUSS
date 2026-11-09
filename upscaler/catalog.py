"""Model library: browse and install open-source models from OpenModelDB (https://openmodeldb.info).

The catalogue (~700 models, cached for a day) lists each model's architecture, scale, licence,
tags and download links with SHA-256 checksums. Models on direct-download hosts install with one
click: the file is downloaded, its checksum verified, then it is identified and validated by
spandrel exactly like a model you add yourself. Models hosted on Google Drive / MEGA / etc. need
a browser, so the library opens their download page instead.
"""
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

from . import __version__, models

CATALOG_URL = "https://openmodeldb.info/api/v1/models.json"
CACHE_PATH = os.path.join(models.MODELS_DIR, "openmodeldb_catalog.json")
CACHE_MAX_AGE = 24 * 3600

# OpenModelDB architecture ids that MUSS (spandrel) can load
SUPPORTED_ARCHS = {
    "esrgan", "esrgan+", "compact", "span", "dat", "swinir", "omnisr", "realplksr", "realplksr-dysample",
    "hat", "rgt", "drct", "atd", "rcan", "real-cugan", "cugan", "mosr", "ditn", "grl", "dctlsa", "craft",
    "swift-srgan",
}
ARCH_LABELS = {"esrgan": "ESRGAN", "esrgan+": "ESRGAN+", "compact": "Compact", "span": "SPAN", "dat": "DAT",
               "swinir": "SwinIR", "omnisr": "OmniSR", "realplksr": "RealPLKSR", "realplksr-dysample": "RealPLKSR",
               "hat": "HAT", "rgt": "RGT", "drct": "DRCT", "atd": "ATD", "rcan": "RCAN", "real-cugan": "Real-CUGAN",
               "cugan": "CUGAN", "mosr": "MoSR", "ditn": "DITN", "grl": "GRL", "dctlsa": "DCTLSA",
               "craft": "CRAFT", "swift-srgan": "Swift-SRGAN"}
# fast enough for video at 1080p (rough guide shown in the library)
FAST_ARCHS = {"compact", "span", "realplksr", "realplksr-dysample", "mosr", "omnisr", "ditn", "swift-srgan", "craft"}
HEAVY_ARCHS = {"dat", "swinir", "hat", "rgt", "drct", "atd", "grl"}
DIRECT_HOSTS = {"objectstorage.us-phoenix-1.oraclecloud.com", "f002.backblazeb2.com", "huggingface.co",
                "raw.githubusercontent.com", "objects.githubusercontent.com"}


@dataclass
class CatalogModel:
    id: str
    name: str
    author: str
    license: str
    tags: list
    description: str
    date: str
    arch: str
    scale: int
    supported: bool
    file_type: str = ""          # pth / safetensors
    size: int = 0
    sha256: str = ""
    direct_url: str = ""         # one-click download, or "" if the file needs a browser
    other_urls: list = field(default_factory=list)

    @property
    def page_url(self):
        return f"https://openmodeldb.info/models/{self.id}"

    @property
    def filename(self):
        return f"{self.id}.{self.file_type}"

    @property
    def arch_label(self):
        return ARCH_LABELS.get(self.arch, self.arch)

    @property
    def speed(self):
        return "fast" if self.arch in FAST_ARCHS else "slow" if self.arch in HEAVY_ARCHS else "medium"


def purge_engines(model_key: str):
    """Delete cached ONNX/TensorRT files for a model (they are stale once the model file changes)."""
    import re
    from .trt_backend import ENGINES_DIR
    fid = re.sub(r"[^A-Za-z0-9._-]+", "-", model_key)
    try:
        for f in os.listdir(ENGINES_DIR):
            if f.startswith(fid + "_"):
                os.remove(os.path.join(ENGINES_DIR, f))
    except OSError:
        pass


def _direct(url: str) -> str:
    """A URL that downloads the file itself, or '' if it is a page that needs a browser."""
    p = urllib.parse.urlparse(url)
    if p.scheme != "https":
        return ""
    if p.netloc == "github.com":
        if "/releases/download/" in p.path or "/raw/" in p.path:
            return url
        if "/blob/" in p.path:
            return url.replace("/blob/", "/raw/", 1)
        return ""
    if p.netloc in DIRECT_HOSTS:
        return url
    return ""


def _parse(mid: str, d: dict) -> CatalogModel:
    arch = (d.get("architecture") or "").lower()
    res = [r for r in d.get("resources", []) if r.get("type") in ("safetensors", "pth")]
    res.sort(key=lambda r: (0 if r.get("type") == "safetensors" else 1))      # prefer safetensors
    best, direct, others = (res[0] if res else {}), "", []
    for r in res:
        for u in r.get("urls", []):
            du = _direct(u)
            if du and not direct:
                best, direct = r, du
            elif not du:
                others.append(u)
    supported = (arch in SUPPORTED_ARCHS and bool(res) and d.get("inputChannels", 3) == 3
                 and d.get("outputChannels", 3) == 3)
    return CatalogModel(
        id=mid, name=d.get("name") or mid, author=", ".join(d["author"]) if isinstance(d.get("author"), list)
        else (d.get("author") or ""), license=d.get("license") or "unknown", tags=list(d.get("tags") or []),
        description=(d.get("description") or "").strip(), date=d.get("date") or "", arch=arch,
        scale=int(d.get("scale") or 0), supported=supported, file_type=best.get("type", ""),
        size=int(best.get("size") or 0), sha256=(best.get("sha256") or "").lower(), direct_url=direct,
        other_urls=others,
    )


def load_catalog(force=False, timeout=30) -> list[CatalogModel]:
    """All catalogue models (cached on disk for a day; falls back to the cache when offline)."""
    data = None
    fresh = os.path.exists(CACHE_PATH) and time.time() - os.path.getmtime(CACHE_PATH) < CACHE_MAX_AGE
    if force or not fresh:
        try:
            req = urllib.request.Request(CATALOG_URL, headers={"User-Agent": f"MUSS/{__version__}"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
            data = json.loads(raw)
            os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
            with open(CACHE_PATH + ".tmp", "wb") as f:
                f.write(raw)
            os.replace(CACHE_PATH + ".tmp", CACHE_PATH)
        except Exception:  # noqa: BLE001
            if not os.path.exists(CACHE_PATH):
                raise
    if data is None:
        with open(CACHE_PATH, encoding="utf-8") as f:
            data = json.load(f)
    out = [_parse(k, v) for k, v in data.items() if isinstance(v, dict)]
    out.sort(key=lambda m: m.date, reverse=True)
    return out


def catalog_age() -> float | None:
    """Seconds since the cached catalogue was downloaded, or None if there is none."""
    return time.time() - os.path.getmtime(CACHE_PATH) if os.path.exists(CACHE_PATH) else None


# which catalogue version (checksum) of each model is installed: {id: sha256}
_INSTALLED = os.path.join(models.CUSTOM_DIR, "openmodeldb_installed.json")


def _installed_map() -> dict:
    try:
        with open(_INSTALLED, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_installed_map(d: dict):
    os.makedirs(models.CUSTOM_DIR, exist_ok=True)
    with open(_INSTALLED + ".tmp", "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)
    os.replace(_INSTALLED + ".tmp", _INSTALLED)


def installed_ids() -> set:
    try:
        return {os.path.splitext(f)[0] for f in os.listdir(models.CUSTOM_DIR)
                if f.lower().endswith(models.MODEL_EXTS)}
    except OSError:
        return set()


def status_of(cm: CatalogModel, installed: set | None = None, inst_map: dict | None = None) -> str:
    """'installed', 'update' (the catalogue has a newer file), or ''."""
    installed = installed_ids() if installed is None else installed
    if cm.id not in installed:
        return ""
    inst_map = _installed_map() if inst_map is None else inst_map
    have = inst_map.get(cm.id)
    return "update" if (have and cm.sha256 and have != cm.sha256) else "installed"


def install(cm: CatalogModel, progress=None, cancel=None, replace=False) -> models.ModelSpec:
    """Download, verify the checksum, validate and register a catalogue model. Returns its ModelSpec.
    replace=True updates an installed model (its cached TensorRT engines are deleted)."""
    if not cm.direct_url:
        raise ValueError("This model is only available from a download page; open it in your browser.")
    os.makedirs(models.CUSTOM_DIR, exist_ok=True)
    existing = [f for f in os.listdir(models.CUSTOM_DIR)
                if os.path.splitext(f)[0] == cm.id and f.lower().endswith(models.MODEL_EXTS)]
    if existing and not replace:
        raise ValueError(f"{cm.name} is already installed.")
    dest = os.path.join(models.CUSTOM_DIR, cm.filename)
    dl_dir = os.path.join(models.CUSTOM_DIR, ".downloads")      # not scanned as installed models
    os.makedirs(dl_dir, exist_ok=True)
    tmp = os.path.join(dl_dir, cm.filename)                     # keeps the real extension for validation
    h = hashlib.sha256()
    req = urllib.request.Request(cm.direct_url, headers={"User-Agent": f"MUSS/{__version__}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or cm.size or 0)
            done = 0
            while chunk := r.read(1 << 18):
                if cancel and cancel():
                    raise InterruptedError("cancelled")
                f.write(chunk)
                h.update(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
        if cm.sha256 and h.hexdigest() != cm.sha256:
            raise ValueError("The downloaded file doesn't match OpenModelDB's checksum (corrupt or changed); "
                             "it was not installed.")
        # validate before touching an installed version
        meta_check = models.inspect_model_file(tmp)       # spandrel identifies + validates it
        del meta_check
        for f in existing:                                # updating: drop the old file and its engines
            old = os.path.join(models.CUSTOM_DIR, f)
            purge_engines(f"custom:{f}")
            os.remove(old)
        os.replace(tmp, dest)
        spec = models.add_custom_model(dest)
        inst = _installed_map()
        inst[cm.id] = h.hexdigest()
        _save_installed_map(inst)
        return spec
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

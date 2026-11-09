"""Video pipeline: ffmpeg decode -> AI upscale on several GPUs in parallel -> ffmpeg encode.

Frames go into a shared queue. Every GPU worker pulls the next frame as soon as it is
free, so a faster card (e.g. RTX 4070) automatically processes more frames than a
slower one (e.g. RTX 3060). A writer thread puts the frames back in order for the encoder.
"""
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from fractions import Fraction

import torch
import torch.nn.functional as F

from . import gpu_io, trt_backend
from .models import ModelSpec, load_model

torch.backends.cudnn.benchmark = True  # frame size is fixed, so pick the fastest conv kernels once

TOPAZ_DIR = r"C:\Program Files\Topaz Video AI"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

ENCODERS = {
    "HEVC / H.265 (NVENC, 10-bit)": ("hevc_nvenc", ["-profile:v", "main10"], "p010le"),
    "AV1 (NVENC, 10-bit, RTX 40 only)": ("av1_nvenc", [], "p010le"),
    "H.264 (NVENC, 8-bit, most compatible)": ("h264_nvenc", ["-profile:v", "high"], "yuv420p"),
}
MP4_SAFE_AUDIO = {"aac", "mp3", "ac3", "eac3", "opus", "alac", "flac"}


def find_tool(name: str) -> str:
    """Prefer ffmpeg bundled with Topaz Video AI, else one on PATH."""
    topaz = os.path.join(TOPAZ_DIR, name + ".exe")
    if os.path.exists(topaz):
        return topaz
    found = shutil.which(name)
    if not found:
        raise FileNotFoundError(f"{name} not found (install ffmpeg or Topaz Video AI)")
    return found


def list_gpus():
    if not torch.cuda.is_available():
        return []
    return [(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: Fraction
    frames: int
    duration: float
    color_space: str
    audio_codec: str | None


def probe(path: str) -> VideoInfo:
    out = subprocess.run(
        [find_tool("ffprobe"), "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
        capture_output=True, text=True, creationflags=NO_WINDOW, check=True,
    ).stdout
    data = json.loads(out)
    v = next(s for s in data["streams"] if s["codec_type"] == "video")
    a = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    fps = Fraction(v.get("avg_frame_rate") or "0/1")
    if fps <= 0:
        fps = Fraction(v.get("r_frame_rate") or "30/1")
    duration = float(v.get("duration") or data["format"].get("duration") or 0)
    frames = int(v.get("nb_frames") or 0) or round(duration * float(fps))
    return VideoInfo(int(v["width"]), int(v["height"]), fps, frames, duration,
                     v.get("color_space", "unknown"), a["codec_name"] if a else None)


def _in_matrix(info: VideoInfo) -> str:
    return "bt709" if info.color_space in ("unknown", "bt709") and info.height >= 720 else "auto"


def grab_frame(path: str, info: VideoInfo, t: float, size: tuple[int, int] | None = None) -> bytes:
    """Return one RGB24 frame at time t seconds (full size, or resized to size=(w, h))."""
    w, h = size or (info.width, info.height)
    scale = f"scale={w}:{h}:" if size else "scale="
    cmd = [find_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin",
           "-ss", f"{max(t, 0):.3f}", "-i", path, "-map", "0:v:0", "-frames:v", "1",
           "-vf", f"{scale}in_color_matrix={_in_matrix(info)}:flags=bicubic,format=rgb24",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    data = subprocess.run(cmd, capture_output=True, creationflags=NO_WINDOW).stdout
    need = w * h * 3
    if len(data) < need:  # seeking past the last frame: step back slightly
        if t > 0.5:
            return grab_frame(path, info, t - 0.5, size)
        raise RuntimeError("Could not read a frame from this video.")
    return data[:need]


def free_ram_bytes() -> int:
    """Available physical memory (Windows), used to size in-memory preview clips."""
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(m)
    try:
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return int(m.ullAvailPhys)
    except Exception:  # noqa: BLE001
        return 8 << 30


def upscale_frame(net, spec: ModelSpec, buf, w, h, dev, tile=0) -> bytes:
    """Upscale one frame to exactly 2x. Returns RGB24 bytes (2w x 2h)."""
    y = upscale_tensor(net, spec, buf, w, h, dev, tile)
    y = y.mul(255.0).round_().to(torch.uint8)
    return y[0].permute(1, 2, 0).contiguous().cpu().numpy().tobytes()


def upscale_tensor(net, spec: ModelSpec, src, w, h, dev, tile=0) -> torch.Tensor:
    """Upscale one frame (RGB24 bytes or a GPU tensor from NVDEC) to exactly 2x.
    Returns a (1,3,2h,2w) float tensor in [0,1] on dev."""
    if isinstance(src, (bytes, memoryview)):
        src = bytearray(src)
    dtype = torch.float16 if spec.half else torch.float32
    x = gpu_io.to_input(src, dev, dtype, w, h)
    m = spec.pad_multiple
    ph, pw = (-h) % m, (-w) % m
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    x = x.contiguous(memory_format=torch.channels_last)
    s = spec.scale
    y = _tiled(net, x, s, tile) if tile else net(x)
    y = y[:, :, : h * s, : w * s]
    if s != 2:  # e.g. 4x network: high-quality resize to exactly 2x (down for 3x/4x/8x, up for 1x)
        y = F.interpolate(y.float(), size=(h * 2, w * 2), mode="bicubic", antialias=s > 2)
    return y.float().clamp_(0, 1)


def _tiled(net, x, s, tile, pad=16):
    _, c, h, w = x.shape
    out = x.new_zeros((1, c, h * s, w * s))
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            py0, px0 = max(y0 - pad, 0), max(x0 - pad, 0)
            py1, px1 = min(y1 + pad, h), min(x1 + pad, w)
            t = net(x[:, :, py0:py1, px0:px1])
            oy, ox = (y0 - py0) * s, (x0 - px0) * s
            out[:, :, y0 * s:y1 * s, x0 * s:x1 * s] = t[:, :, oy:oy + (y1 - y0) * s, ox:ox + (x1 - x0) * s]
    return out


# --------------------------------------------------------------------------- model cache
# Loaded models stay on the GPUs between previews and exports, so you only wait for
# loading once per model / resolution / backend. Switching any of those frees the old ones.
_CACHE_LOCK = threading.RLock()
_CACHE = {"key": None, "nets": {}}


def trt_engines_ready(spec: ModelSpec, gpu_ids, w: int, h: int) -> bool:
    if not trt_backend.available():
        return False
    ph, pw = trt_backend.padded_size(spec, h, w)
    if trt_backend.failed_reason(spec, ph, pw):
        return False
    return all(os.path.exists(trt_backend.engine_path(spec, g, ph, pw)) for g in gpu_ids)


def trt_unsupported(spec: ModelSpec, w: int, h: int) -> str | None:
    """Reason TensorRT previously failed for this model/resolution, or None."""
    if not trt_backend.available():
        return None
    ph, pw = trt_backend.padded_size(spec, h, w)
    return trt_backend.failed_reason(spec, ph, pw)


def get_nets(spec: ModelSpec, gpu_ids, use_tensorrt: bool, w: int, h: int, log=print,
             build_trt=True, cancel_event=None):
    """Return {gpu_id: model} for the given GPUs, loading (and building TensorRT engines) as needed."""
    if use_tensorrt and not trt_backend.available():
        log(trt_backend.status() + " - falling back to PyTorch.")
        use_tensorrt = False
    if use_tensorrt and trt_unsupported(spec, w, h):
        use_tensorrt = False      # TensorRT already failed for this model: don't retry every time
    if use_tensorrt and not build_trt and not trt_engines_ready(spec, gpu_ids, w, h):
        use_tensorrt = False      # quick previews never trigger a multi-minute engine build
    with _CACHE_LOCK:
        for attempt_trt in ([True, False] if use_tensorrt else [False]):
            key = (spec.key, attempt_trt, w, h)
            if _CACHE["key"] != key:
                release_models()
                _CACHE["key"] = key
            nets = _CACHE["nets"]
            missing = [g for g in gpu_ids if g not in nets]
            if not missing:
                return {g: nets[g] for g in gpu_ids}
            try:
                loaded = (_load_trt if attempt_trt else _load_torch)(spec, missing, w, h, log, cancel_event)
            except Exception as e:  # noqa: BLE001
                if not attempt_trt:
                    raise
                # Custom architectures don't always convert to ONNX/TensorRT. Remember it and use PyTorch.
                ph, pw = trt_backend.padded_size(spec, h, w)
                reason = str(e).strip().splitlines()[-1][:300] if str(e).strip() else type(e).__name__
                trt_backend.mark_failed(spec, ph, pw, reason)
                log(f"TensorRT can't be used with {spec.label}: {reason}. Using PyTorch instead.")
                release_models()
                continue
            if loaded is None:
                return None
            nets.update(loaded)
            return {g: nets[g] for g in gpu_ids}
    return None


def release_models():
    with _CACHE_LOCK:
        _CACHE["key"], _CACHE["nets"] = None, {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_torch(spec, gpu_ids, w, h, log, cancel_event):
    nets = {}
    for gid in gpu_ids:
        log(f"Loading model on GPU {gid}: {torch.cuda.get_device_name(gid)}")
        nets[gid] = load_model(spec, torch.device("cuda", gid), half=True, log=log)
        if cancel_event is not None and cancel_event.is_set():
            return None
    return nets


def _load_trt(spec, gpu_ids, w, h, log, cancel_event):
    ph, pw = trt_backend.padded_size(spec, h, w)
    paths, errors, nets = {}, [], {}

    def run_all(fn):
        threads = [threading.Thread(target=fn, args=(g,), daemon=True) for g in gpu_ids]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def build(gid):
        try:
            paths[gid] = trt_backend.ensure_engine(spec, gid, ph, pw, log=log)
        except Exception as e:  # noqa: BLE001
            errors.append(f"GPU {gid}: {e}")

    # Export to ONNX happens once (inside the first build); then each GPU builds its own engine in parallel.
    build(gpu_ids[0])
    if not errors and len(gpu_ids) > 1:
        rest = gpu_ids[1:]
        threads = [threading.Thread(target=build, args=(g,), daemon=True) for g in rest]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    if cancel_event is not None and cancel_event.is_set():
        return None
    if errors:
        raise RuntimeError("TensorRT engine build failed:\n" + "\n".join(errors))

    def load(gid):
        try:
            log(f"Loading TensorRT engine on GPU {gid}: {torch.cuda.get_device_name(gid)}")
            nets[gid] = trt_backend.TRTModel(paths[gid], gid)
        except Exception as e:  # noqa: BLE001
            errors.append(f"GPU {gid}: {e}")

    run_all(load)
    if errors:
        raise RuntimeError("Could not load TensorRT engine:\n" + "\n".join(errors))
    return nets


class FramePreviewer:
    """Renders a single AI frame using the shared model cache."""

    def __init__(self, log=print):
        self.log = log
        self._lock = threading.Lock()
        self._streams = {}

    def render(self, spec: ModelSpec, gpu_id: int, buf: bytes, w: int, h: int, use_tensorrt=False) -> bytes:
        """Upscale one frame. Uses TensorRT only if its engine was already built (never builds here)."""
        with self._lock:
            net = get_nets(spec, [gpu_id], use_tensorrt, w, h, log=self.log, build_trt=False)[gpu_id]
            dev = torch.device("cuda", gpu_id)
            stream = self._streams.setdefault(gpu_id, torch.cuda.Stream(dev))
            with torch.cuda.device(dev), torch.cuda.stream(stream), torch.inference_mode():
                try:
                    return upscale_frame(net, spec, buf, w, h, dev)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if getattr(net, "is_tensorrt", False):
                        raise
                    return upscale_frame(net, spec, buf, w, h, dev, tile=512)

    def release(self):
        release_models()


@dataclass
class Progress:
    done: int
    total: int
    fps: float
    eta_s: float
    per_gpu: dict = field(default_factory=dict)


class UpscaleJob:
    def __init__(self, input_path, output_path, spec: ModelSpec, gpu_ids, encoder_label="",
                 cq=18, max_seconds=None, tile=0, use_tensorrt=False, on_log=print, on_progress=None,
                 start=0.0, frame_sink=None, max_frames=None):
        """start/max_seconds select a time range. With frame_sink(idx, src_rgb, out_rgb) no file is
        written: frames are handed back in order (used for in-app preview clips)."""
        self.input_path, self.output_path, self.spec = input_path, output_path, spec
        self.use_tensorrt = use_tensorrt
        self.start = max(float(start or 0), 0.0)
        self.frame_sink = frame_sink
        self.max_frames = max_frames
        self.gpu_ids, self.cq, self.max_seconds, self.tile = list(gpu_ids), cq, max_seconds, tile
        self.encoder_label = encoder_label
        self.log = on_log
        self.on_progress = on_progress or (lambda p: None)
        self.cancel_event = threading.Event()
        self._error = None
        self._procs = []

    # ------------------------------------------------------------------ public
    def cancel(self):
        self.cancel_event.set()
        for p in self._procs:
            if p.poll() is None:
                p.kill()

    def run(self):
        """Blocking. Returns True on success, False if cancelled; raises on error."""
        if not self.gpu_ids:
            raise RuntimeError("Select at least one GPU.")
        info = probe(self.input_path)
        self.info = info
        w, h = info.width, info.height
        self.out_w, self.out_h = w * 2, h * 2
        start_frame = round(self.start * float(info.fps))
        total = max(info.frames - start_frame, 0)
        if self.max_seconds:
            total = min(total, round(self.max_seconds * float(info.fps)))
        if self.max_frames:
            total = min(total, self.max_frames)
        rng = f" from {self.start:.2f}s" if self.start else ""
        self.log(f"Input {w}x{h} @ {float(info.fps):.3f} fps, ~{total} frames{rng} -> output {self.out_w}x{self.out_h}")
        self.log(f"Model: {self.spec.label}")

        # Load the model on every selected GPU before starting ffmpeg.
        nets = self._load_nets(w, h)
        if nets is None:
            return False

        source = gpu_io.open_reader(self.input_path, info, self.gpu_ids[0], find_tool("ffmpeg"),
                                    _in_matrix(info), log=self.log)
        self.log(f"Decoding: {source.kind}")
        encoder = self._start_encoder(info) if self.frame_sink is None else None

        n_workers = sum(self.spec.workers_per_gpu for _ in self.gpu_ids)
        in_q: queue.Queue = queue.Queue(maxsize=n_workers * 2)
        results: dict[int, bytes] = {}
        cond = threading.Condition()
        state = {"next": 0, "done": 0}
        per_gpu = {gid: 0 for gid in self.gpu_ids}
        max_ahead = n_workers * 4

        def put(item):
            # Never block forever: give up if the job was cancelled.
            while True:
                try:
                    in_q.put(item, timeout=0.5)
                    return True
                except queue.Full:
                    if self.cancel_event.is_set():
                        return False

        def reader():
            try:
                for idx, src in enumerate(source.frames(start_frame, total)):
                    if self.cancel_event.is_set() or not put((idx, src)):
                        break
            except Exception as e:  # noqa: BLE001
                self._fail(e)
            finally:
                source.close()
                for _ in range(n_workers):
                    if not put(None):
                        break

        def worker(gid):
            net = nets[gid]
            dev = torch.device("cuda", gid)
            stream = torch.cuda.Stream(dev)
            tile = self.tile
            try:
                while True:
                    item = in_q.get()
                    if item is None or self.cancel_event.is_set():
                        return
                    idx, buf = item
                    with torch.cuda.device(dev), torch.cuda.stream(stream), torch.inference_mode():
                        while True:
                            try:
                                out = self._upscale_frame(net, buf, w, h, dev, tile)
                                break
                            except torch.cuda.OutOfMemoryError:
                                torch.cuda.empty_cache()
                                if getattr(net, "is_tensorrt", False):
                                    raise RuntimeError(f"GPU {gid} ran out of memory with TensorRT. "
                                                       "Turn TensorRT off for this video.")
                                if tile and tile <= 128:
                                    raise
                                tile = (tile // 2) if tile else 512
                                self.log(f"GPU {gid}: out of memory, switching to {tile}px tiles")
                        stream.synchronize()
                    with cond:
                        while idx - state["next"] > max_ahead and not self.cancel_event.is_set():
                            cond.wait(0.5)
                        results[idx] = (buf, out)
                        per_gpu[gid] += 1
                        cond.notify_all()
            except Exception as e:  # noqa: BLE001
                self._fail(e)

        def writer():
            t0 = time.time()
            last_report = 0.0
            try:
                while True:
                    with cond:
                        while state["next"] not in results:
                            if self.cancel_event.is_set():
                                return
                            if not any(t.is_alive() for t in workers):
                                if state["next"] not in results:
                                    return
                            cond.wait(0.25)
                        src, data = results.pop(state["next"])
                        idx = state["next"]
                        state["next"] += 1
                        cond.notify_all()
                    if self.frame_sink is not None:
                        self.frame_sink(idx, gpu_io.src_to_rgb_bytes(src, w, h), data)
                    else:
                        encoder.stdin.write(data)
                    state["done"] += 1
                    now = time.time()
                    if now - last_report > 0.5:
                        last_report = now
                        fps = state["done"] / max(now - t0, 1e-6)
                        remaining = max(total - state["done"], 0)
                        self.on_progress(Progress(state["done"], total, fps, remaining / fps if fps else 0, dict(per_gpu)))
            except Exception as e:  # noqa: BLE001
                self._fail(e)

        workers = []
        for gid in self.gpu_ids:
            for _ in range(self.spec.workers_per_gpu):
                workers.append(threading.Thread(target=worker, args=(gid,), daemon=True))
        threads = [threading.Thread(target=reader, daemon=True), *workers]
        for t in threads:
            t.start()
        wt = threading.Thread(target=writer, daemon=True)
        wt.start()
        wt.join()
        for t in threads:
            t.join(timeout=5)

        if encoder is not None:
            try:
                encoder.stdin.close()
            except OSError:
                pass
        if self.cancel_event.is_set() or self._error:
            self.cancel()
            source.close()
            if encoder is not None:
                encoder.wait()
                self._remove_output()
            if self._error:
                raise RuntimeError(f"{self._error}\n{self._stderr_tail()}")
            self.log("Cancelled.")
            return False

        if encoder is None:  # preview clip
            source.close()
            self.on_progress(Progress(state["done"], state["done"], 0, 0, dict(per_gpu)))
            return True
        rc = encoder.wait()
        if rc != 0:
            self._remove_output()
            raise RuntimeError(f"Encoder failed (exit {rc}).\n{self._stderr_tail()}")
        self.on_progress(Progress(state["done"], state["done"], 0, 0, dict(per_gpu)))
        self.log(f"Done: {state['done']} frames. " + ", ".join(f"GPU {g}: {n}" for g, n in per_gpu.items()))
        return True

    # ------------------------------------------------------------------ helpers
    def _load_nets(self, w, h):
        nets = get_nets(self.spec, self.gpu_ids, self.use_tensorrt, w, h, log=self.log,
                        cancel_event=self.cancel_event)
        if nets:
            ws = [threading.Thread(target=n.warm_up, args=(self.spec.workers_per_gpu,), daemon=True)
                  for n in nets.values() if hasattr(n, "warm_up")]
            for t in ws:
                t.start()
            for t in ws:
                t.join()
        return nets

    def _upscale_frame(self, net, src, w, h, dev, tile):
        return upscale_frame(net, self.spec, src, w, h, dev, tile)

    def _start_decoder(self, info):
        matrix = _in_matrix(info)
        cmd = [find_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin"]
        if self.start:
            cmd += ["-ss", f"{self.start:.3f}"]
        cmd += ["-i", self.input_path]
        if self.max_seconds:
            cmd += ["-t", str(self.max_seconds)]
        cmd += ["-map", "0:v:0", "-an", "-sn", "-fps_mode", "passthrough",
                "-vf", f"scale=in_color_matrix={matrix}:flags=bicubic,format=rgb24",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        return self._popen(cmd, stdout=subprocess.PIPE)

    def _start_encoder(self, info):
        codec, extra, pix = ENCODERS[self.encoder_label]
        is_mp4 = self.output_path.lower().endswith((".mp4", ".m4v", ".mov"))
        cmd = [find_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{self.out_w}x{self.out_h}",
               "-framerate", str(info.fps), "-i", "-",
               *(["-ss", f"{self.start:.3f}"] if self.start else []), "-i", self.input_path,
               "-map", "0:v:0", "-map", "1:a?", "-map_metadata", "1"]
        if self.max_seconds:
            cmd += ["-t", str(self.max_seconds)]
        cmd += ["-vf", f"scale=out_color_matrix=bt709:out_range=tv,format={pix}",
                "-c:v", codec, *extra, "-preset", "p6", "-tune", "hq", "-rc", "vbr",
                "-cq", str(self.cq), "-b:v", "0",
                "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]
        if info.audio_codec and (not is_mp4 or info.audio_codec in MP4_SAFE_AUDIO):
            cmd += ["-c:a", "copy"]
        else:
            cmd += ["-c:a", "aac", "-b:a", "320k"]
        if is_mp4:
            cmd += ["-movflags", "+faststart"]
        cmd.append(self.output_path)
        return self._popen(cmd, stdin=subprocess.PIPE)

    def _popen(self, cmd, **kw):
        p = subprocess.Popen(cmd, stderr=subprocess.PIPE, creationflags=NO_WINDOW, bufsize=0, **kw)
        p.stderr_lines = []
        def drain():
            for line in iter(p.stderr.readline, b""):
                text = line.decode(errors="replace").rstrip()
                p.stderr_lines.append(text)
                self.log("ffmpeg: " + text)
        threading.Thread(target=drain, daemon=True).start()
        self._procs.append(p)
        return p

    def _stderr_tail(self):
        lines = [l for p in self._procs for l in p.stderr_lines[-5:]]
        return "\n".join(lines)

    def _fail(self, e):
        if self._error is None and not self.cancel_event.is_set():
            self._error = e
            self.log(f"Error: {e}")
        self.cancel_event.set()
        for p in self._procs:
            if p.poll() is None:
                p.kill()

    def _remove_output(self):
        try:
            if os.path.exists(self.output_path):
                os.remove(self.output_path)
        except OSError:
            pass

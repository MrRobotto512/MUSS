"""NVIDIA TensorRT acceleration.

A model is exported to ONNX (FP16) for one input resolution, then TensorRT compiles it into
an "engine" optimized for one specific GPU architecture. Your RTX 4070 (Ada, sm89) and
RTX 3060 (Ampere, sm86) each get their own engine. Engines are cached in ./engines, so
the slow build (a few minutes) only happens the first time.
"""
import os
import re
import threading

import torch

from .models import MODELS_DIR, ModelSpec, load_model

ENGINES_DIR = os.path.join(os.path.dirname(MODELS_DIR), "engines")
_export_lock = threading.Lock()

try:
    import tensorrt as trt
    _TRT_ERROR = None
    _LOGGER = trt.Logger(trt.Logger.WARNING)
except Exception as e:  # noqa: BLE001
    trt = None
    _TRT_ERROR = str(e)


def available() -> bool:
    return trt is not None


def status() -> str:
    return f"TensorRT {trt.__version__}" if trt else f"TensorRT unavailable: {_TRT_ERROR}"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")


def engine_path(spec: ModelSpec, gpu_id: int, h: int, w: int) -> str:
    major, minor = torch.cuda.get_device_capability(gpu_id)
    gpu = _slug(torch.cuda.get_device_name(gpu_id).replace("NVIDIA GeForce", ""))
    return os.path.join(ENGINES_DIR, f"{spec.file_id}_{w}x{h}_{gpu}_sm{major}{minor}_trt{trt.__version__}_v3.engine")


def _precision(spec: ModelSpec) -> str:
    return "fp16" if spec.half else "fp32"


def _onnx_path(spec: ModelSpec, h: int, w: int) -> str:
    # "v3": dynamic-shape export (older fixed-shape exports gave multi-GB engines)
    return os.path.join(ENGINES_DIR, f"{spec.file_id}_{w}x{h}_{_precision(spec)}_v3.onnx")


def _failed_path(spec: ModelSpec, h: int, w: int) -> str:
    ver = trt.__version__ if trt else "none"
    return os.path.join(ENGINES_DIR, f"{spec.file_id}_{w}x{h}_trt{ver}.failed")


def mark_failed(spec: ModelSpec, h: int, w: int, reason: str):
    os.makedirs(ENGINES_DIR, exist_ok=True)
    with open(_failed_path(spec, h, w), "w", encoding="utf-8") as f:
        f.write(reason)
    for p in (_onnx_path(spec, h, w), _onnx_path(spec, h, w) + ".tmp"):
        try:
            os.remove(p)
        except OSError:
            pass


def failed_reason(spec: ModelSpec, h: int, w: int) -> str | None:
    try:
        with open(_failed_path(spec, h, w), encoding="utf-8") as f:
            return f.read().strip() or "conversion failed"
    except OSError:
        return None


def _export_onnx(spec: ModelSpec, h: int, w: int, gpu_id: int, log) -> str:
    path = _onnx_path(spec, h, w)
    with _export_lock:
        if os.path.exists(path):
            return path
        os.makedirs(ENGINES_DIR, exist_ok=True)
        log(f"Exporting {spec.filename} to ONNX ({w}x{h}, {_precision(spec).upper()})...")
        dev = torch.device("cuda", gpu_id)
        net = load_model(spec, dev, half=True, log=log).to(memory_format=torch.contiguous_format)
        x = torch.rand(1, 3, h, w, device=dev, dtype=torch.float16 if spec.half else torch.float32)
        tmp = path + ".tmp"
        try:
            with torch.inference_mode():
                # Dynamic height/width: with a fixed 1080p shape TensorRT bakes shape-specific data into
                # the engine (1.8-4.5 GB for Compact models); dynamic engines are ~1 MB and as fast.
                torch.onnx.export(net, (x,), tmp, input_names=["input"], output_names=["output"],
                                  opset_version=17, dynamo=False,
                                  dynamic_axes={"input": {2: "height", 3: "width"},
                                                "output": {2: "out_height", 3: "out_width"}})
            os.replace(tmp, path)
        finally:
            del net
            torch.cuda.empty_cache()
    return path


def ensure_engine(spec: ModelSpec, gpu_id: int, h: int, w: int, log=print) -> str:
    """Build (or reuse) the TensorRT engine for this model, resolution and GPU."""
    path = engine_path(spec, gpu_id, h, w)
    if os.path.exists(path):
        return path
    onnx_file = _export_onnx(spec, h, w, gpu_id, log)
    name = torch.cuda.get_device_name(gpu_id)
    log(f"Building TensorRT engine for GPU {gpu_id} ({name}) - first time only, this can take several minutes...")
    with torch.cuda.device(gpu_id):
        builder = trt.Builder(_LOGGER)
        # The ONNX graph is already FP16, so a strongly typed network keeps it FP16.
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        parser = trt.OnnxParser(network, _LOGGER)
        if not parser.parse_from_file(onnx_file):
            errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"TensorRT could not read the ONNX model: {errs}")
        config = builder.create_builder_config()
        profile = builder.create_optimization_profile()
        m = max(spec.pad_multiple, 1)
        small = (1, 3, min(h, 64 * m), min(w, 64 * m))
        profile.set_shape("input", small, (1, 3, h, w), (1, 3, h, w))
        config.add_optimization_profile(profile)
        free, _ = torch.cuda.mem_get_info(gpu_id)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(min(4 << 30, free * 0.6)))
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT engine build failed (see log).")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(serialized)
    os.replace(tmp, path)
    log(f"TensorRT engine ready for GPU {gpu_id}.")
    return path


class TRTModel:
    """Callable like the PyTorch model: takes a (1,3,H,W) FP16 CUDA tensor, returns the output tensor.

    Each calling thread gets its own execution context, so several worker threads can
    share one engine on the same GPU.
    """

    is_tensorrt = True

    def __init__(self, path: str, gpu_id: int):
        self.gpu_id = gpu_id
        self.device = torch.device("cuda", gpu_id)
        with torch.cuda.device(gpu_id):
            self._runtime = trt.Runtime(_LOGGER)
            with open(path, "rb") as f:
                self.engine = self._runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Could not load TensorRT engine {os.path.basename(path)} (delete it to rebuild).")
        shape = tuple(self.engine.get_tensor_shape("input"))
        # dynamic engines report -1 for height/width: the frame size is set per call (up to the profile max)
        self.dynamic = any(d < 0 for d in shape)
        self.in_shape = tuple(self.engine.get_tensor_profile_shape("input", 0)[2]) if self.dynamic else shape
        to_torch = {trt.DataType.HALF: torch.float16, trt.DataType.FLOAT: torch.float32}
        self.in_dtype = to_torch.get(self.engine.get_tensor_dtype("input"), torch.float16)
        self.out_dtype = to_torch.get(self.engine.get_tensor_dtype("output"), torch.float16)
        # Execution contexts are expensive to create (they allocate scratch GPU memory), so they
        # are pooled and reused by later jobs instead of being created per worker thread.
        self._pool: list = []
        self._pool_lock = threading.Lock()

    def _acquire(self):
        with self._pool_lock:
            if self._pool:
                return self._pool.pop()
        with torch.cuda.device(self.gpu_id):
            return self.engine.create_execution_context()

    def _release(self, ctx):
        with self._pool_lock:
            self._pool.append(ctx)

    def warm_up(self, n_contexts: int):
        """Create contexts ahead of time so the first frames are not delayed."""
        with self._pool_lock:
            need = n_contexts - len(self._pool)
        for _ in range(max(need, 0)):
            self._release(self._acquire())

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if (tuple(x.shape) != self.in_shape if not self.dynamic
                else any(a > b for a, b in zip(x.shape, self.in_shape))):
            raise RuntimeError(f"TensorRT engine expects input {self.in_shape}, got {tuple(x.shape)}")
        x = x.to(self.in_dtype).contiguous()  # engines use plain NCHW layout
        ctx = self._acquire()
        try:
            if self.dynamic:
                ctx.set_input_shape("input", tuple(x.shape))
            out = torch.empty(tuple(ctx.get_tensor_shape("output")), dtype=self.out_dtype, device=self.device)
            ctx.set_tensor_address("input", x.data_ptr())
            ctx.set_tensor_address("output", out.data_ptr())
            stream = torch.cuda.current_stream(self.device)
            if not ctx.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT inference failed.")
            stream.synchronize()  # the context may be handed to another thread right after this
        finally:
            self._release(ctx)
        return out


def padded_size(spec: ModelSpec, h: int, w: int):
    m = spec.pad_multiple
    return h + (-h) % m, w + (-w) % m

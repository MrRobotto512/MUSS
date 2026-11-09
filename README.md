# MUSS — Movie UpScaling Software

A desktop app for AI video upscaling (1080p → 4K) that uses **every NVIDIA GPU you have**, and optionally the GPUs of other PCs on your network.

## Features

- **Multi-GPU rendering**: video is split into segments; each GPU decodes, upscales and encodes its own segments, and faster cards automatically take more of the work.
- **GPU end to end**: NVDEC decoding straight into GPU memory, AI on the GPU, RGB→YUV conversion on the GPU, NVENC encoding (HEVC 10-bit, AV1, H.264). Falls back to CPU decoding for formats NVDEC can't handle.
- **NVIDIA TensorRT** acceleration (~1.5–1.7× faster than PyTorch), with engines built once per model, resolution and GPU.
- **Load any model**: built-in Real-ESRGAN models, plus any `.pth` / `.safetensors` upscaling model from e.g. [OpenModelDB](https://openmodeldb.info) (ESRGAN, Compact, SPAN, SwinIR, HAT, DAT, OmniSR, … via [spandrel](https://github.com/chaiNNer-org/spandrel)). Model files are loaded weights-only, so they can't run code.
- **Preview like a pro**: timeline with filmstrip, in/out preview range, real-time playback of the rendered range, split / side-by-side / original / enhanced compare, zoom and pan.
- **Resume**: stopped, crashed or interrupted exports continue from the last finished segment.
- **Render nodes (optional)**: other PCs running MUSS can render segments over your local network (pairing-code protected). If a node drops out, its work goes back to the queue.
- **Schedule**: run the queue only within a time window (e.g. overnight); it pauses at the end of the window and resumes in the next one.
- Queue with batch export, range export, quick 10-second tests.

## Requirements

- Windows 10/11, one or more NVIDIA RTX GPUs (RTX 20 series or newer recommended) with a recent driver
- Python 3.14
- `ffmpeg` / `ffprobe` with NVENC: MUSS uses the copy bundled with Topaz Video AI if it is installed, otherwise the ones on your `PATH`

## Setup

```bat
setup.bat
```

This creates `.venv` and installs PyTorch (CUDA 12.8), TensorRT, PySide6 and the other dependencies (several GB). Then start the app with **`MUSS.bat`**.

> `torch` / `torchvision` are pinned to the CUDA builds in `requirements.txt`. Installing other torch-dependent packages without `--extra-index-url https://download.pytorch.org/whl/cu128` can silently replace them with CPU-only builds.

## Render node on another PC

Install MUSS on the other PC, then either switch on **Render node** in the app's top bar or run `Render Node.bat` (`python render_node.py --help` for options). On the main PC add it under **Settings → Render nodes → + Add node** with the address and pairing code it shows. Allow Python through Windows Firewall on private networks. Video is sent unencrypted over your local network.

## Project layout

| Path | What it is |
|---|---|
| `app.py` | The desktop app (PySide6) |
| `timeline.py` | Timeline widget (filmstrip, preview range, playhead) |
| `upscaler/engine.py` | Model cache, frame upscaling, preview rendering |
| `upscaler/export.py` | Segmented, resumable multi-GPU export |
| `upscaler/gpu_io.py` | NVDEC decoding and GPU colour conversion |
| `upscaler/trt_backend.py` | ONNX export and TensorRT engines |
| `upscaler/models.py` | Built-in and custom model handling |
| `upscaler/node.py` | Render node server and client |
| `upscaler/temporal.py` | Experimental temporal stabilizer (not wired in yet) |
| `render_node.py` | Headless render node |

## Third-party components

Downloaded at runtime or installed as dependencies, not included in this repository: Real-ESRGAN model weights (BSD-3-Clause), PyTorch / torchvision (BSD), spandrel (MIT), PyNvVideoCodec (MIT), NVIDIA TensorRT (NVIDIA license), PySide6 (LGPL-3.0), ffmpeg (LGPL/GPL).

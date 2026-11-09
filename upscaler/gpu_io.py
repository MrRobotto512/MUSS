"""Frame input/output on the GPU.

Decoding: NVIDIA NVDEC via PyNvVideoCodec decodes straight into GPU memory and converts to RGB
there; frames are handed to PyTorch with zero copies (DLPack). Each GPU decodes its own frames.
If NVDEC can't handle a file (e.g. ProRes), frames are decoded by ffmpeg on the CPU instead.

Encoding: the AI output is converted from RGB to YUV (BT.709, limited range, 4:2:0) on the GPU,
so the encoder receives ready-to-encode frames (half the data, no CPU colour conversion).
"""
import subprocess
from typing import Iterator

import torch

try:
    import PyNvVideoCodec as nvc
    _NVDEC_ERROR = None
except Exception as e:  # noqa: BLE001
    nvc = None
    _NVDEC_ERROR = str(e)

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def nvdec_available() -> bool:
    return nvc is not None


class NvdecReader:
    """Frame-exact random access decoding on one GPU. Yields uint8 CHW RGB tensors on that GPU."""

    kind = "GPU (NVDEC)"

    def __init__(self, path: str, gpu_id: int):
        self.gpu_id = gpu_id
        self.dec = nvc.SimpleDecoder(path, gpu_id=gpu_id, use_device_memory=True,
                                     output_color_type=nvc.OutputColorType.RGBP)
        self.num_frames = int(self.dec.get_stream_metadata().num_frames)

    def frames(self, start: int, count: int) -> Iterator[torch.Tensor]:
        start = max(0, min(start, self.num_frames))
        count = max(0, min(count, self.num_frames - start))
        self.dec.seek_to_index(start)
        left = count
        while left > 0:
            batch = self.dec.get_batch_frames(min(8, left))
            if not batch:
                break
            for fr in batch:
                # clone: the decoder recycles its output surfaces
                yield torch.from_dlpack(fr).clone()
            left -= len(batch)

    def close(self):
        self.dec = None


class FfmpegReader:
    """CPU fallback: ffmpeg decodes, frames arrive as RGB24 bytes."""

    kind = "CPU (ffmpeg)"

    def __init__(self, path: str, info, ffmpeg: str, matrix: str):
        self.path, self.info, self.ffmpeg, self.matrix = path, info, ffmpeg, matrix
        self.proc = None

    def frames(self, start: int, count: int) -> Iterator[bytearray]:
        fps = float(self.info.fps)
        w, h = self.info.width, self.info.height
        cmd = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
        if start > 0:
            # seek half a frame early: the exact timestamp of frame N can round to just past it
            cmd += ["-ss", f"{(start - 0.5) / fps:.6f}"]
        cmd += ["-i", self.path, "-map", "0:v:0", "-an", "-sn", "-fps_mode", "passthrough", "-frames:v", str(count),
                "-vf", f"scale=in_color_matrix={self.matrix}:flags=bicubic,format=rgb24",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     creationflags=NO_WINDOW, bufsize=0)
        size = w * h * 3
        try:
            for _ in range(count):
                buf = bytearray(size)
                view, got = memoryview(buf), 0
                while got < size:
                    n = self.proc.stdout.readinto(view[got:])
                    if not n:
                        break
                    got += n
                if got < size:
                    break
                yield buf
        finally:
            self.close()

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
        if self.proc:
            self.proc.wait()
        self.proc = None


def open_reader(path, info, gpu_id, ffmpeg, matrix, prefer_gpu=True, log=print):
    """NVDEC reader on gpu_id if it can decode this file, otherwise the ffmpeg CPU reader."""
    if prefer_gpu and nvc is not None:
        try:
            r = NvdecReader(path, gpu_id)
            next(r.frames(0, 1))           # proves this codec/profile is supported by NVDEC
            return r
        except Exception as e:  # noqa: BLE001
            log(f"GPU decoding not available for this file ({str(e).splitlines()[0][:120]}); using CPU decoding.")
    return FfmpegReader(path, info, ffmpeg, matrix)


def to_input(src, dev, dtype, w, h) -> torch.Tensor:
    """Frame from either reader -> (1,3,H,W) float tensor in [0,1] on dev."""
    if isinstance(src, torch.Tensor):
        t = src.to(dev, non_blocking=True)
        scale = 65535.0 if t.dtype == torch.uint16 else 255.0
        return t.unsqueeze(0).to(dtype).div_(scale)
    t = torch.frombuffer(src, dtype=torch.uint8).to(dev, non_blocking=True)
    return t.view(h, w, 3).permute(2, 0, 1).unsqueeze(0).to(dtype).div_(255.0)


def src_to_rgb_bytes(src, w, h) -> bytes:
    """Original frame as RGB24 bytes (for the preview viewer)."""
    if isinstance(src, torch.Tensor):
        if src.dtype != torch.uint8:
            src = (src.float() / 257.0).round().clamp(0, 255).to(torch.uint8)
        return src.permute(1, 2, 0).contiguous().cpu().numpy().tobytes()
    return bytes(src)


# --------------------------------------------------------------------------- RGB -> YUV on the GPU
_KR, _KB = 0.2126, 0.0722          # BT.709
_KG = 1 - _KR - _KB


def rgb_to_yuv420(y: torch.Tensor, ten_bit: bool) -> bytes:
    """(1,3,H,W) float RGB in [0,1] -> raw yuv420p (8-bit) or p010le (10-bit) bytes, BT.709 limited range."""
    rgb = y[0].float().clamp(0, 1)
    r, g, b = rgb[0], rgb[1], rgb[2]
    luma = _KR * r + _KG * g + _KB * b
    cb = (b - luma) / (2 * (1 - _KB))
    cr = (r - luma) / (2 * (1 - _KR))
    # 4:2:0: average each 2x2 block of chroma (H and W are even)
    cb = torch.nn.functional.avg_pool2d(cb[None, None], 2)[0, 0]
    cr = torch.nn.functional.avg_pool2d(cr[None, None], 2)[0, 0]
    if ten_bit:
        Y = (64 + 876 * luma).round().clamp(0, 1023).to(torch.int32)
        U = (512 + 896 * cb).round().clamp(0, 1023).to(torch.int32)
        V = (512 + 896 * cr).round().clamp(0, 1023).to(torch.int32)
        # p010le: 16-bit little-endian samples with the 10 bits in the top bits; UV interleaved
        vals = torch.cat(((Y << 6).flatten(), (torch.stack((U, V), dim=-1) << 6).flatten()))
        le = torch.stack(((vals & 0xFF), (vals >> 8)), dim=-1).to(torch.uint8)   # explicit little-endian bytes
        return le.flatten().cpu().numpy().tobytes()
    Y = (16 + 219 * luma).round().clamp(0, 255).to(torch.uint8)
    U = (128 + 224 * cb).round().clamp(0, 255).to(torch.uint8)
    V = (128 + 224 * cr).round().clamp(0, 255).to(torch.uint8)
    return torch.cat((Y.flatten(), U.flatten(), V.flatten())).cpu().numpy().tobytes()

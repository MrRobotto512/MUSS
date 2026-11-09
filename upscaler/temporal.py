"""Temporal stability: reduces the frame-to-frame "shimmer" of single-frame upscaling models.

For each frame t, optical flow (RAFT-small, at reduced resolution) tells where every pixel was
in frame t-1. The previous *stabilized* 4K output is warped along that flow and blended into
the new AI output - but only where the two source frames actually agree after warping (a
photometric check), so moving edges, newly uncovered areas and lighting changes stay sharp
and never ghost. A large mismatch over the whole frame is treated as a scene cut (reset).

The result is a per-pixel recursive filter: stable areas average the invented detail over
several frames (no boiling), changing areas follow the new frame.
"""
import os
import threading

import torch
import torch.nn.functional as F

from .models import MODELS_DIR

_RAFT = {}
_RAFT_LOCK = threading.Lock()
FLOW_SIZE = (544, 960)          # flow is computed at about half of 1080p (h, w)


def _raft(dev: torch.device):
    """RAFT-small optical flow network on dev (weights ~4 MB, downloaded once from pytorch.org)."""
    with _RAFT_LOCK:
        if dev not in _RAFT:
            os.environ.setdefault("TORCH_HOME", os.path.join(MODELS_DIR, "torch_hub"))
            from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
            _RAFT[dev] = raft_small(weights=Raft_Small_Weights.DEFAULT).eval().to(dev)
        return _RAFT[dev]


def _grid(h, w, dev):
    yy, xx = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32),
                            torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")
    return xx, yy


def _warp(img: torch.Tensor, flow: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Backward-warp img (N,C,H,W) with flow (N,2,H,W) in pixels: out(x) = img(x + flow(x))."""
    _, _, h, w = img.shape
    xx, yy = _grid(h, w, img.device)
    gx = (xx + flow[:, 0]) * (2.0 / max(w - 1, 1)) - 1.0
    gy = (yy + flow[:, 1]) * (2.0 / max(h - 1, 1)) - 1.0
    grid = torch.stack((gx, gy), dim=-1)
    return F.grid_sample(img, grid, mode=mode, padding_mode="border", align_corners=True)


class TemporalStabilizer:
    """Stateful per-stream filter. Feed frames in order with process(); call reset() between clips."""

    def __init__(self, strength: float, dev: torch.device, sigma: float = 0.035, cut_threshold: float = 0.12):
        self.strength = float(max(0.0, min(strength, 1.0)))
        self.dev = dev
        self.sigma = sigma                  # photometric tolerance (0..1 scale) before blending fades out
        self.cut_threshold = cut_threshold  # mean warp error above this = scene cut
        self.prev_src = None
        self.prev_out = None
        self.last_mask_mean = 0.0
        self.cuts = 0

    @property
    def enabled(self) -> bool:
        return self.strength > 0

    def reset(self):
        self.prev_src = None
        self.prev_out = None

    @torch.inference_mode()
    def process(self, src: torch.Tensor, ai: torch.Tensor) -> torch.Tensor:
        """src: (1,3,h,w) source frame in [0,1]; ai: (1,3,H,W) AI output in [0,1] (H=2h). Returns stabilized output."""
        src = src.to(self.dev, torch.float32)
        ai = ai.to(self.dev, torch.float32)
        if not self.enabled:
            return ai
        if self.prev_src is None:
            self.prev_src, self.prev_out = src, ai
            return ai
        h, w = src.shape[-2:]
        H, W = ai.shape[-2:]
        fh, fw = min(FLOW_SIZE[0], h), min(FLOW_SIZE[1], w)
        fh, fw = fh - fh % 8, fw - fw % 8           # RAFT needs multiples of 8

        # flow from the current frame back to the previous one, at reduced size
        a = F.interpolate(src, size=(fh, fw), mode="bilinear", antialias=True, align_corners=False) * 2 - 1
        b = F.interpolate(self.prev_src, size=(fh, fw), mode="bilinear", antialias=True, align_corners=False) * 2 - 1
        flow = _raft(self.dev)(a, b, num_flow_updates=6)[-1]

        # Does the warped previous frame match the current one? Checked on reduced, slightly blurred
        # frames so film grain / sensor noise (which changes every frame) doesn't count as a mismatch,
        # while real motion errors, occlusions and lighting changes still do.
        err = (_warp(b, flow) - a).abs().mean(dim=1, keepdim=True) * 0.5     # back to the 0..1 scale
        err = F.avg_pool2d(err, 3, stride=1, padding=1)
        if err.mean().item() > self.cut_threshold:  # scene cut or flash: start over
            self.cuts += 1
            self.prev_src, self.prev_out = src, ai
            return ai
        err = F.max_pool2d(err, 5, stride=1, padding=2)              # be conservative around edges
        mask = torch.exp(-(err / self.sigma) ** 2)
        flow_s = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False)
        flow_s[:, 0] *= w / fw
        flow_s[:, 1] *= h / fh

        # warp the previous stabilized 4K output and blend where the mask trusts it
        flow_o = F.interpolate(flow_s, size=(H, W), mode="bilinear", align_corners=False) * (W / w)
        # bicubic: a bilinear warp, re-applied frame after frame, slowly blurs the recursive result
        warped = _warp(self.prev_out, flow_o, mode="bicubic").clamp_(0, 1)
        mask_o = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False)
        wgt = self.strength * 0.85 * mask_o                          # never fully freeze a pixel
        out = ai + wgt * (warped - ai)
        self.last_mask_mean = float(mask.mean())
        self.prev_src, self.prev_out = src, out
        return out

"""Audio upscaling: restore the high frequencies lost in low-bitrate or band-limited audio (AudioSR).

AudioSR (https://github.com/haoheliu/versatile_audio_super_resolution, MIT; 'basic' weights Apache-2.0)
is a latent-diffusion model that generates the missing band above a cutoff for any audio - music,
effects and speech - at 48 kHz. MUSS uses it like this:

  1. measure where the audio's treble actually stops (cutoff); full-range audio is left alone
  2. per channel, process 10.24 s chunks (the size the model was trained on) with overlap
  3. keep the original audio untouched and ADD only the generated band above the cutoff,
     level-matched to the original and cross-faded between chunks - so the existing sound can't
     change and the volume can't jump between chunks
"""
import os
import threading
import time

import numpy as np
import torch

from .models import MODELS_DIR

SR = 48000
CHUNK = 10.24                       # seconds: the model's native window (latent_t_size 128)
OVERLAP = 1.28                      # seconds of cross-fade between chunks
WEIGHTS_REPO = "haoheliu/audiosr_basic"
WEIGHTS_PATH = os.path.join(MODELS_DIR, "audiosr", "audiosr_basic.bin")
WEIGHTS_SIZE = 6_178_000_000        # approximate, for messages
FULL_RANGE_HZ = 18500               # cutoffs at or above this don't need restoring (320k MP3, AC-3, CD: 19-22 kHz)
DEFAULT_STEPS = 25                  # measured: as good as 50 steps (LSD 1.31 vs 1.34) at twice the speed

_MODEL = {}
_MODEL_LOCK = threading.Lock()


# --------------------------------------------------------------------------- model
def _config():
    from .audiosr.latent_diffusion.util import instantiate_from_config  # noqa: F401  (import check)
    vae = {
        "target": "upscaler.audiosr.latent_encoder.autoencoder.AutoencoderKL",
        "params": {
            "sampling_rate": SR, "batchsize": 4, "monitor": "val/rec_loss", "image_key": "fbank", "subband": 1,
            "embed_dim": 16, "time_shuffle": 1,
            "ddconfig": {"double_z": True, "mel_bins": 256, "z_channels": 16, "resolution": 256,
                         "downsample_time": False, "in_channels": 1, "out_ch": 1, "ch": 128, "ch_mult": [1, 2, 4, 8],
                         "num_res_blocks": 2, "attn_resolutions": [], "dropout": 0.1},
        },
    }
    return {
        "first_stage_config": vae, "base_learning_rate": 0.0001, "warmup_steps": 5000,
        "optimize_ddpm_parameter": True, "sampling_rate": SR, "batchsize": 16, "beta_schedule": "cosine",
        "linear_start": 0.0015, "linear_end": 0.0195, "num_timesteps_cond": 1, "log_every_t": 200,
        "timesteps": 1000, "unconditional_prob_cfg": 0.1, "parameterization": "v", "first_stage_key": "fbank",
        "latent_t_size": 128, "latent_f_size": 32, "channels": 16, "monitor": "val/loss_simple_ema",
        "scale_by_std": True,
        "unet_config": {
            "target": "upscaler.audiosr.latent_diffusion.modules.diffusionmodules.openaimodel.UNetModel",
            "params": {"image_size": 64, "in_channels": 32, "out_channels": 16, "model_channels": 128,
                       "attention_resolutions": [8, 4, 2], "num_res_blocks": 2, "channel_mult": [1, 2, 3, 5],
                       "num_head_channels": 32, "extra_sa_layer": True, "use_spatial_transformer": True,
                       "transformer_depth": 1},
        },
        "evaluation_params": {"unconditional_guidance_scale": 3.5, "ddim_sampling_steps": 200,
                              "n_candidates_per_samples": 1},
        "cond_stage_config": {
            "concat_lowpass_cond": {
                "cond_stage_key": "lowpass_mel", "conditioning_key": "concat",
                "target": "upscaler.audiosr.latent_diffusion.modules.encoders.modules.VAEFeatureExtract",
                "params": {"first_stage_config": vae},
            }
        },
    }


def weights_present() -> bool:
    return os.path.exists(WEIGHTS_PATH) and os.path.getsize(WEIGHTS_PATH) > 1_000_000_000


def download_weights(progress=None, log=print):
    """Download the 6.2 GB AudioSR 'basic' weights (Apache-2.0) from Hugging Face, SHA-256 verified."""
    import hashlib
    import json
    import urllib.request
    from . import __version__
    ua = {"User-Agent": f"MUSS/{__version__}"}
    info = json.loads(urllib.request.urlopen(urllib.request.Request(
        f"https://huggingface.co/api/models/{WEIGHTS_REPO}?blobs=true", headers=ua), timeout=30).read())
    sib = next(s for s in info["siblings"] if s["rfilename"] == "pytorch_model.bin")
    sha, size = sib["lfs"]["sha256"], sib["size"]
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    tmp = WEIGHTS_PATH + ".part"
    h = hashlib.sha256()
    log(f"Downloading the AudioSR model ({size / 1e9:.1f} GB, once)...")
    req = urllib.request.Request(f"https://huggingface.co/{WEIGHTS_REPO}/resolve/main/pytorch_model.bin", headers=ua)
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        done = 0
        while chunk := r.read(4 << 20):
            f.write(chunk)
            h.update(chunk)
            done += len(chunk)
            if progress:
                progress(done, size)
    if h.hexdigest() != sha:
        os.remove(tmp)
        raise RuntimeError("The AudioSR download was corrupted (checksum mismatch); please try again.")
    os.replace(tmp, WEIGHTS_PATH)


def load_model(gpu_id: int, log=print):
    """AudioSR on cuda:gpu_id (cached). Weights are read weights-only: no code in the file can run."""
    with _MODEL_LOCK:                                   # serialised: GPUs load one after another from shared weights
        if gpu_id in _MODEL:
            return _MODEL[gpu_id]
        if not weights_present():
            raise RuntimeError("The AudioSR model hasn't been downloaded yet.")
        from .audiosr.latent_diffusion.models.ddpm import LatentDiffusion
        dev = torch.device("cuda", gpu_id)
        log(f"Loading AudioSR on GPU {gpu_id}...")
        params = _config()
        params["device"] = dev
        model = LatentDiffusion(**params)
        if "state" not in _MODEL:                       # read the 6 GB file once, shared by all GPUs
            ckpt = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True, mmap=True)
            _MODEL["state"] = ckpt.get("state_dict", ckpt)
        state = _MODEL["state"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        # The upstream loads non-strictly. Every model weight must load; the only extras allowed are
        # the CLAP weights, which MUSS doesn't build because super-resolution never uses them.
        extra = [k for k in unexpected if not k.startswith("clap.")]
        if missing or extra:
            raise RuntimeError(f"AudioSR weights don't match the model ({len(missing)} missing, "
                               f"{len(extra)} unexpected, e.g. {(missing + extra)[:3]}).")
        model.eval().to(dev)
        _MODEL[gpu_id] = model
        log(f"AudioSR ready on GPU {gpu_id} ({len(state)} tensors loaded, {len(unexpected)} unused).")
        return model


def release():
    with _MODEL_LOCK:
        _MODEL.clear()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- features (as AudioSR, without librosa)
def _hz_to_mel(f):
    f = np.asarray(f, dtype=np.float64)
    f_sp, min_log_hz, logstep = 200.0 / 3, 1000.0, np.log(6.4) / 27.0
    mels = f / f_sp
    return np.where(f >= min_log_hz, min_log_hz / f_sp + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, mels)


def _mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    f_sp, min_log_hz, logstep = 200.0 / 3, 1000.0, np.log(6.4) / 27.0
    min_log_mel = min_log_hz / f_sp
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)


def slaney_mel_basis(sr=SR, n_fft=2048, n_mels=256, fmin=20.0, fmax=24000.0):
    """Same as librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax) with its defaults (Slaney scale + norm)."""
    fftfreqs = np.linspace(0, sr / 2, 1 + n_fft // 2)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fftfreqs[None, :]
    weights = np.zeros((n_mels, len(fftfreqs)))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels]))[:, None]
    return weights.astype(np.float32)


_MEL = {}


def _mel_and_stft(wav: torch.Tensor):
    """AudioSR's mel_spectrogram_train: wav (1, samples) -> log-mel (frames, 256), |stft| (frames, 1025)."""
    dev = wav.device
    if dev not in _MEL:
        _MEL[dev] = (torch.from_numpy(slaney_mel_basis()).to(dev), torch.hann_window(2048, device=dev))
    basis, win = _MEL[dev]
    y = torch.nn.functional.pad(wav.unsqueeze(1), (784, 784), mode="reflect").squeeze(1)
    spec = torch.stft(y, 2048, hop_length=480, win_length=2048, window=win, center=False, pad_mode="reflect",
                      normalized=False, onesided=True, return_complex=True).abs()
    mel = torch.log(torch.clamp(basis @ spec, min=1e-5))
    return mel[0].T, spec[0].T


def _pad_spec(x, frames):
    if x.shape[0] < frames:
        x = torch.nn.functional.pad(x, (0, 0, 0, frames - x.shape[0]))
    x = x[:frames]
    return x[..., :-1] if x.shape[-1] % 2 else x


def _cutoff_bin(stft_mag, percentile=0.985):
    """AudioSR's cutoff estimate: the bin below which `percentile` of the spectral energy lies."""
    energy = torch.cumsum(stft_mag.sum(0), 0)
    thr = energy[-1] * percentile
    above = torch.nonzero(energy >= thr)
    return int(above[0]) if len(above) else stft_mag.shape[1]


# --------------------------------------------------------------------------- analysis
def measure_cutoff(samples: np.ndarray, sr: int = SR) -> float:
    """Highest frequency (Hz) with real content, from the long-term average spectrum of the whole track.

    Finds where the spectrum falls to the noise floor and stays there - the "shelf" left by MP3/AAC
    encoders or low sample rates. Full-range 48 kHz audio returns ~20-24 kHz."""
    x = samples.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=0)
    n_fft = 4096
    if len(x) < n_fft * 4:
        return sr / 2
    # average power spectrum over up to ~400 evenly spaced frames
    starts = np.linspace(0, len(x) - n_fft, num=min(400, len(x) // n_fft)).astype(int)
    win = np.hanning(n_fft).astype(np.float32)
    frames = np.stack([x[s:s + n_fft] * win for s in starts])
    loud = np.sqrt((frames ** 2).mean(axis=1)) > 1e-4               # skip silence
    if loud.sum() < 4:
        return sr / 2
    psd = (np.abs(np.fft.rfft(frames[loud], axis=1)) ** 2).mean(axis=0) + 1e-20
    db = np.maximum(10 * np.log10(psd), -160.0)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    # A lossy encoder or a low sample rate leaves a "cliff": everything above the cutoff sits far below
    # the band just under it. Natural treble roll-off is gradual and never drops like that.
    best_f, best_drop = sr / 2, 0.0
    bw = 1000.0
    for f in np.arange(4000.0, sr / 2 - 500.0, 125.0):
        below = db[(freqs >= f - bw) & (freqs < f)].mean()
        above = db[freqs >= f + 125.0].mean()
        drop = below - above
        if drop > best_drop:
            best_f, best_drop = f, drop
    if best_drop <= 25.0:
        return sr / 2
    # refine: the exact frequency where the level falls halfway down the cliff
    k = 5
    db_s = np.convolve(db, np.ones(k) / k, mode="same")
    below = db[(freqs >= best_f - bw) & (freqs < best_f)].mean()
    mid = below - best_drop / 2
    win = np.nonzero((freqs >= best_f - 2 * bw) & (freqs <= best_f + 500) & (db_s > mid))[0]
    return float(freqs[win[-1]]) if len(win) else float(best_f)


# --------------------------------------------------------------------------- processing
def _generate_chunk(model, x: np.ndarray, steps: int, guidance: float, seed: int) -> np.ndarray:
    """Run AudioSR on one mono 48 kHz chunk of exactly CHUNK seconds. Returns its (normalised) output."""
    from .audiosr.lowpass import lowpass
    dev = next(model.parameters()).device
    xn = x - x.mean()
    xn = 0.5 * xn / (np.abs(xn).max() + 1e-8)
    frames = int(round(CHUNK * 100))
    wav = torch.from_numpy(xn[None].astype(np.float32))
    mel, stft = _mel_and_stft(wav)
    mel, stft = _pad_spec(mel, frames), _pad_spec(stft, frames)
    cutoff = _cutoff_bin(stft) / 1024 * 24000
    if cutoff < 1000:
        cutoff = 24000
    rng = np.random.RandomState(seed)
    ftype = rng.choice(["butter", "cheby1", "ellip", "bessel"])
    low = lowpass(xn.astype(np.float64), highcut=min(cutoff, 23999), fs=SR, order=8, _type=ftype)
    low = torch.from_numpy(np.ascontiguousarray(low[: len(xn)], dtype=np.float32))[None]
    low_mel, _ = _mel_and_stft(low)
    low_mel = _pad_spec(low_mel, frames)
    batch = {"waveform": wav[None], "stft": stft[None], "log_mel_spec": mel[None], "sampling_rate": SR,
             "waveform_lowpass": low[None], "lowpass_mel": low_mel[None]}
    torch.manual_seed(seed)
    with torch.no_grad(), torch.cuda.device(dev):
        out = model.generate_batch(batch, unconditional_guidance_scale=guidance, ddim_steps=steps, duration=CHUNK)
    return np.asarray(out, dtype=np.float32).reshape(-1)[: len(x)]


def _band_above(y: np.ndarray, x: np.ndarray, cutoff_hz: float) -> np.ndarray:
    """The part of model output y above cutoff_hz, level-matched to the original x using the band below."""
    n_fft, hop = 2048, 512
    win = torch.hann_window(n_fft)
    X = torch.stft(torch.from_numpy(x), n_fft, hop, window=win, return_complex=True)
    Y = torch.stft(torch.from_numpy(y), n_fft, hop, window=win, return_complex=True)
    freqs = torch.linspace(0, SR / 2, X.shape[0])
    ref = (freqs > cutoff_hz * 0.5) & (freqs < cutoff_hz * 0.95)      # band both signals share
    gain = (X[ref].abs().sum() / (Y[ref].abs().sum() + 1e-9)).clamp(0.05, 20.0)
    ramp = ((freqs - cutoff_hz) / 400.0).clamp(0, 1)                  # 400 Hz soft edge at the cutoff
    Hb = Y * gain * ramp[:, None]
    return torch.istft(Hb, n_fft, hop, window=win, length=len(x)).numpy()


def chunk_plan(n: int):
    """[(start, length)] of the overlapping chunks covering n samples."""
    L, ov = int(CHUNK * SR), int(OVERLAP * SR)
    return [(s, min(L, n - s)) for s in range(0, max(n - ov, 1), L - ov)]


def restore_chunk(model, seg: np.ndarray, cutoff_hz: float, steps=DEFAULT_STEPS, guidance=3.5, seed=42) -> np.ndarray:
    """The generated band above cutoff_hz for one mono chunk (same length as seg), level-matched to seg."""
    L = int(CHUNK * SR)
    if np.abs(seg).max() < 1e-4:                                       # silence: nothing to add
        return np.zeros(len(seg), dtype=np.float32)
    padded = np.zeros(L, dtype=np.float32)
    padded[:len(seg)] = seg
    y = _generate_chunk(model, padded, steps, guidance, seed)
    return _band_above(y, padded, cutoff_hz)[:len(seg)].astype(np.float32)


def upscale_channel(model, x: np.ndarray, cutoff_hz: float, steps=50, guidance=3.5, progress=None,
                    cancel=None, seed=42) -> np.ndarray:
    """Restore the band above cutoff_hz for one 48 kHz mono channel; returns original + restored band."""
    n = len(x)
    L = int(CHUNK * SR)
    ov = int(OVERLAP * SR)
    step = L - ov
    added = np.zeros(n, dtype=np.float32)
    fade = np.hanning(2 * ov).astype(np.float32)
    starts = list(range(0, max(n - ov, 1), step))
    for k, s in enumerate(starts):
        if cancel and cancel():
            raise InterruptedError("cancelled")
        seg = x[s:s + L]
        seg_len = len(seg)
        if np.abs(seg).max() < 1e-4:                                   # silence: nothing to add
            if progress:
                progress(k + 1, len(starts))
            continue
        padded = np.zeros(L, dtype=np.float32)
        padded[:seg_len] = seg
        y = _generate_chunk(model, padded, steps, guidance, seed + k)
        hb = _band_above(y, padded, cutoff_hz)[:seg_len]
        w = np.ones(seg_len, dtype=np.float32)
        if k > 0:
            w[:ov] = fade[:ov][: seg_len]
        if s + L < n:
            w[-ov:] = fade[ov:]
        added[s:s + seg_len] += hb * w
        if progress:
            progress(k + 1, len(starts))
    return x + added


def chunks_per_channel(n_samples: int) -> int:
    L, ov = int(CHUNK * SR), int(OVERLAP * SR)
    return len(range(0, max(n_samples - ov, 1), L - ov))

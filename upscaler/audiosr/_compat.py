"""Stand-ins for the librosa functions AudioSR uses, computed with torch (librosa 0.9.2 defaults)."""
import numpy as np
import torch

_WIN = {}


def _window(n):
    if n not in _WIN:
        _WIN[n] = torch.hann_window(n)        # periodic Hann = scipy get_window("hann", fftbins=True)
    return _WIN[n]


class librosa:  # noqa: N801  (namespace mimicking the module)
    @staticmethod
    def stft(y, n_fft=2048, hop_length=None, center=True):
        hop = hop_length or n_fft // 4
        t = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
        s = torch.stft(t, n_fft, hop_length=hop, win_length=n_fft, window=_window(n_fft), center=center,
                       pad_mode="reflect", return_complex=True)
        return s.numpy()

    @staticmethod
    def istft(stft_matrix, length=None, n_fft=None, hop_length=None):
        n = n_fft or 2 * (stft_matrix.shape[0] - 1)
        hop = hop_length or n // 4
        s = torch.from_numpy(np.ascontiguousarray(stft_matrix))
        return torch.istft(s, n, hop_length=hop, win_length=n, window=_window(n), center=True, length=length).numpy()

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import stft


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    fmax = float(fmax) if fmax is not None else float(sr) / 2.0
    n_freqs = n_fft // 2 + 1
    freqs = np.linspace(0.0, float(sr) / 2.0, n_freqs)

    m_min = _hz_to_mel(np.array([fmin], dtype=np.float64))[0]
    m_max = _hz_to_mel(np.array([fmax], dtype=np.float64))[0]
    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    hz_pts = _mel_to_hz(m_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center == left:
            center = min(left + 1, n_freqs - 1)
        if right == center:
            right = min(center + 1, n_freqs - 1)
        if right <= left:
            continue
        for k in range(left, center):
            fb[m - 1, k] = (k - left) / float(center - left)
        for k in range(center, right):
            fb[m - 1, k] = (right - k) / float(right - center)

    enorm = 2.0 / (hz_pts[2 : n_mels + 2] - hz_pts[:n_mels])
    fb *= enorm[:, None].astype(np.float32)
    return fb


@dataclass(frozen=True)
class LogMelStatsEmbedder:
    sample_rate: int = 16000
    n_mels: int = 64
    win_ms: float = 25.0
    hop_ms: float = 10.0
    max_audio_sec: float | None = None

    def dim(self) -> int:
        return int(self.n_mels) * 2

    def embed(self, y: np.ndarray, sr: int) -> np.ndarray:
        if sr != self.sample_rate:
            raise ValueError(f"Expected sr={self.sample_rate}, got {sr}")
        if y.size == 0:
            return np.zeros(self.dim(), dtype=np.float32)

        if self.max_audio_sec is not None:
            max_n = int(self.max_audio_sec * sr)
            if y.size > max_n:
                y = y[:max_n]

        win = max(16, int(sr * (self.win_ms / 1000.0)))
        hop = max(8, int(sr * (self.hop_ms / 1000.0)))
        noverlap = max(0, win - hop)
        n_fft = 1
        while n_fft < win:
            n_fft *= 2

        _, _, Zxx = stft(
            y,
            fs=sr,
            window="hann",
            nperseg=win,
            noverlap=noverlap,
            nfft=n_fft,
            boundary=None,
            padded=False,
        )
        power = (np.abs(Zxx) ** 2).astype(np.float32)
        fb = mel_filterbank(sr=sr, n_fft=n_fft, n_mels=int(self.n_mels))
        mel = fb @ power
        mel = np.log(mel + 1e-8)
        if mel.shape[1] == 0:
            mean = np.zeros(int(self.n_mels), dtype=np.float32)
            std = np.zeros(int(self.n_mels), dtype=np.float32)
        else:
            mean = mel.mean(axis=1).astype(np.float32)
            std = mel.std(axis=1).astype(np.float32)
        return np.concatenate([mean, std], axis=0)

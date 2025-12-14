from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


@dataclass(frozen=True)
class AudioData:
    y: np.ndarray  # float32 mono [-1, 1]
    sr: int


def _to_mono(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y
    if y.ndim == 2:
        return y.mean(axis=1)
    raise ValueError(f"Unexpected audio shape: {y.shape}")


def _read_with_ffmpeg(path: Path, target_sr: int) -> AudioData:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found; install ffmpeg or ensure soundfile can decode .ogg")
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(int(target_sr)),
        "pipe:1",
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {p.stderr.decode('utf-8', errors='ignore')[:300]}")
    y = np.frombuffer(p.stdout, dtype=np.float32)
    return AudioData(y=y, sr=int(target_sr))


def load_audio(path: str | Path, target_sr: int) -> AudioData:
    path = Path(path)
    try:
        y, sr = sf.read(str(path), dtype="float32", always_2d=True)
        y = _to_mono(y)
        sr = int(sr)
        if sr != target_sr:
            g = np.gcd(sr, target_sr)
            y = resample_poly(y, target_sr // g, sr // g).astype(np.float32, copy=False)
            sr = target_sr
        return AudioData(y=y, sr=sr)
    except Exception:
        return _read_with_ffmpeg(path, target_sr=target_sr)

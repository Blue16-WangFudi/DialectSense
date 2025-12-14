from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class PreprocessResult:
    y: np.ndarray  # float32 mono [-1, 1]
    sr: int
    rms_dbfs: float
    peak: float
    clipping_flag: int
    norm_gain_db: float


def _rms_dbfs(y: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(y), dtype=np.float64))) if y.size else 0.0
    return 20.0 * math.log10(max(1e-12, rms))


def _peak(y: np.ndarray) -> float:
    return float(np.max(np.abs(y))) if y.size else 0.0


def _run_ffmpeg_decode_trim(in_path: Path, out_wav_path: Path, cfg: dict[str, Any]) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg to enable robust .ogg decoding + silence trimming."
        )

    target_sr = int(cfg.get("target_sr", 16000))

    filters: list[str] = []
    trim_cfg = cfg.get("trim_silence", {}) if isinstance(cfg.get("trim_silence"), dict) else {}
    if bool(trim_cfg.get("enabled", True)):
        threshold_db = float(trim_cfg.get("threshold_db", -40.0))
        min_silence_sec = float(trim_cfg.get("min_silence_sec", 0.2))
        filters.append(
            "silenceremove="
            f"start_periods=1:start_duration={min_silence_sec}:start_threshold={threshold_db}dB:"
            f"stop_periods=1:stop_duration={min_silence_sec}:stop_threshold={threshold_db}dB"
        )

    cmd = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(in_path),
        "-ac",
        "1",
        "-ar",
        str(target_sr),
    ]
    if filters:
        cmd += ["-af", ",".join(filters)]
    cmd += ["-vn", "-sn", str(out_wav_path)]

    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if p.returncode != 0:
        err = p.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg failed for {in_path.name}: {err[:500]}")


def preprocess_audio(
    in_path: str | Path,
    out_wav_path: str | Path,
    audio_cfg: dict[str, Any],
) -> PreprocessResult:
    in_path = Path(in_path)
    out_wav_path = Path(out_wav_path)
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)

    tmp = out_wav_path.with_suffix(".tmp.wav")
    _run_ffmpeg_decode_trim(in_path=in_path, out_wav_path=tmp, cfg=audio_cfg)

    y, sr = sf.read(str(tmp), dtype="float32", always_2d=False)
    y = np.asarray(y, dtype=np.float32)
    sr = int(sr)
    if y.ndim != 1:
        y = y.reshape(-1).astype(np.float32, copy=False)

    peak0 = _peak(y)
    rms0 = _rms_dbfs(y)
    clipping0 = int(peak0 >= 0.999)

    norm_cfg = audio_cfg.get("normalize", {}) if isinstance(audio_cfg.get("normalize"), dict) else {}
    norm_enabled = bool(norm_cfg.get("enabled", True))
    norm_gain_db = 0.0
    if norm_enabled and y.size:
        target_rms_dbfs = float(norm_cfg.get("target_rms_dbfs", -20.0))
        max_gain_db = float(norm_cfg.get("max_gain_db", 20.0))
        peak_limit = float(norm_cfg.get("peak_limit", 0.99))

        gain_db = float(np.clip(target_rms_dbfs - rms0, -max_gain_db, max_gain_db))
        gain = float(10.0 ** (gain_db / 20.0))
        y2 = y * gain

        peak2 = _peak(y2)
        if peak2 > peak_limit and peak2 > 0:
            y2 = y2 * (peak_limit / peak2)
            peak2 = _peak(y2)

        y = y2.astype(np.float32, copy=False)
        norm_gain_db = gain_db

    peak1 = _peak(y)
    rms1 = _rms_dbfs(y)
    clipping1 = int(peak1 >= 0.999)

    # Overwrite with normalized audio for determinism.
    sf.write(str(out_wav_path), y, sr)
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass

    return PreprocessResult(
        y=y,
        sr=sr,
        rms_dbfs=rms1,
        peak=peak1,
        clipping_flag=max(clipping0, clipping1),
        norm_gain_db=float(norm_gain_db),
    )


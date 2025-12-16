from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .streaming import AudioStreamChunker
from .wavlm import WavLMEmbedder


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def _to_mono_f32(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if y.ndim == 2:
        # Accept both (samples, channels) and (channels, samples).
        if y.shape[0] <= 4 and y.shape[1] > y.shape[0]:
            y = y.mean(axis=0)
        else:
            y = y.mean(axis=1)
    if y.dtype.kind in {"i", "u"}:
        y = y.astype(np.float32) / max(1.0, float(np.iinfo(y.dtype).max))
    return np.asarray(y, dtype=np.float32).reshape(-1)


def _resample_poly(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if int(sr) == int(target_sr):
        return np.asarray(y, dtype=np.float32)
    from scipy.signal import resample_poly

    sr = int(sr)
    target_sr = int(target_sr)
    g = np.gcd(sr, target_sr)
    return resample_poly(y, target_sr // g, sr // g).astype(np.float32, copy=False)


def _predict_proba_with_classes(model: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    proba = model.predict_proba(X).astype(np.float64, copy=False)

    classes = None
    if hasattr(model, "classes_"):
        try:
            classes = np.asarray(getattr(model, "classes_"), dtype=int)
        except Exception:
            classes = None
    if classes is None and hasattr(model, "calibrator"):
        try:
            classes = np.asarray(model.calibrator.classes_, dtype=int)
        except Exception:
            classes = None
    if classes is None:
        classes = np.arange(proba.shape[1], dtype=int)
    return classes, proba


@dataclass
class CoarsePredictor:
    embed_cfg: dict[str, Any]
    model: Any
    classes: np.ndarray
    cluster_to_labels: dict[str, list[str]]
    target_sr: int = 16000
    _embedder: WavLMEmbedder | None = None

    @classmethod
    def from_artifacts(cls, artifact_dir: str | Path, cfg: dict[str, Any]) -> "CoarsePredictor":
        artifact_dir = Path(artifact_dir)
        model_path = artifact_dir / "models" / "coarse_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model: {model_path}")

        embed_cfg = cfg.get("embed", {}) if isinstance(cfg.get("embed"), dict) else {}

        bundle = joblib.load(model_path)
        model = bundle["model"]

        classes = None
        if isinstance(bundle, dict) and "classes" in bundle:
            try:
                classes = np.asarray(bundle["classes"], dtype=int)
            except Exception:
                classes = None
        if classes is None:
            try:
                classes = np.asarray(getattr(model, "classes_"), dtype=int)
            except Exception:
                classes = np.arange(int(bundle.get("n_classes", 0) or 0) or 0, dtype=int)
        if classes is None or classes.size == 0:
            classes = np.arange(0, 0, dtype=int)

        c2l_path = artifact_dir / "cluster_to_labels.json"
        cluster_to_labels: dict[str, list[str]] = {}
        if c2l_path.exists():
            cluster_to_labels = json.loads(c2l_path.read_text(encoding="utf-8"))

        return cls(embed_cfg=embed_cfg, model=model, classes=classes, cluster_to_labels=cluster_to_labels)

    @property
    def embedder(self) -> WavLMEmbedder:
        if self._embedder is None:
            self._embedder = WavLMEmbedder(self.embed_cfg)
        return self._embedder

    def preprocess_audio_array(self, audio: tuple[int, np.ndarray] | np.ndarray, sr: int | None = None) -> tuple[int, np.ndarray]:
        if isinstance(audio, tuple):
            sr, y = audio
        else:
            if sr is None:
                raise ValueError("sr is required when audio is a raw array")
            y = audio
        y = _to_mono_f32(y)
        y = _resample_poly(y, int(sr), int(self.target_sr))
        return int(self.target_sr), y

    def predict_chunk_proba(self, y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        sr, y = self.preprocess_audio_array(y, sr=sr)
        emb = self.embedder.embed_audio(y, sr=sr)
        classes, proba = _predict_proba_with_classes(self.model, emb.reshape(1, -1))
        return classes, proba[0]

    def stream_predict(
        self,
        chunker: AudioStreamChunker,
        y_new: np.ndarray,
        sr: int,
    ) -> list[tuple[float, np.ndarray, np.ndarray]]:
        sr_t, y_new = self.preprocess_audio_array(y_new, sr=sr)
        if int(sr_t) != int(chunker.sr):
            raise ValueError(f"Chunker sr={chunker.sr} does not match predictor sr={sr_t}")

        out: list[tuple[float, np.ndarray, np.ndarray]] = []
        for t_end, chunk in chunker.push(y_new):
            classes, proba = self.predict_chunk_proba(chunk, sr=chunker.sr)
            out.append((t_end, classes, proba))
        return out

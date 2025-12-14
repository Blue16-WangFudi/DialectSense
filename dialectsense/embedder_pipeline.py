from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .audio import load_audio


@dataclass(frozen=True)
class EmbedResult:
    embedding: np.ndarray
    rms: float
    duration_sec: float


def compute_embedding_for_path(embedder: Any, audio_path: str) -> EmbedResult:
    ad = load_audio(audio_path, target_sr=int(embedder.sample_rate))
    y = ad.y.astype(np.float32, copy=False)
    if y.size == 0:
        return EmbedResult(embedding=np.zeros(embedder.dim(), dtype=np.float32), rms=0.0, duration_sec=0.0)
    rms = float(np.sqrt(np.mean(np.square(y), dtype=np.float64)))
    emb = embedder.embed(y=y, sr=ad.sr).astype(np.float32, copy=False)
    return EmbedResult(embedding=emb, rms=rms, duration_sec=float(y.size) / float(ad.sr))


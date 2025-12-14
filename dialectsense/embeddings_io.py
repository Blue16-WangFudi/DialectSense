from __future__ import annotations

from pathlib import Path

import numpy as np


def embedding_path(artifact_dir: Path, clip_id: str) -> Path:
    return artifact_dir / "embeddings" / f"{clip_id}.npy"


def load_embeddings_matrix(artifact_dir: Path, clip_ids: list[str]) -> np.ndarray:
    if not clip_ids:
        return np.zeros((0, 0), dtype=np.float32)
    xs: list[np.ndarray] = []
    for cid in clip_ids:
        p = embedding_path(artifact_dir, cid)
        if not p.exists():
            raise FileNotFoundError(f"Missing embedding for clip_id={cid}: {p}")
        x = np.load(str(p))
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        xs.append(x)
    return np.stack(xs, axis=0).astype(np.float32, copy=False)


from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .util import stable_json_hash


@dataclass(frozen=True)
class EmbeddingCachePaths:
    cache_key: str
    cache_dir: Path

    def split_npz(self, split: str) -> Path:
        return self.cache_dir / f"{split}.npz"

    def meta_json(self) -> Path:
        return self.cache_dir / "meta.json"


def embedding_cache_paths(artifact_dir: Path, embed_cfg: dict[str, Any]) -> EmbeddingCachePaths:
    cache_key = stable_json_hash(embed_cfg)
    cache_dir = artifact_dir / "embeddings" / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    return EmbeddingCachePaths(cache_key=cache_key, cache_dir=cache_dir)


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(str(path), allow_pickle=True)
    return data["ids"], data["X"]


def save_npz(path: Path, ids: np.ndarray, X: np.ndarray) -> None:
    np.savez_compressed(str(path), ids=ids, X=X)


def write_meta(meta_path: Path, meta: dict[str, Any]) -> None:
    with meta_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

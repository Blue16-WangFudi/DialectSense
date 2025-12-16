from __future__ import annotations

from pathlib import Path

import numpy as np


def embedding_path(artifact_dir: Path, clip_id: str) -> Path:
    return artifact_dir / "embeddings" / f"{clip_id}.npy"


def _matrix_paths(artifact_dir: Path) -> tuple[Path, Path, Path]:
    emb_dir = artifact_dir / "embeddings"
    return emb_dir / "matrix.npy", emb_dir / "matrix_clip_ids.txt", emb_dir / "index.csv"


def ensure_embeddings_matrix_cache(artifact_dir: Path) -> tuple[Path, Path]:
    """
    Builds a contiguous float32 matrix cache for fast subset loading.

    Outputs:
      - artifacts/<run>/embeddings/matrix.npy            (N, D) float32
      - artifacts/<run>/embeddings/matrix_clip_ids.txt   (N lines, clip_id order)
    """
    matrix_path, ids_path, index_csv = _matrix_paths(artifact_dir)
    if matrix_path.exists() and ids_path.exists():
        return matrix_path, ids_path

    if not index_csv.exists():
        raise FileNotFoundError(f"Missing embeddings index: {index_csv}")

    # Read clip_id list from index.csv (header: clip_id,...)
    clip_ids: list[str] = []
    with index_csv.open("r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
        if not header:
            raise ValueError(f"Empty embeddings index: {index_csv}")
        for line in f:
            line = line.strip()
            if not line:
                continue
            cid = line.split(",", 1)[0].strip()
            if cid:
                clip_ids.append(cid)

    if not clip_ids:
        raise ValueError(f"No clip_ids found in {index_csv}")

    # Determine embedding dimension from the first file.
    first = np.load(str(embedding_path(artifact_dir, clip_ids[0])))
    d = int(np.asarray(first).reshape(-1).shape[0])
    n = int(len(clip_ids))

    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = matrix_path.with_suffix(".tmp.npy")
    mm = np.memmap(str(tmp), mode="w+", dtype=np.float32, shape=(n, d))
    mm[0] = np.asarray(first, dtype=np.float32).reshape(-1)
    for i, cid in enumerate(clip_ids[1:], start=1):
        x = np.load(str(embedding_path(artifact_dir, cid)))
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.shape[0] != d:
            raise ValueError(f"Embedding dim mismatch for clip_id={cid}: got {x.shape[0]}, expected {d}")
        mm[i] = x
        if i % 2000 == 0:
            mm.flush()
    mm.flush()
    del mm

    # Atomically publish.
    tmp.replace(matrix_path)
    ids_path.write_text("\n".join(clip_ids) + "\n", encoding="utf-8")
    return matrix_path, ids_path


def _load_matrix_cache(artifact_dir: Path) -> tuple[np.memmap, dict[str, int]]:
    matrix_path, ids_path, _ = _matrix_paths(artifact_dir)
    if not (matrix_path.exists() and ids_path.exists()):
        matrix_path, ids_path = ensure_embeddings_matrix_cache(artifact_dir)

    clip_ids = ids_path.read_text(encoding="utf-8").splitlines()
    id_to_row = {cid: i for i, cid in enumerate(clip_ids) if cid}

    # Discover shape from file size and embedding dim inferred from first row.
    # memmap can load without explicit shape if we know dim; infer dim by reading first embedding file.
    if not clip_ids:
        raise ValueError(f"Empty {ids_path}")
    d = int(np.load(str(embedding_path(artifact_dir, clip_ids[0]))).reshape(-1).shape[0])
    n = int(len(clip_ids))
    mm = np.memmap(str(matrix_path), mode="r", dtype=np.float32, shape=(n, d))
    return mm, id_to_row


def load_embeddings_matrix(artifact_dir: Path, clip_ids: list[str]) -> np.ndarray:
    if not clip_ids:
        return np.zeros((0, 0), dtype=np.float32)
    # Fast path: use contiguous matrix cache if present/creatable.
    try:
        mm, id_to_row = _load_matrix_cache(artifact_dir)
        idx = []
        for cid in clip_ids:
            if cid not in id_to_row:
                raise KeyError(cid)
            idx.append(id_to_row[cid])
        X = np.asarray(mm[np.asarray(idx, dtype=int)], dtype=np.float32)
        return X
    except Exception:
        # Fallback: per-file loading.
        xs: list[np.ndarray] = []
        for cid in clip_ids:
            p = embedding_path(artifact_dir, cid)
            if not p.exists():
                raise FileNotFoundError(f"Missing embedding for clip_id={cid}: {p}")
            x = np.load(str(p))
            x = np.asarray(x, dtype=np.float32).reshape(-1)
            xs.append(x)
        return np.stack(xs, axis=0).astype(np.float32, copy=False)

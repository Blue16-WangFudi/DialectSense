from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .audio_qc import read_audio_qc_csv
from .config import require
from .util import ensure_dir, stable_json_hash, write_json
from .wavlm import WavLMEmbedder


log = logging.getLogger("dialectsense")


def _write_embeddings_index(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run_embed(cfg: dict[str, Any], artifact_dir: Path) -> Path:
    embed_cfg = require(cfg, "embed")
    backend = str(embed_cfg.get("backend", "wavlm_large"))
    if backend != "wavlm_large":
        raise RuntimeError(f"Only WavLM-Large embeddings are supported (embed.backend must be 'wavlm_large'), got: {backend}")
    qc_rows = read_audio_qc_csv(artifact_dir / "audio_qc.csv")
    kept = [r for r in qc_rows if int(r.kept) == 1 and r.preprocessed_wav]

    out_dir = ensure_dir(artifact_dir / "embeddings")
    meta_path = out_dir / "meta.json"

    cache_key = stable_json_hash(
        {
            "model_id": embed_cfg.get("model_id", "microsoft/wavlm-large"),
            "layer_start": int(embed_cfg.get("layer_start", 6)),
            "layer_end": int(embed_cfg.get("layer_end", 12)),
            "chunk": embed_cfg.get("chunk", {}),
        }
    )

    if meta_path.exists():
        try:
            import json

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        if meta.get("cache_key") != cache_key:
            raise RuntimeError(
                f"Embedding cache mismatch for {artifact_dir}. "
                f"Expected cache_key={cache_key}, found {meta.get('cache_key')}. "
                "Delete artifacts/<run_name>/embeddings or run `make clean`."
            )

    embedder = WavLMEmbedder(embed_cfg)
    chunk_cfg = embed_cfg.get("chunk", {}) if isinstance(embed_cfg.get("chunk"), dict) else {}

    index_rows: list[dict[str, Any]] = []
    for i, r in enumerate(kept):
        out_path = out_dir / f"{r.clip_id}.npy"
        if out_path.exists():
            continue
        wav_path = Path(r.preprocessed_wav)
        if not wav_path.exists():
            log.warning("embed: missing preprocessed wav for %s: %s", r.clip_id, wav_path)
            continue
        res = embedder.embed_wav_path(wav_path, chunk_cfg=chunk_cfg)
        np.save(str(out_path), res.embedding.astype(np.float32, copy=False))
        index_rows.append(
            {
                "clip_id": r.clip_id,
                "embedding_path": str(out_path),
                "n_chunks": int(res.n_chunks),
                "effective_dur_sec": r.effective_dur_sec,
            }
        )
        if (i + 1) % 25 == 0 or (i + 1) == len(kept):
            log.info("embed: %d/%d", i + 1, len(kept))

    write_json(
        meta_path,
        {
            "cache_key": cache_key,
            "embed_cfg": embed_cfg,
            "dim": int(embedder.dim),
        },
    )
    _write_embeddings_index(out_dir / "index.csv", index_rows)
    log.info("embed: embeddings_dir=%s", out_dir)
    return out_dir

from __future__ import annotations

from typing import Any

from .logmel_stats import LogMelStatsEmbedder


def create_embedder(embed_cfg: dict[str, Any]) -> Any:
    backend = str(embed_cfg.get("backend", "logmel_stats"))
    if backend == "logmel_stats":
        return LogMelStatsEmbedder(
            sample_rate=int(embed_cfg.get("sample_rate", 16000)),
            n_mels=int(embed_cfg.get("n_mels", 64)),
            win_ms=float(embed_cfg.get("win_ms", 25.0)),
            hop_ms=float(embed_cfg.get("hop_ms", 10.0)),
            max_audio_sec=embed_cfg.get("max_audio_sec", None),
        )
    raise ValueError(f"Unknown embedding backend: {backend}")

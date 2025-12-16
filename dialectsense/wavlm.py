from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


log = logging.getLogger("dialectsense")


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = float(np.linalg.norm(x))
    if not math.isfinite(n) or n <= 0:
        return np.zeros_like(x)
    return (x / n).astype(np.float32, copy=False)


def _iter_chunk_ranges(n: int, sr: int, chunk_sec: float, hop_sec: float) -> list[tuple[int, int]]:
    chunk = max(1, int(round(float(chunk_sec) * sr)))
    hop = max(1, int(round(float(hop_sec) * sr)))
    if n <= chunk:
        return [(0, n)]
    out: list[tuple[int, int]] = []
    for start in range(0, max(1, n - 1), hop):
        end = min(n, start + chunk)
        out.append((start, end))
        if end >= n:
            break
    if out and out[-1][1] < n:
        out.append((max(0, n - chunk), n))
    return out


@dataclass(frozen=True)
class ChunkedEmbeddingResult:
    embedding: np.ndarray  # float32 [dim]
    n_chunks: int


@dataclass(frozen=True)
class ChunkEmbedding:
    start_sec: float
    end_sec: float
    embedding: np.ndarray  # float32 [dim]


class WavLMEmbedder:
    def __init__(self, embed_cfg: dict[str, Any]):
        self.embed_cfg = embed_cfg

        try:
            import torch
            from modelscope import snapshot_download
            from transformers import AutoFeatureExtractor, WavLMModel
        except Exception as e:
            raise RuntimeError(
                "Missing WavLM dependencies. Install: torch, transformers, modelscope."
            ) from e

        self._torch = torch
        self._snapshot_download = snapshot_download
        self._AutoFeatureExtractor = AutoFeatureExtractor
        self._WavLMModel = WavLMModel

        device_cfg = str(embed_cfg.get("device", "auto"))
        if device_cfg == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device_cfg

        self.model_id = str(embed_cfg.get("model_id", "microsoft/wavlm-large"))
        cache_dir = embed_cfg.get("model_cache_dir", ".cache/modelscope")
        self.model_dir = Path(self._snapshot_download(self.model_id, cache_dir=str(cache_dir)))

        self.fe = self._AutoFeatureExtractor.from_pretrained(str(self.model_dir))
        self.model = self._WavLMModel.from_pretrained(str(self.model_dir)).to(self.device)
        self.model.eval()

        self.layer_start = int(embed_cfg.get("layer_start", 6))
        self.layer_end = int(embed_cfg.get("layer_end", 12))
        if self.layer_end < self.layer_start:
            raise ValueError("embed.layer_end must be >= embed.layer_start")

    @property
    def dim(self) -> int:
        try:
            return int(self.model.config.hidden_size)
        except Exception:
            return 1024

    @property
    def sample_rate(self) -> int:
        return 16000
    
    def _embed_chunk(self, chunk: np.ndarray, sr: int) -> np.ndarray:
        torch = self._torch
        inputs = self.fe(chunk, sampling_rate=sr, return_tensors="pt")
        input_values = inputs["input_values"].to(self.device)
        attn = inputs.get("attention_mask")
        if attn is not None:
            attn = attn.to(self.device)

        out = self.model(input_values=input_values, attention_mask=attn, output_hidden_states=True)
        hs = out.hidden_states
        if hs is None:
            raise RuntimeError("WavLMModel did not return hidden_states")

        start = int(self.layer_start)
        end = int(self.layer_end)
        if end >= len(hs):
            raise ValueError(f"Requested layer_end={end}, but model has {len(hs)-1} layers")
        layers = hs[start : end + 1]
        x = torch.stack(layers, dim=0).mean(dim=0)  # [1, T, D]
        emb = x.mean(dim=1).squeeze(0)  # [D]
        return _l2_normalize(emb.detach().to("cpu").float().numpy())

    def embed_audio(self, y: np.ndarray, sr: int) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if y.size == 0:
            return np.zeros((self.dim,), dtype=np.float32)
        torch = self._torch
        with torch.inference_mode():
            return self._embed_chunk(y, sr=int(sr))

    def embed_wav_path_chunks(self, wav_path: str | Path, chunk_cfg: dict[str, Any]) -> list[ChunkEmbedding]:
        wav_path = Path(wav_path)
        y, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        sr = int(sr)
        if sr != 16000:
            raise ValueError(f"Expected 16kHz wav, got sr={sr} for {wav_path}")

        max_sec = float(chunk_cfg.get("max_sec", 12.0))
        strategy = str(chunk_cfg.get("strategy", "chunk"))
        chunk_sec = float(chunk_cfg.get("chunk_sec", 3.0))
        hop_sec = float(chunk_cfg.get("hop_sec", 1.5))
        max_chunks = chunk_cfg.get("max_chunks", 40)
        if max_chunks is not None:
            max_chunks = int(max_chunks)

        eff_sec = float(y.size) / float(sr) if y.size else 0.0

        ranges: list[tuple[int, int]] = [(0, y.size)]
        if eff_sec > max_sec:
            if strategy == "truncate":
                ranges = [(0, min(y.size, int(round(max_sec * sr))))]
            elif strategy == "chunk":
                ranges = _iter_chunk_ranges(y.size, sr=sr, chunk_sec=chunk_sec, hop_sec=hop_sec)
            else:
                raise ValueError("embed.chunk.strategy must be 'chunk' or 'truncate'")

        if max_chunks is not None and len(ranges) > max_chunks:
            idx = np.linspace(0, len(ranges) - 1, max_chunks).round().astype(int)
            ranges = [ranges[i] for i in idx.tolist()]

        out: list[ChunkEmbedding] = []
        torch = self._torch
        with torch.inference_mode():
            for (a, b) in ranges:
                chunk = y[a:b]
                emb = self._embed_chunk(chunk, sr=sr)
                out.append(ChunkEmbedding(start_sec=float(a) / float(sr), end_sec=float(b) / float(sr), embedding=emb))
        return out

    def embed_wav_path(self, wav_path: str | Path, chunk_cfg: dict[str, Any]) -> ChunkedEmbeddingResult:
        chunks = self.embed_wav_path_chunks(wav_path, chunk_cfg=chunk_cfg)
        if not chunks:
            return ChunkedEmbeddingResult(embedding=np.zeros((self.dim,), dtype=np.float32), n_chunks=0)

        chunk_arr = np.stack([c.embedding for c in chunks], axis=0).astype(np.float32, copy=False)
        agg = str(chunk_cfg.get("agg", "mean"))
        if agg == "mean" or chunk_arr.shape[0] == 1:
            emb_u = chunk_arr.mean(axis=0)
        elif agg == "attention":
            temp = float(chunk_cfg.get("attention_temperature", 0.5))
            temp = max(1e-3, temp)
            q = _l2_normalize(chunk_arr.mean(axis=0))
            k = chunk_arr / np.clip(np.linalg.norm(chunk_arr, axis=1, keepdims=True), 1e-12, None)
            score = (k @ q.reshape(-1, 1)).reshape(-1) / temp
            score = score - float(np.max(score))
            w = np.exp(score).astype(np.float64)
            w = w / float(np.sum(w))
            emb_u = (chunk_arr * w.reshape(-1, 1).astype(np.float32)).sum(axis=0)
        else:
            raise ValueError("embed.chunk.agg must be 'mean' or 'attention'")

        return ChunkedEmbeddingResult(embedding=_l2_normalize(emb_u), n_chunks=int(chunk_arr.shape[0]))

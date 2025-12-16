from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AudioStreamChunker:
    sr: int
    chunk_sec: float
    hop_sec: float
    max_buffer_sec: float = 60.0

    _buf: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32), repr=False)
    _offset: int = 0  # absolute sample index of _buf[0]
    _next_start: int = 0  # absolute sample index for next chunk start

    def reset(self) -> None:
        self._buf = np.zeros((0,), dtype=np.float32)
        self._offset = 0
        self._next_start = 0

    @property
    def chunk_n(self) -> int:
        return max(1, int(round(float(self.chunk_sec) * float(self.sr))))

    @property
    def hop_n(self) -> int:
        return max(1, int(round(float(self.hop_sec) * float(self.sr))))

    def push(self, y: np.ndarray) -> list[tuple[float, np.ndarray]]:
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if y.size == 0:
            return []

        self._buf = np.concatenate([self._buf, y], axis=0)

        out: list[tuple[float, np.ndarray]] = []
        end_available = self._offset + int(self._buf.size)
        n_chunk = self.chunk_n
        n_hop = self.hop_n

        while self._next_start + n_chunk <= end_available:
            a = self._next_start - self._offset
            b = a + n_chunk
            chunk = self._buf[a:b]
            t_end = float(self._next_start + n_chunk) / float(self.sr)
            out.append((t_end, chunk))
            self._next_start += n_hop

        # Drop samples before next_start (no longer needed).
        drop = self._next_start - self._offset
        if drop > 0:
            drop = min(drop, int(self._buf.size))
            self._buf = self._buf[drop:]
            self._offset += drop

        # Hard cap buffer to avoid runaway memory.
        max_buf = int(round(float(self.max_buffer_sec) * float(self.sr)))
        if max_buf > 0 and self._buf.size > max_buf:
            extra = int(self._buf.size - max_buf)
            self._buf = self._buf[extra:]
            self._offset += extra
            self._next_start = max(self._next_start, self._offset)

        return out


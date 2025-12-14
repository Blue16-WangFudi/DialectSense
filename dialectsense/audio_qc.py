from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AudioQCRow:
    clip_id: str
    label: str
    uploader_id: str
    audio_path: str
    preprocessed_wav: str
    orig_dur_sec: float | None
    orig_dur_source: str
    effective_dur_sec: float | None
    kept: int
    drop_reason: str
    sample_rate: int
    n_samples: int | None
    rms_dbfs: float | None
    peak: float | None
    clipping_flag: int
    norm_enabled: int
    norm_target_rms_dbfs: float | None
    norm_gain_db: float | None


_AUDIO_QC_FIELDS = [f.name for f in AudioQCRow.__dataclass_fields__.values()]


def write_audio_qc_csv(path: str | Path, rows: list[AudioQCRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_AUDIO_QC_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def read_audio_qc_csv(path: str | Path) -> list[AudioQCRow]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        dr = csv.DictReader(f)
        rows: list[AudioQCRow] = []
        for r in dr:
            if r is None:
                continue

            def _f(name: str) -> str:
                return (r.get(name) or "").strip()

            def _i(name: str) -> int:
                v = _f(name)
                return int(v) if v else 0

            def _fo(name: str) -> float | None:
                v = _f(name)
                return float(v) if v else None

            def _io(name: str) -> int | None:
                v = _f(name)
                return int(v) if v else None

            rows.append(
                AudioQCRow(
                    clip_id=_f("clip_id"),
                    label=_f("label"),
                    uploader_id=_f("uploader_id"),
                    audio_path=_f("audio_path"),
                    preprocessed_wav=_f("preprocessed_wav"),
                    orig_dur_sec=_fo("orig_dur_sec"),
                    orig_dur_source=_f("orig_dur_source"),
                    effective_dur_sec=_fo("effective_dur_sec"),
                    kept=_i("kept"),
                    drop_reason=_f("drop_reason"),
                    sample_rate=_i("sample_rate"),
                    n_samples=_io("n_samples"),
                    rms_dbfs=_fo("rms_dbfs"),
                    peak=_fo("peak"),
                    clipping_flag=_i("clipping_flag"),
                    norm_enabled=_i("norm_enabled"),
                    norm_target_rms_dbfs=_fo("norm_target_rms_dbfs"),
                    norm_gain_db=_fo("norm_gain_db"),
                )
            )
    return rows


def audio_qc_stats(rows: list[AudioQCRow]) -> dict[str, Any]:
    kept = [r for r in rows if int(r.kept) == 1]
    dropped = [r for r in rows if int(r.kept) != 1]
    by_reason: dict[str, int] = {}
    for r in dropped:
        by_reason[r.drop_reason] = by_reason.get(r.drop_reason, 0) + 1
    return {
        "n_total": int(len(rows)),
        "n_kept": int(len(kept)),
        "n_dropped": int(len(dropped)),
        "drop_reasons": dict(sorted(by_reason.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


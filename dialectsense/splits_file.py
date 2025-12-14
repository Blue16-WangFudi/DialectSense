from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SplitRow:
    clip_id: str
    split: str  # train/val/test
    label: str
    uploader_id: str
    audio_path: str
    preprocessed_wav: str
    effective_dur_sec: float | None
    rms_dbfs: float | None
    peak: float | None
    clipping_flag: int


def write_splits_csv(path: str | Path, rows: list[SplitRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "clip_id",
                "split",
                "label",
                "uploader_id",
                "audio_path",
                "preprocessed_wav",
                "effective_dur_sec",
                "rms_dbfs",
                "peak",
                "clipping_flag",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "clip_id": r.clip_id,
                    "split": r.split,
                    "label": r.label,
                    "uploader_id": r.uploader_id,
                    "audio_path": r.audio_path,
                    "preprocessed_wav": r.preprocessed_wav,
                    "effective_dur_sec": "" if r.effective_dur_sec is None else str(float(r.effective_dur_sec)),
                    "rms_dbfs": "" if r.rms_dbfs is None else str(float(r.rms_dbfs)),
                    "peak": "" if r.peak is None else str(float(r.peak)),
                    "clipping_flag": str(int(r.clipping_flag)),
                }
            )


def read_splits_csv(path: str | Path) -> list[SplitRow]:
    path = Path(path)
    out: list[SplitRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        dr = csv.DictReader(f)
        for r in dr:
            clip_id = (r.get("clip_id") or "").strip()
            split = (r.get("split") or "").strip()
            label = (r.get("label") or "").strip()
            uploader_id = (r.get("uploader_id") or "").strip()
            audio_path = (r.get("audio_path") or "").strip()
            preprocessed_wav = (r.get("preprocessed_wav") or "").strip()
            eff_raw = (r.get("effective_dur_sec") or "").strip()
            rms_raw = (r.get("rms_dbfs") or "").strip()
            peak_raw = (r.get("peak") or "").strip()
            clip_raw = (r.get("clipping_flag") or "").strip()
            out.append(
                SplitRow(
                    clip_id=clip_id,
                    split=split,
                    label=label,
                    uploader_id=uploader_id,
                    audio_path=audio_path,
                    preprocessed_wav=preprocessed_wav,
                    effective_dur_sec=float(eff_raw) if eff_raw else None,
                    rms_dbfs=float(rms_raw) if rms_raw else None,
                    peak=float(peak_raw) if peak_raw else None,
                    clipping_flag=int(float(clip_raw)) if clip_raw else 0,
                )
            )
    return out

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SplitRow:
    clip_id: str
    audio_path: str
    label: str
    group: str
    split: str  # train/val/test
    votes: int | None
    sound_length: float | None


def write_splits_csv(path: str | Path, rows: list[SplitRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "audio_path",
                "label",
                "group",
                "split",
                "votes",
                "sound_length",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "id": r.clip_id,
                    "audio_path": r.audio_path,
                    "label": r.label,
                    "group": r.group,
                    "split": r.split,
                    "votes": "" if r.votes is None else str(int(r.votes)),
                    "sound_length": "" if r.sound_length is None else str(float(r.sound_length)),
                }
            )


def read_splits_csv(path: str | Path) -> list[SplitRow]:
    path = Path(path)
    out: list[SplitRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        dr = csv.DictReader(f)
        for r in dr:
            clip_id = (r.get("id") or "").strip()
            audio_path = (r.get("audio_path") or "").strip()
            label = (r.get("label") or "").strip()
            group = (r.get("group") or "").strip()
            split = (r.get("split") or "").strip()
            votes_raw = (r.get("votes") or "").strip()
            sl_raw = (r.get("sound_length") or "").strip()
            votes = int(float(votes_raw)) if votes_raw else None
            sound_length = float(sl_raw) if sl_raw else None
            out.append(
                SplitRow(
                    clip_id=clip_id,
                    audio_path=audio_path,
                    label=label,
                    group=group,
                    split=split,
                    votes=votes,
                    sound_length=sound_length,
                )
            )
    return out


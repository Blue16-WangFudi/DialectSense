from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Row:
    clip_id: str
    audio_path: str
    label: str
    uploader_id: str
    votes: int | None
    sound_length_sec: float | None


def _to_int(v: str) -> int | None:
    v = (v or "").strip()
    if not v:
        return None
    try:
        return int(float(v))
    except ValueError:
        return None


def _to_float(v: str) -> float | None:
    v = (v or "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def read_metadata_rows(
    metadata_csv: str | Path,
    audio_dir: str | Path,
    id_col: str,
    label_col: str,
    uploader_col: str,
) -> tuple[list[Row], dict[str, Any]]:
    metadata_csv = Path(metadata_csv)
    audio_dir = Path(audio_dir)
    rows: list[Row] = []
    stats: dict[str, Any] = {
        "metadata_path": str(metadata_csv),
        "audio_dir": str(audio_dir),
        "n_metadata_rows": 0,
        "n_missing_id": 0,
        "n_missing_label": 0,
        "n_missing_uploader_id": 0,
        "n_missing_audio_file": 0,
    }

    with metadata_csv.open("r", encoding="utf-8", newline="") as f:
        dr = csv.DictReader(f)
        for r in dr:
            stats["n_metadata_rows"] += 1
            clip_id = (r.get(id_col) or "").strip()
            if not clip_id:
                stats["n_missing_id"] += 1
                continue
            label = (r.get(label_col) or "").strip()
            if not label:
                stats["n_missing_label"] += 1
                continue
            uploader_id = (r.get(uploader_col) or "").strip()
            if not uploader_id:
                stats["n_missing_uploader_id"] += 1
                continue

            audio_path = audio_dir / f"{clip_id}.ogg"
            if not audio_path.exists():
                stats["n_missing_audio_file"] += 1
                continue

            votes = _to_int(r.get("votes", ""))
            sound_length_sec = _to_float(r.get("sound_length", ""))
            rows.append(
                Row(
                    clip_id=clip_id,
                    audio_path=str(audio_path),
                    label=label,
                    uploader_id=uploader_id,
                    votes=votes,
                    sound_length_sec=sound_length_sec,
                )
            )

    return rows, stats


def filter_rows_by_votes(rows: Iterable[Row], min_votes: int | None) -> tuple[list[Row], dict[str, Any]]:
    out: list[Row] = []
    stats: dict[str, Any] = {
        "min_votes": min_votes,
        "n_in": 0,
        "n_out": 0,
        "n_drop_votes": 0,
    }

    for row in rows:
        stats["n_in"] += 1
        if min_votes is not None:
            v = row.votes if row.votes is not None else 0
            if v < int(min_votes):
                stats["n_drop_votes"] += 1
                continue
        out.append(row)
        stats["n_out"] += 1

    return out, stats


def counts_by_label(rows: Iterable[Row]) -> dict[str, int]:
    c: dict[str, int] = {}
    for r in rows:
        c[r.label] = c.get(r.label, 0) + 1
    return c


def subset_rows(
    rows: list[Row],
    seed: int,
    top_k_labels: int | None,
    max_per_label: int | None,
    max_total: int | None = None,
) -> tuple[list[Row], dict[str, Any]]:
    import random

    counts = counts_by_label(rows)
    labels_sorted = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    chosen_labels = [k for k, _ in labels_sorted]
    if top_k_labels is not None:
        chosen_labels = chosen_labels[: int(top_k_labels)]

    chosen_set = set(chosen_labels)
    kept = [r for r in rows if r.label in chosen_set]

    rng = random.Random(seed)
    by_label: dict[str, list[Row]] = {}
    for r in kept:
        by_label.setdefault(r.label, []).append(r)
    for lab in by_label:
        rng.shuffle(by_label[lab])

    out: list[Row] = []
    for lab in chosen_labels:
        items = by_label.get(lab, [])
        if max_per_label is not None:
            items = items[: int(max_per_label)]
        out.extend(items)

    if max_total is not None and len(out) > int(max_total):
        rng.shuffle(out)
        out = out[: int(max_total)]

    stats: dict[str, Any] = {
        "top_k_labels": top_k_labels,
        "max_per_label": max_per_label,
        "max_total": max_total,
        "n_in": len(rows),
        "n_after_label_filter": len(kept),
        "n_out": len(out),
        "chosen_labels": chosen_labels,
    }
    return out, stats


def label_group_counts(rows: list[Row]) -> tuple[dict[str, int], dict[str, int]]:
    counts: dict[str, int] = {}
    groups: dict[str, set[str]] = {}
    for r in rows:
        counts[r.label] = counts.get(r.label, 0) + 1
        groups.setdefault(r.label, set()).add(r.uploader_id)
    group_counts = {k: len(v) for k, v in groups.items()}
    for k in counts:
        group_counts.setdefault(k, 0)
    return counts, group_counts


def prune_labels_impossible_for_group_split(
    rows: list[Row],
    min_per_label_train: int,
    min_per_label_val: int,
    min_per_label_test: int,
    min_groups_per_label: int = 3,
) -> tuple[list[Row], dict[str, Any]]:
    required_total = int(min_per_label_train) + int(min_per_label_val) + int(min_per_label_test)
    counts, group_counts = label_group_counts(rows)

    bad: list[str] = []
    for lab, n in counts.items():
        g = group_counts.get(lab, 0)
        if n < required_total or g < int(min_groups_per_label):
            bad.append(lab)

    bad_set = set(bad)
    kept = [r for r in rows if r.label not in bad_set]
    stats: dict[str, Any] = {
        "min_per_label_train": int(min_per_label_train),
        "min_per_label_val": int(min_per_label_val),
        "min_per_label_test": int(min_per_label_test),
        "min_groups_per_label": int(min_groups_per_label),
        "required_total_samples_per_label": int(required_total),
        "n_in": int(len(rows)),
        "n_out": int(len(kept)),
        "n_labels_in": int(len(counts)),
        "n_labels_out": int(len(set(r.label for r in kept))),
        "dropped_labels": sorted(bad),
    }
    return kept, stats

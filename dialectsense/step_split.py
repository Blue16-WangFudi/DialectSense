from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .audio_qc import read_audio_qc_csv
from .config import require
from .data import Row, counts_by_label, prune_labels_impossible_for_group_split
from .split import make_group_splits
from .splits_file import SplitRow, write_splits_csv
from .util import write_json


log = logging.getLogger("dialectsense")


def run_split(cfg: dict[str, Any], artifact_dir: Path) -> Path:
    split_cfg = require(cfg, "split")

    qc = read_audio_qc_csv(artifact_dir / "audio_qc.csv")
    kept = [r for r in qc if int(r.kept) == 1]
    if not kept:
        raise RuntimeError("No kept samples found in audio_qc.csv")

    rows = [
        Row(
            clip_id=r.clip_id,
            audio_path=r.audio_path,
            label=r.label,
            uploader_id=r.uploader_id,
            votes=None,
            sound_length_sec=r.orig_dur_sec,
        )
        for r in kept
    ]

    rows_pruned, stats_prune = prune_labels_impossible_for_group_split(
        rows,
        min_per_label_train=int(split_cfg.get("min_per_label_train", 1)),
        min_per_label_val=int(split_cfg.get("min_per_label_val", 1)),
        min_per_label_test=int(split_cfg.get("min_per_label_test", 1)),
        min_groups_per_label=int(split_cfg.get("min_groups_per_label", 2)),
    )

    removed_labels_iter: list[str] = []
    split_res = None
    rows_for_split = rows_pruned
    for _ in range(int(split_cfg.get("max_label_drop_iterations", 200))):
        try:
            split_res = make_group_splits(
                rows=rows_for_split,
                seed=int(cfg.get("seed", 42)),
                train_ratio=float(split_cfg.get("train_ratio", 0.7)),
                val_ratio=float(split_cfg.get("val_ratio", 0.1)),
                test_ratio=float(split_cfg.get("test_ratio", 0.2)),
                max_tries=int(split_cfg.get("max_tries", 100)),
                min_per_label_train=int(split_cfg.get("min_per_label_train", 1)),
                min_per_label_val=int(split_cfg.get("min_per_label_val", 1)),
                min_per_label_test=int(split_cfg.get("min_per_label_test", 1)),
            )
            break
        except RuntimeError:
            counts_now = counts_by_label(rows_for_split)
            if not counts_now:
                raise
            smallest_label = min(counts_now.items(), key=lambda kv: (kv[1], kv[0]))[0]
            removed_labels_iter.append(smallest_label)
            rows_for_split = [r for r in rows_for_split if r.label != smallest_label]

    if split_res is None:
        raise RuntimeError(
            "Failed to find a speaker-disjoint split after dropping rare labels. "
            "Try lowering split.min_per_label_* / split.min_groups_per_label, or increasing split.max_tries."
        )

    qc_by_id = {r.clip_id: r for r in kept}
    split_rows: list[SplitRow] = []
    for r in rows_for_split:
        q = qc_by_id[r.clip_id]
        split_rows.append(
            SplitRow(
                clip_id=r.clip_id,
                split=split_res.split_by_id[r.clip_id],
                label=r.label,
                uploader_id=r.uploader_id,
                audio_path=r.audio_path,
                preprocessed_wav=q.preprocessed_wav,
                effective_dur_sec=q.effective_dur_sec,
                rms_dbfs=q.rms_dbfs,
                peak=q.peak,
                clipping_flag=int(q.clipping_flag),
            )
        )

    out_path = artifact_dir / "splits.csv"
    write_splits_csv(out_path, split_rows)
    write_json(
        artifact_dir / "split_stats.json",
        {
            "prune_impossible_for_split": stats_prune,
            "prune_iterative_removed_labels": removed_labels_iter,
            "split": {
                "used_seed": split_res.used_seed,
                "tries": split_res.tries,
                "counts": split_res.counts,
            },
        },
    )
    log.info("split: wrote %s", out_path)
    return out_path


from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from .data import Row


@dataclass(frozen=True)
class SplitResult:
    split_by_id: dict[str, str]  # clip_id -> train/val/test
    used_seed: int
    tries: int
    counts: dict[str, dict[str, int]]  # split -> label -> count


def _label_counts(rows: Iterable[Row]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r.label] = out.get(r.label, 0) + 1
    return out


def _meets_min(counts: dict[str, int], min_per_label: int) -> bool:
    return all(v >= min_per_label for v in counts.values()) if counts else False


def make_group_splits(
    rows: list[Row],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    max_tries: int,
    min_per_label_train: int,
    min_per_label_val: int,
    min_per_label_test: int,
) -> SplitResult:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    clip_ids = np.array([r.clip_id for r in rows])
    labels = np.array([r.label for r in rows])
    groups = np.array([r.uploader_id for r in rows])
    unique_labels = sorted(set(labels.tolist()))

    def split_ok(mask_train: np.ndarray, mask_val: np.ndarray, mask_test: np.ndarray) -> bool:
        train_counts = _label_counts([rows[i] for i in np.where(mask_train)[0]])
        val_counts = _label_counts([rows[i] for i in np.where(mask_val)[0]])
        test_counts = _label_counts([rows[i] for i in np.where(mask_test)[0]])
        for lab in unique_labels:
            train_counts.setdefault(lab, 0)
            val_counts.setdefault(lab, 0)
            test_counts.setdefault(lab, 0)
        return (
            _meets_min(train_counts, min_per_label_train)
            and _meets_min(val_counts, min_per_label_val)
            and _meets_min(test_counts, min_per_label_test)
        )

    for t in range(max_tries):
        used_seed = seed + t
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=used_seed)
        (trainval_idx, test_idx) = next(gss_test.split(clip_ids, labels, groups))

        trainval_ids = clip_ids[trainval_idx]
        trainval_labels = labels[trainval_idx]
        trainval_groups = groups[trainval_idx]

        val_size_rel = val_ratio / (train_ratio + val_ratio)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_rel, random_state=used_seed + 10_000)
        (train_idx_rel, val_idx_rel) = next(gss_val.split(trainval_ids, trainval_labels, trainval_groups))

        train_idx = trainval_idx[train_idx_rel]
        val_idx = trainval_idx[val_idx_rel]

        mask_train = np.zeros(len(rows), dtype=bool)
        mask_val = np.zeros(len(rows), dtype=bool)
        mask_test = np.zeros(len(rows), dtype=bool)
        mask_train[train_idx] = True
        mask_val[val_idx] = True
        mask_test[test_idx] = True

        if split_ok(mask_train, mask_val, mask_test):
            split_by_id = {}
            for i, cid in enumerate(clip_ids):
                if mask_train[i]:
                    split_by_id[str(cid)] = "train"
                elif mask_val[i]:
                    split_by_id[str(cid)] = "val"
                else:
                    split_by_id[str(cid)] = "test"

            counts = {
                "train": _label_counts([rows[i] for i in np.where(mask_train)[0]]),
                "val": _label_counts([rows[i] for i in np.where(mask_val)[0]]),
                "test": _label_counts([rows[i] for i in np.where(mask_test)[0]]),
            }
            for split in counts:
                for lab in unique_labels:
                    counts[split].setdefault(lab, 0)

            return SplitResult(split_by_id=split_by_id, used_seed=used_seed, tries=t + 1, counts=counts)

    raise RuntimeError(
        "Failed to find a group-disjoint split meeting per-label minimums. "
        "Try lowering min_per_label_* or increasing max_tries."
    )

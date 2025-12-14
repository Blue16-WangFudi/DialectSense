from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .cache import embedding_cache_paths, load_npz, save_npz, write_meta
from .cluster import project_2d_and_save
from .config import load_config, require, resolve_paths, save_resolved_config
from .data import (
    counts_by_label,
    filter_rows,
    prune_labels_impossible_for_group_split,
    read_metadata_rows,
    subset_rows,
)
from .embed.registry import create_embedder
from .embedder_pipeline import compute_embedding_for_path
from .eval import evaluate_and_save
from .splits_file import SplitRow, read_splits_csv, write_splits_csv
from .split import make_group_splits
from .train import load_model, train_and_save
from .util import ensure_dir, set_global_seed
from .viz import plot_label_distribution


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dialectsense")


def _artifact_dir(cfg: dict[str, Any]) -> Path:
    paths = resolve_paths(cfg)
    return paths.artifact_dir


def cmd_preprocess(cfg: dict[str, Any]) -> None:
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    artifact_dir = _artifact_dir(cfg)
    ensure_dir(artifact_dir)
    save_resolved_config(cfg, artifact_dir)

    data_cfg = require(cfg, "data")
    filter_cfg = require(cfg, "filter")
    subset_cfg = require(cfg, "subset")
    split_cfg = require(cfg, "split")

    rows_raw, stats_read = read_metadata_rows(
        metadata_csv=data_cfg["metadata_csv"],
        audio_dir=data_cfg["audio_dir"],
        id_col=data_cfg["id_col"],
        label_col=data_cfg["label_col"],
        group_col=data_cfg["group_col"],
    )
    counts_raw = counts_by_label(rows_raw)

    rows_filt, stats_filter = filter_rows(
        rows_raw,
        min_sound_length_sec=filter_cfg.get("min_sound_length_sec"),
        max_sound_length_sec=filter_cfg.get("max_sound_length_sec"),
        min_votes=filter_cfg.get("min_votes"),
    )
    counts_filt = counts_by_label(rows_filt)

    rows_sub, stats_subset = subset_rows(
        rows_filt,
        seed=seed,
        top_k_labels=subset_cfg.get("top_k_labels"),
        max_per_label=subset_cfg.get("max_per_label"),
    )
    counts_sub = counts_by_label(rows_sub)

    rows_pruned, stats_prune = prune_labels_impossible_for_group_split(
        rows_sub,
        min_per_label_train=int(split_cfg.get("min_per_label_train", 1)),
        min_per_label_val=int(split_cfg.get("min_per_label_val", 1)),
        min_per_label_test=int(split_cfg.get("min_per_label_test", 1)),
        min_groups_per_label=int(split_cfg.get("min_groups_per_label", 3)),
    )
    counts_pruned = counts_by_label(rows_pruned)

    removed_labels_iter: list[str] = []
    split_res = None
    rows_for_split = rows_pruned
    for _ in range(int(split_cfg.get("max_label_drop_iterations", 200))):
        try:
            split_res = make_group_splits(
                rows=rows_for_split,
                seed=seed,
                train_ratio=float(split_cfg["train_ratio"]),
                val_ratio=float(split_cfg["val_ratio"]),
                test_ratio=float(split_cfg["test_ratio"]),
                max_tries=int(split_cfg.get("max_tries", 50)),
                min_per_label_train=int(split_cfg.get("min_per_label_train", 1)),
                min_per_label_val=int(split_cfg.get("min_per_label_val", 1)),
                min_per_label_test=int(split_cfg.get("min_per_label_test", 1)),
            )
            break
        except RuntimeError:
            counts_now = counts_by_label(rows_for_split)
            if not counts_now:
                raise
            # Drop the smallest remaining label and retry.
            smallest_label = min(counts_now.items(), key=lambda kv: (kv[1], kv[0]))[0]
            removed_labels_iter.append(smallest_label)
            rows_for_split = [r for r in rows_for_split if r.label != smallest_label]

    if split_res is None:
        raise RuntimeError(
            "Failed to find a group-disjoint split even after dropping rare labels. "
            "Try lowering min_per_label_* or increasing max_tries."
        )

    split_rows: list[SplitRow] = []
    for r in rows_for_split:
        split_rows.append(
            SplitRow(
                clip_id=r.clip_id,
                audio_path=r.audio_path,
                label=r.label,
                group=r.group,
                split=split_res.split_by_id[r.clip_id],
                votes=r.votes,
                sound_length=r.sound_length,
            )
        )

    write_splits_csv(artifact_dir / "splits.csv", split_rows)

    figs_dir = ensure_dir(artifact_dir / "figs")
    plot_label_distribution(
        out_path=figs_dir / "class_distribution.png",
        counts_before=counts_raw,
        counts_after=counts_filt,
        title="Province distribution (before/after filtering)",
        max_labels=50,
    )

    stats = {
        "read": stats_read,
        "filter": stats_filter,
        "subset": stats_subset,
        "prune_impossible_for_split": stats_prune,
        "prune_iterative_removed_labels": removed_labels_iter,
        "counts_raw": counts_raw,
        "counts_filtered": counts_filt,
        "counts_subset": counts_sub,
        "counts_pruned_for_split": counts_pruned,
        "split": {
            "used_seed": split_res.used_seed,
            "tries": split_res.tries,
            "counts": split_res.counts,
        },
    }
    (artifact_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    log.info("preprocess: wrote %s", artifact_dir / "splits.csv")
    log.info(
        "preprocess: subset rows=%d labels=%d -> pruned rows=%d labels=%d (dropped=%d)",
        len(rows_sub),
        len(counts_sub),
        len(rows_for_split),
        len(set(r.label for r in rows_for_split)),
        len(stats_prune.get("dropped_labels", [])) + len(removed_labels_iter),
    )


def cmd_embed(cfg: dict[str, Any]) -> None:
    artifact_dir = _artifact_dir(cfg)
    splits = read_splits_csv(artifact_dir / "splits.csv")
    embed_cfg = require(cfg, "embed")

    embedder = create_embedder(embed_cfg)
    cache = embedding_cache_paths(artifact_dir, embed_cfg)
    write_meta(
        cache.meta_json(),
        {
            "cache_key": cache.cache_key,
            "embed_cfg": embed_cfg,
            "dim": int(embedder.dim()),
        },
    )

    by_split: dict[str, list[SplitRow]] = {"train": [], "val": [], "test": []}
    for r in splits:
        by_split.setdefault(r.split, []).append(r)

    for split, items in by_split.items():
        if not items:
            continue
        out_npz = cache.split_npz(split)
        ids_expected = np.array([r.clip_id for r in items], dtype=str)

        if out_npz.exists():
            ids_cached, X_cached = load_npz(out_npz)
            if ids_cached.shape == ids_expected.shape and np.all(ids_cached == ids_expected) and X_cached.shape[1] == embedder.dim():
                log.info("embed: cache hit %s (%d, %d)", out_npz, X_cached.shape[0], X_cached.shape[1])
                continue

        X = np.zeros((len(items), embedder.dim()), dtype=np.float32)
        for i, r in enumerate(items):
            res = compute_embedding_for_path(embedder, r.audio_path)
            X[i] = res.embedding
            if (i + 1) % 50 == 0 or (i + 1) == len(items):
                log.info("embed: %s %d/%d", split, i + 1, len(items))

        save_npz(out_npz, ids=ids_expected, X=X)
        log.info("embed: wrote %s", out_npz)


def _labels_for_ids(
    splits: list[SplitRow], split: str, ids: np.ndarray
) -> tuple[list[str], np.ndarray | None]:
    by_id: dict[str, SplitRow] = {r.clip_id: r for r in splits if r.split == split}
    y: list[str] = []
    votes: list[int] = []
    has_votes = True
    for cid in ids.tolist():
        r = by_id.get(str(cid))
        if r is None:
            raise RuntimeError(f"Missing id={cid} in splits.csv for split={split}")
        y.append(r.label)
        if r.votes is None:
            has_votes = False
            votes.append(0)
        else:
            votes.append(int(r.votes))
    v = np.array(votes, dtype=np.int64) if has_votes else None
    return y, v


def cmd_train(cfg: dict[str, Any]) -> None:
    artifact_dir = _artifact_dir(cfg)
    splits = read_splits_csv(artifact_dir / "splits.csv")
    embed_cfg = require(cfg, "embed")
    model_cfg = require(cfg, "model")
    cache = embedding_cache_paths(artifact_dir, embed_cfg)

    ids_tr, X_tr = load_npz(cache.split_npz("train"))
    ids_va, X_va = load_npz(cache.split_npz("val"))

    y_train, votes_train = _labels_for_ids(splits, "train", ids_tr)
    y_val, _ = _labels_for_ids(splits, "val", ids_va)

    out = train_and_save(
        artifact_dir=artifact_dir,
        model_cfg=model_cfg,
        X_train=X_tr,
        y_train=y_train,
        votes_train=votes_train,
        X_val=X_va,
        y_val=y_val,
    )
    log.info("train: wrote %s", out)


def cmd_eval(cfg: dict[str, Any]) -> None:
    artifact_dir = _artifact_dir(cfg)
    splits = read_splits_csv(artifact_dir / "splits.csv")
    embed_cfg = require(cfg, "embed")
    cache = embedding_cache_paths(artifact_dir, embed_cfg)

    ids_te, X_te = load_npz(cache.split_npz("test"))
    y_test, _ = _labels_for_ids(splits, "test", ids_te)

    model_bundle = load_model(artifact_dir / "models" / "svm.joblib")
    metrics = evaluate_and_save(artifact_dir=artifact_dir, model_bundle=model_bundle, X_test=X_te, y_test=y_test)
    log.info("eval: macro_f1=%.4f balanced_acc=%.4f", metrics["macro_f1"], metrics["balanced_accuracy"])


def cmd_cluster(cfg: dict[str, Any]) -> None:
    artifact_dir = _artifact_dir(cfg)
    splits = read_splits_csv(artifact_dir / "splits.csv")
    embed_cfg = require(cfg, "embed")
    cluster_cfg = require(cfg, "cluster")
    cache = embedding_cache_paths(artifact_dir, embed_cfg)

    Xs = []
    ys: list[str] = []
    for split in ["train", "val", "test"]:
        ids, X = load_npz(cache.split_npz(split))
        y_split, _ = _labels_for_ids(splits, split, ids)
        Xs.append(X)
        ys.extend(y_split)
    X_all = np.concatenate(Xs, axis=0)

    info = project_2d_and_save(artifact_dir=artifact_dir, X=X_all, y=ys, cluster_cfg=cluster_cfg)
    (artifact_dir / "reports" / "cluster.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    log.info("cluster: %s", info)


def cmd_ui(cfg: dict[str, Any]) -> None:
    from .ui.app import launch

    launch(cfg)


def main() -> None:
    ap = argparse.ArgumentParser(prog="dialectsense")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["preprocess", "embed", "train", "eval", "cluster", "ui"]:
        sp = sub.add_parser(name)
        sp.add_argument("--config", required=True, help="Path to JSON config (e.g., configs/smoke.json)")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.cmd == "preprocess":
        cmd_preprocess(cfg)
    elif args.cmd == "embed":
        cmd_embed(cfg)
    elif args.cmd == "train":
        cmd_train(cfg)
    elif args.cmd == "eval":
        cmd_eval(cfg)
    elif args.cmd == "cluster":
        cmd_cluster(cfg)
    elif args.cmd == "ui":
        cmd_ui(cfg)
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()

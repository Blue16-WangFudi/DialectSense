from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from .config import require
from .embeddings_io import load_embeddings_matrix
from .splits_file import read_splits_csv
from .util import ensure_dir, write_json


log = logging.getLogger("dialectsense")


def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(n, 1e-12, None)


def _centroid_mean(X: np.ndarray) -> np.ndarray:
    return X.mean(axis=0)


def _centroid_trimmed_mean(X: np.ndarray, trim_fraction: float) -> np.ndarray:
    if X.shape[0] <= 2:
        return _centroid_mean(X)
    trim_fraction = float(np.clip(trim_fraction, 0.0, 0.45))
    m = _centroid_mean(X)
    m = m / np.clip(np.linalg.norm(m), 1e-12, None)
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    dist = 1.0 - (Xn @ m.reshape(-1, 1)).reshape(-1)
    keep_n = max(2, int(round((1.0 - 2.0 * trim_fraction) * X.shape[0])))
    idx = np.argsort(dist)[:keep_n]
    return X[idx].mean(axis=0)


def _centroid_qc_weighted_mean(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    w = np.clip(w, 0.0, None)
    if not np.any(w > 0):
        return _centroid_mean(X)
    w = w / float(np.sum(w))
    return (X.astype(np.float64) * w.reshape(-1, 1)).sum(axis=0).astype(np.float32)


def run_coarsen(cfg: dict[str, Any], artifact_dir: Path) -> Path:
    coarse_cfg = require(cfg, "coarse")
    if not bool(coarse_cfg.get("enabled", True)):
        raise RuntimeError("coarse.enabled is false; coarse-only training requires coarsening enabled.")

    K = int(coarse_cfg.get("k", 3))
    split_rows = read_splits_csv(artifact_dir / "splits.csv")
    train_rows = [r for r in split_rows if r.split == "train"]
    if not train_rows:
        raise RuntimeError("splits.csv has no train rows")

    by_label: dict[str, list[str]] = defaultdict(list)
    by_label_w: dict[str, list[float]] = defaultdict(list)
    for r in train_rows:
        by_label[r.label].append(r.clip_id)
        w = float(r.effective_dur_sec) if r.effective_dur_sec is not None else 1.0
        if int(r.clipping_flag) == 1:
            w *= 0.5
        by_label_w[r.label].append(w)

    labels = sorted(by_label.keys())
    if len(labels) < K:
        raise RuntimeError(f"Need at least {K} labels in train to build {K} clusters; got {len(labels)}")

    centroid_cfg = coarse_cfg.get("centroid", {}) if isinstance(coarse_cfg.get("centroid"), dict) else {}
    centroid_method = str(centroid_cfg.get("method", "mean"))
    trim_fraction = float(centroid_cfg.get("trim_fraction", 0.1))

    label_centroids: list[np.ndarray] = []
    label_counts: dict[str, int] = {}
    for lab in labels:
        ids = by_label[lab]
        X = load_embeddings_matrix(artifact_dir, ids)
        if centroid_method == "mean":
            c = _centroid_mean(X)
        elif centroid_method == "trimmed_mean":
            c = _centroid_trimmed_mean(X, trim_fraction=trim_fraction)
        elif centroid_method == "qc_weighted_mean":
            w = np.asarray(by_label_w[lab], dtype=np.float64)
            c = _centroid_qc_weighted_mean(X, w=w)
        else:
            raise ValueError("coarse.centroid.method must be one of: mean, trimmed_mean, qc_weighted_mean")
        c = c.astype(np.float32, copy=False)
        c = c / np.clip(np.linalg.norm(c), 1e-12, None)
        label_centroids.append(c)
        label_counts[lab] = int(len(ids))

    C = np.stack(label_centroids, axis=0).astype(np.float32, copy=False)
    C = _l2_normalize_rows(C)

    kmeans_cfg = coarse_cfg.get("kmeans", {}) if isinstance(coarse_cfg.get("kmeans"), dict) else {}
    n_init = int(max(20, int(kmeans_cfg.get("n_init", 20))))
    km = KMeans(n_clusters=K, n_init=n_init, random_state=int(cfg.get("seed", 42)))
    cluster_ids = km.fit_predict(C).astype(int)

    label_to_cluster = {lab: int(cid) for lab, cid in zip(labels, cluster_ids.tolist(), strict=True)}
    cluster_to_labels: dict[str, list[str]] = defaultdict(list)
    for lab, cid in label_to_cluster.items():
        cluster_to_labels[str(int(cid))].append(lab)
    for cid in cluster_to_labels:
        cluster_to_labels[cid] = sorted(cluster_to_labels[cid])

    out_dir = ensure_dir(artifact_dir)
    write_json(out_dir / "label_to_cluster.json", label_to_cluster)
    write_json(out_dir / "cluster_to_labels.json", dict(sorted(cluster_to_labels.items(), key=lambda kv: int(kv[0]))))

    cluster_centroids = km.cluster_centers_.astype(np.float32, copy=False)
    cluster_centroids = _l2_normalize_rows(cluster_centroids)
    np.save(str(out_dir / "cluster_centroids.npy"), cluster_centroids)

    md_lines: list[str] = []
    md_lines.append("# Coarse clusters\n")
    md_lines.append(f"- K: {K}\n")
    md_lines.append(f"- centroid_method: `{centroid_method}`\n")
    md_lines.append("\n## Clusters\n")
    for cid_str, labs in sorted(cluster_to_labels.items(), key=lambda kv: int(kv[0])):
        md_lines.append(f"### Cluster {cid_str}\n")
        md_lines.append(f"- labels ({len(labs)}): {', '.join(labs)}\n")
        md_lines.append(f"- train_samples: {sum(label_counts.get(l, 0) for l in labs)}\n\n")
    (out_dir / "cluster_summary.md").write_text("".join(md_lines), encoding="utf-8")

    log.info("coarsen: wrote label_to_cluster.json / cluster_centroids.npy")
    return out_dir / "label_to_cluster.json"


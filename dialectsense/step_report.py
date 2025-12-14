from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .audio_qc import read_audio_qc_csv
from .embeddings_io import load_embeddings_matrix
from .splits_file import read_splits_csv
from .util import ensure_dir, write_json
from .viz import (
    plot_bar_counts,
    plot_cosine_distance_heatmap,
    plot_duration_hist,
    plot_embedding_2d,
    plot_f1_by_class,
    plot_top_confusions_bar,
)


log = logging.getLogger("dialectsense")


def _try_umap():
    try:
        import umap  # type: ignore

        return umap
    except Exception:
        return None


def _project_train_and_transform(
    X_train: np.ndarray,
    X_other: np.ndarray | None,
    seed: int,
    method: str,
    perplexity: int,
) -> tuple[str, np.ndarray, np.ndarray | None]:
    if "umap" in method:
        umap_mod = _try_umap()
        if umap_mod is not None:
            proj = umap_mod.UMAP(
                n_components=2,
                random_state=seed,
                n_neighbors=15,
                min_dist=0.1,
                metric="cosine",
            )
            X2_train = proj.fit_transform(X_train)
            X2_other = proj.transform(X_other) if X_other is not None else None
            return "umap", X2_train.astype(np.float32), None if X2_other is None else X2_other.astype(np.float32)

    from sklearn.manifold import TSNE

    X2_train = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(X_train)
    X2_other = None
    if X_other is not None:
        X2_other = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed + 1).fit_transform(X_other)
    return "tsne", X2_train.astype(np.float32), None if X2_other is None else X2_other.astype(np.float32)


def run_report(cfg: dict[str, Any], artifact_dir: Path) -> Path:
    seed = int(cfg.get("seed", 42))
    figures_dir = ensure_dir(artifact_dir / "figures")

    qc = read_audio_qc_csv(artifact_dir / "audio_qc.csv")
    kept_qc = [r for r in qc if int(r.kept) == 1]

    splits = read_splits_csv(artifact_dir / "splits.csv")
    label_to_cluster = json.loads((artifact_dir / "label_to_cluster.json").read_text(encoding="utf-8"))

    # 1) Data distribution
    label_counts = Counter([r.label for r in kept_qc])
    plot_bar_counts(
        out_path=figures_dir / "label_counts.png",
        counts=dict(label_counts),
        title="Original label counts (kept after audio QC)",
        max_labels=int(cfg.get("report", {}).get("max_labels_bar", 30)),
    )

    cluster_counts = Counter([str(label_to_cluster.get(r.label, "unknown")) for r in splits])
    plot_bar_counts(
        out_path=figures_dir / "cluster_counts.png",
        counts=dict(cluster_counts),
        title="Coarse cluster counts (splits.csv)",
        max_labels=50,
        other_label="",
    )

    speakers_per_split: dict[str, int] = {}
    for split in ["train", "val", "test"]:
        speakers_per_split[split] = len(set(r.uploader_id for r in splits if r.split == split))
    plot_bar_counts(
        out_path=figures_dir / "speakers_per_split.png",
        counts=speakers_per_split,
        title="Speakers per split (uploader_id)",
        max_labels=10,
        other_label="",
    )

    orig_durs = [r.orig_dur_sec for r in kept_qc if r.orig_dur_sec is not None]
    eff_durs = [r.effective_dur_sec for r in kept_qc if r.effective_dur_sec is not None]
    plot_duration_hist(
        out_path=figures_dir / "duration_hist_orig.png",
        durations_sec=orig_durs,
        title="Original duration histogram (kept)",
    )
    plot_duration_hist(
        out_path=figures_dir / "duration_hist_effective.png",
        durations_sec=eff_durs,
        title="Effective duration histogram (kept, after trim)",
    )

    # 2) Embedding space (train/test)
    report_cfg = cfg.get("report", {}) if isinstance(cfg.get("report"), dict) else {}
    emb2_cfg = report_cfg.get("embedding_2d", {}) if isinstance(report_cfg.get("embedding_2d"), dict) else {}
    method = str(emb2_cfg.get("method", "umap_if_available_else_tsne"))
    max_points = int(emb2_cfg.get("max_points", 3000))
    perplexity = int(emb2_cfg.get("perplexity", 30))

    def _sample(rows: list[Any]) -> list[Any]:
        if len(rows) <= max_points:
            return rows
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(rows), size=max_points, replace=False)
        idx.sort()
        return [rows[i] for i in idx.tolist()]

    train_rows = _sample([r for r in splits if r.split == "train"])
    test_rows = _sample([r for r in splits if r.split == "test"])

    def _cluster_label(r) -> str:
        return str(int(label_to_cluster.get(r.label, -1)))

    X_train = load_embeddings_matrix(artifact_dir, [r.clip_id for r in train_rows])
    X_test = load_embeddings_matrix(artifact_dir, [r.clip_id for r in test_rows]) if test_rows else None
    used, X2_train, X2_test = _project_train_and_transform(
        X_train=X_train,
        X_other=X_test,
        seed=seed,
        method=method,
        perplexity=perplexity,
    )

    plot_embedding_2d(
        out_path=figures_dir / "embedding_2d_train.png",
        X2=X2_train,
        labels=[_cluster_label(r) for r in train_rows],
        title=f"Train embeddings 2D ({used}), colored by coarse cluster",
        max_points=max_points,
    )
    if X2_test is not None:
        plot_embedding_2d(
            out_path=figures_dir / "embedding_2d_test.png",
            X2=X2_test,
            labels=[_cluster_label(r) for r in test_rows],
            title=f"Test embeddings 2D ({used}), colored by coarse cluster",
            max_points=max_points,
        )

    # Label centroids projection (train-only)
    by_label_ids: dict[str, list[str]] = defaultdict(list)
    for r in splits:
        if r.split == "train":
            by_label_ids[r.label].append(r.clip_id)
    labels = sorted(by_label_ids.keys())
    label_centroids: list[np.ndarray] = []
    label_cluster: list[str] = []
    for lab in labels:
        X = load_embeddings_matrix(artifact_dir, by_label_ids[lab])
        c = X.mean(axis=0)
        c = c / np.clip(np.linalg.norm(c), 1e-12, None)
        label_centroids.append(c.astype(np.float32, copy=False))
        label_cluster.append(str(int(label_to_cluster.get(lab, -1))))
    C = np.stack(label_centroids, axis=0).astype(np.float32, copy=False) if label_centroids else np.zeros((0, 1), dtype=np.float32)

    if C.shape[0] >= 2:
        used_cent, C2, _ = _project_train_and_transform(C, None, seed=seed + 7, method=method, perplexity=min(perplexity, max(5, C.shape[0] // 2)))
        plot_embedding_2d(
            out_path=figures_dir / "label_centroids_2d.png",
            X2=C2,
            labels=label_cluster,
            title=f"Label centroids 2D ({used_cent}), colored by coarse cluster",
            max_points=None,
        )

    # 3) Cluster structure
    cluster_centroids = np.load(str(artifact_dir / "cluster_centroids.npy"))
    plot_cosine_distance_heatmap(
        out_path=figures_dir / "cluster_centroid_cosine_distance.png",
        X=cluster_centroids,
        labels=[str(i) for i in range(cluster_centroids.shape[0])],
        title="Cluster centroid cosine distance",
    )

    # 4) Model performance
    report_path = artifact_dir / "report_coarse.json"
    coarse_report = json.loads(report_path.read_text(encoding="utf-8"))
    cr = coarse_report.get("classification_report", {})
    f1_by_class: dict[str, float] = {}
    for k, v in cr.items():
        if k in ("accuracy", "macro avg", "weighted avg"):
            continue
        if isinstance(v, dict) and "f1-score" in v:
            f1_by_class[str(k)] = float(v["f1-score"])
    if f1_by_class:
        plot_f1_by_class(
            out_path=figures_dir / "f1_by_cluster.png",
            f1_by_class=f1_by_class,
            title="Per-cluster F1 (test)",
        )

    top_conf_path = artifact_dir / "top_confusions.csv"
    pairs: list[tuple[str, str, int]] = []
    if top_conf_path.exists():
        import csv

        with top_conf_path.open("r", encoding="utf-8", newline="") as f:
            dr = csv.DictReader(f)
            for r in dr:
                try:
                    pairs.append((str(r["true"]), str(r["pred"]), int(float(r["count"]))))
                except Exception:
                    continue
    if pairs:
        plot_top_confusions_bar(
            out_path=figures_dir / "top_confusions.png",
            pairs=pairs,
            title="Top coarse confusions (count)",
            max_pairs=20,
        )

    write_json(
        artifact_dir / "report_index.json",
        {
            "figures_dir": str(figures_dir),
            "embedding_2d_method": used,
        },
    )

    log.info("report: wrote figures to %s", figures_dir)
    return figures_dir


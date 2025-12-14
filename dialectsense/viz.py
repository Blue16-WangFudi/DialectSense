from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Microsoft YaHei",
    "SimHei",
    "WenQuanYi Zen Hei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


def plot_bar_counts(
    out_path: str | Path,
    counts: dict[str, int],
    title: str,
    max_labels: int = 30,
    other_label: str = "other",
) -> None:
    out_path = Path(out_path)
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if max_labels is not None and len(items) > int(max_labels):
        head = items[: int(max_labels)]
        tail = items[int(max_labels) :]
        head.append((other_label, int(sum(v for _, v in tail))))
        items = head

    labels = [k for k, _ in items]
    vals = np.array([v for _, v in items], dtype=float)

    fig_w = max(10.0, 0.35 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))
    x = np.arange(len(labels))
    ax.bar(x, vals, width=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_drop_reasons(out_path: str | Path, reason_counts: dict[str, int], title: str) -> None:
    out_path = Path(out_path)
    items = sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    labels = [k for k, _ in items]
    vals = np.array([v for _, v in items], dtype=float)

    fig_w = max(9.0, 0.45 * max(1, len(labels)))
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    x = np.arange(len(labels))
    ax.bar(x, vals)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_duration_hist(
    out_path: str | Path,
    durations_sec: Iterable[float],
    title: str,
    bins: int = 60,
) -> None:
    out_path = Path(out_path)
    d = np.array([float(x) for x in durations_sec if x is not None and math.isfinite(float(x))], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    if d.size:
        ax.hist(d, bins=bins, color="#4C78A8", alpha=0.85)
        ax.set_xlim(left=0.0, right=float(np.percentile(d, 99.5)))
    ax.set_xlabel("seconds")
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_confusion_matrix(
    out_path: str | Path,
    cm: np.ndarray,
    labels: list[str],
    title: str,
    normalize: bool = True,
) -> None:
    out_path = Path(out_path)
    cm_disp = cm.astype(np.float64)
    if normalize:
        row_sum = cm_disp.sum(axis=1, keepdims=True)
        cm_disp = np.divide(cm_disp, row_sum, out=np.zeros_like(cm_disp), where=row_sum != 0)

    fig_w = max(7.5, 0.45 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_w))
    im = ax.imshow(cm_disp, cmap="Blues", vmin=0.0, vmax=max(1e-9, float(cm_disp.max())))
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("pred")
    ax.set_ylabel("true")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_embedding_2d(
    out_path: str | Path,
    X2: np.ndarray,
    labels: Iterable[str],
    title: str,
    max_points: int | None = None,
) -> None:
    out_path = Path(out_path)
    y = np.asarray(list(labels), dtype=object)
    if max_points is not None and X2.shape[0] > int(max_points):
        idx = np.linspace(0, X2.shape[0] - 1, int(max_points)).astype(int)
        X2 = X2[idx]
        y = y[idx]

    uniq = sorted(set(y.tolist()))
    color_map = {lab: i for i, lab in enumerate(uniq)}
    c = np.array([color_map[str(v)] for v in y], dtype=int)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(X2[:, 0], X2[:, 1], c=c, s=8, alpha=0.75, cmap="tab20")
    ax.set_title(title)
    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")

    handles = []
    for lab in uniq[:20]:
        denom = max(1, len(uniq) - 1)
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=lab,
                markerfacecolor=sc.cmap(color_map[lab] / denom),
                markersize=6,
            )
        )
    if handles:
        ax.legend(handles=handles, loc="best", fontsize=8, frameon=True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_cosine_distance_heatmap(out_path: str | Path, X: np.ndarray, labels: list[str], title: str) -> None:
    out_path = Path(out_path)
    X = X.astype(np.float64, copy=False)
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    sim = X @ X.T
    dist = 1.0 - sim

    fig_w = max(7.5, 0.45 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_w))
    im = ax.imshow(dist, cmap="magma", vmin=0.0, vmax=float(np.percentile(dist, 99)))
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_f1_by_class(out_path: str | Path, f1_by_class: dict[str, float], title: str) -> None:
    out_path = Path(out_path)
    items = sorted(f1_by_class.items(), key=lambda kv: (-kv[1], kv[0]))
    labels = [k for k, _ in items]
    vals = np.array([float(v) for _, v in items], dtype=float)

    fig_w = max(8.0, 0.55 * max(1, len(labels)))
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    x = np.arange(len(labels))
    ax.bar(x, vals, color="#72B7B2")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel("F1")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_top_confusions_bar(out_path: str | Path, pairs: list[tuple[str, str, int]], title: str, max_pairs: int = 20) -> None:
    out_path = Path(out_path)
    pairs = pairs[: int(max_pairs)]
    labels = [f"{t}→{p}" for t, p, _ in pairs]
    vals = np.array([int(c) for _, _, c in pairs], dtype=float)

    fig_w = max(10.0, 0.55 * max(1, len(labels)))
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    x = np.arange(len(labels))
    ax.bar(x, vals, color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)

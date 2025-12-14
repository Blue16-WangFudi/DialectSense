from __future__ import annotations

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


def plot_label_distribution(
    out_path: str | Path,
    counts_before: dict[str, int],
    counts_after: dict[str, int],
    title: str,
    max_labels: int = 30,
) -> None:
    out_path = Path(out_path)
    labels = sorted(set(counts_before) | set(counts_after))
    labels = sorted(labels, key=lambda k: (-counts_after.get(k, 0), k))[:max_labels]

    before = np.array([counts_before.get(l, 0) for l in labels], dtype=float)
    after = np.array([counts_after.get(l, 0) for l in labels], dtype=float)

    fig_w = max(10.0, 0.35 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, before, width=0.4, label="before")
    ax.bar(x + 0.2, after, width=0.4, label="after")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("samples")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
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

    fig_w = max(8.0, 0.35 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_w))
    im = ax.imshow(cm_disp, cmap="Blues")
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
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_embedding_2d(
    out_path: str | Path,
    X2: np.ndarray,
    y: Iterable[str],
    title: str,
    max_points: int | None = None,
) -> None:
    out_path = Path(out_path)
    y = np.array(list(y))
    if max_points is not None and X2.shape[0] > max_points:
        idx = np.linspace(0, X2.shape[0] - 1, max_points).astype(int)
        X2 = X2[idx]
        y = y[idx]

    labels = sorted(set(y.tolist()))
    color_map = {lab: i for i, lab in enumerate(labels)}
    c = np.array([color_map[v] for v in y], dtype=int)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(X2[:, 0], X2[:, 1], c=c, s=8, alpha=0.75, cmap="tab20")
    ax.set_title(title)
    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    handles = []
    for lab in labels[:20]:
        handles.append(plt.Line2D([0], [0], marker="o", color="w", label=lab, markerfacecolor=sc.cmap(color_map[lab] / max(1, len(labels) - 1)), markersize=6))
    if handles:
        ax.legend(handles=handles, loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

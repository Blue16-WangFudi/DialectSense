from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .embeddings_io import load_embeddings_matrix
from .splits_file import read_splits_csv
from .util import ensure_dir, write_json
from .viz import plot_confusion_matrix


log = logging.getLogger("dialectsense")


def _write_top_confusions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["true", "pred", "count", "rate"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run_eval(cfg: dict[str, Any], artifact_dir: Path) -> Path:
    model_path = artifact_dir / "models" / "coarse_svm.joblib"
    bundle = joblib.load(model_path)
    model = bundle["model"]
    label_to_cluster = bundle["label_to_cluster"]

    splits = read_splits_csv(artifact_dir / "splits.csv")

    def _coarse(label: str) -> int | None:
        cid = label_to_cluster.get(label)
        return int(cid) if cid is not None else None

    test = [r for r in splits if r.split == "test" and _coarse(r.label) is not None]
    if not test:
        raise RuntimeError("No test rows after coarse mapping")

    y_true = np.array([_coarse(r.label) for r in test], dtype=int)
    X = load_embeddings_matrix(artifact_dir, [r.clip_id for r in test])

    y_pred = model.predict(X)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))

    classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)

    top: list[dict[str, Any]] = []
    for i, t in enumerate(classes):
        row_sum = int(cm[i].sum())
        if row_sum <= 0:
            continue
        for j, p in enumerate(classes):
            if i == j:
                continue
            c = int(cm[i, j])
            if c <= 0:
                continue
            top.append({"true": str(t), "pred": str(p), "count": c, "rate": float(c / row_sum)})
    top.sort(key=lambda r: (-int(r["count"]), -float(r["rate"]), r["true"], r["pred"]))

    figures_dir = ensure_dir(artifact_dir / "figures")
    plot_confusion_matrix(
        out_path=figures_dir / "confusion_matrix_coarse.png",
        cm=cm,
        labels=[str(c) for c in classes],
        title="Confusion matrix (coarse clusters, row-normalized)",
        normalize=True,
    )

    _write_top_confusions(artifact_dir / "top_confusions.csv", top[:200])

    write_json(
        artifact_dir / "report_coarse.json",
        {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "n_test": int(X.shape[0]),
            "classes": classes,
            "classification_report": report,
        },
    )
    log.info("eval: accuracy=%.4f macro_f1=%.4f", acc, macro_f1)
    return artifact_dir / "report_coarse.json"


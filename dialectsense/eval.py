from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score

from .viz import plot_confusion_matrix


def evaluate_and_save(
    artifact_dir: Path,
    model_bundle: dict[str, Any],
    X_test: np.ndarray,
    y_test: list[str],
) -> dict[str, Any]:
    reports_dir = artifact_dir / "reports"
    figs_dir = artifact_dir / "figs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    label_list: list[str] = model_bundle["label_list"]
    label_to_index: dict[str, int] = model_bundle["label_to_index"]
    model = model_bundle["model"]

    y_true = np.array([label_to_index.get(y, -1) for y in y_test], dtype=int)
    keep = y_true >= 0
    X_test = X_test[keep]
    y_true = y_true[keep]

    y_pred = model.predict(X_test)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(label_list)))

    report_txt = classification_report(y_true, y_pred, target_names=label_list, digits=4)
    (reports_dir / "classification_report.txt").write_text(report_txt + "\n", encoding="utf-8")

    metrics = {
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "n_test_used": int(X_test.shape[0]),
        "n_classes": int(len(label_list)),
    }
    (reports_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    plot_confusion_matrix(
        out_path=figs_dir / "confusion_matrix.png",
        cm=cm,
        labels=label_list,
        title="Confusion matrix (row-normalized)",
        normalize=True,
    )

    np.savez_compressed(str(reports_dir / "confusion_matrix.npz"), cm=cm, labels=np.array(label_list, dtype=object))
    return metrics


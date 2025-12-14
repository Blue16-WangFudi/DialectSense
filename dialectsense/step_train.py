from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .config import require
from .embeddings_io import load_embeddings_matrix
from .splits_file import read_splits_csv
from .util import ensure_dir, write_json


log = logging.getLogger("dialectsense")


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class DecisionToProbaWrapper:
    estimator: Any
    classes_: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.estimator.decision_function(X)
        if scores.ndim == 1:
            scores = np.stack([-scores, scores], axis=1)
        return _softmax(scores.astype(np.float64))


def run_train(cfg: dict[str, Any], artifact_dir: Path) -> Path:
    model_cfg = require(cfg, "model")

    splits = read_splits_csv(artifact_dir / "splits.csv")
    label_to_cluster = json.loads((artifact_dir / "label_to_cluster.json").read_text(encoding="utf-8"))

    def _coarse(r) -> int | None:
        cid = label_to_cluster.get(r.label)
        return int(cid) if cid is not None else None

    train = [r for r in splits if r.split == "train" and _coarse(r) is not None]
    val = [r for r in splits if r.split == "val" and _coarse(r) is not None]
    if not train:
        raise RuntimeError("No train rows after coarse mapping")
    if not val:
        raise RuntimeError("No val rows after coarse mapping (needed for calibration)")

    y_train = np.array([_coarse(r) for r in train], dtype=int)
    y_val = np.array([_coarse(r) for r in val], dtype=int)

    X_train = load_embeddings_matrix(artifact_dir, [r.clip_id for r in train])
    X_val = load_embeddings_matrix(artifact_dir, [r.clip_id for r in val])

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                LinearSVC(
                    class_weight=model_cfg.get("class_weight", "balanced"),
                    C=float(model_cfg.get("C", 1.0)),
                    max_iter=int(model_cfg.get("max_iter", 8000)),
                    dual="auto",
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)

    calibrated: Any
    if len(set(y_val.tolist())) >= 2 and bool(model_cfg.get("calibration", {}).get("enabled", True)):
        calibrated = CalibratedClassifierCV(pipe, method=str(model_cfg.get("calibration", {}).get("method", "sigmoid")), cv="prefit")
        calibrated.fit(X_val, y_val)
    else:
        classes_ = np.array(sorted(set(y_train.tolist())), dtype=int)
        calibrated = DecisionToProbaWrapper(estimator=pipe, classes_=classes_)

    models_dir = ensure_dir(artifact_dir / "models")
    out_path = models_dir / "coarse_svm.joblib"
    joblib.dump(
        {
            "model": calibrated,
            "model_cfg": model_cfg,
            "label_to_cluster": label_to_cluster,
            "classes": sorted(set(y_train.tolist())),
        },
        out_path,
    )

    write_json(
        models_dir / "coarse_model_meta.json",
        {
            "n_train": int(X_train.shape[0]),
            "n_val": int(X_val.shape[0]),
            "n_dim": int(X_train.shape[1]),
            "classes": sorted(set(y_train.tolist())),
        },
    )

    log.info("train: wrote %s", out_path)
    return out_path


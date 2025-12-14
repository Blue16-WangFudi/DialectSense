from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


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


@dataclass
class DecisionLogitCalibrator:
    estimator: Any
    calibrator: LogisticRegression
    classes_: np.ndarray

    def _scores(self, X: np.ndarray) -> np.ndarray:
        s = self.estimator.decision_function(X)
        if s.ndim == 1:
            s = s.reshape(-1, 1)
        return s

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.calibrator.predict(self._scores(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.calibrator.predict_proba(self._scores(X))


def _label_mapping(labels: list[str]) -> tuple[list[str], dict[str, int]]:
    uniq = sorted(set(labels))
    return uniq, {lab: i for i, lab in enumerate(uniq)}


def _build_svm(model_cfg: dict[str, Any]) -> Any:
    kind = str(model_cfg.get("svm", "linear_svc"))
    class_weight = model_cfg.get("class_weight", None)
    if kind == "linear_svc":
        return LinearSVC(class_weight=class_weight, dual="auto", max_iter=5000)
    if kind == "svc_rbf":
        return SVC(kernel="rbf", class_weight=class_weight, probability=False)
    raise ValueError(f"Unknown svm kind: {kind}")


def train_and_save(
    artifact_dir: Path,
    model_cfg: dict[str, Any],
    X_train: np.ndarray,
    y_train: list[str],
    votes_train: np.ndarray | None,
    X_val: np.ndarray,
    y_val: list[str],
) -> Path:
    models_dir = artifact_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    label_list, label_to_index = _label_mapping(y_train)
    y_train_i = np.array([label_to_index[y] for y in y_train], dtype=int)
    y_val_i = np.array([label_to_index.get(y, -1) for y in y_val], dtype=int)

    base = _build_svm(model_cfg)
    pipe = Pipeline([("scaler", StandardScaler()), ("svm", base)])

    sample_weight = None
    if model_cfg.get("sample_weight_by_votes") == "log1p" and votes_train is not None:
        sample_weight = np.log1p(votes_train.clip(min=0)).astype(np.float64)

    pipe.fit(X_train, y_train_i, svm__sample_weight=sample_weight)

    calibrated: Any = None
    calibration = str(model_cfg.get("calibration", "sigmoid_on_val"))
    if calibration == "sigmoid_on_val":
        present = set(y_val_i.tolist())
        missing = [i for i in range(len(label_list)) if i not in present]
        if (-1 in present) or missing or len(set(y_val_i.tolist())) < 2:
            calibrated = DecisionToProbaWrapper(estimator=pipe, classes_=np.arange(len(label_list)))
        else:
            scores_val = pipe.decision_function(X_val)
            if scores_val.ndim == 1:
                scores_val = scores_val.reshape(-1, 1)
            lr = LogisticRegression(max_iter=2000)
            lr.fit(scores_val, y_val_i)
            calibrated = DecisionLogitCalibrator(estimator=pipe, calibrator=lr, classes_=np.arange(len(label_list)))
    else:
        calibrated = DecisionToProbaWrapper(estimator=pipe, classes_=np.arange(len(label_list)))

    (models_dir / "label_list.json").write_text(
        json.dumps(label_list, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (models_dir / "label_to_index.json").write_text(
        json.dumps(label_to_index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    out_path = models_dir / "svm.joblib"
    joblib.dump(
        {
            "model": calibrated,
            "label_list": label_list,
            "label_to_index": label_to_index,
            "model_cfg": model_cfg,
        },
        out_path,
    )
    return out_path


def load_model(model_path: Path) -> dict[str, Any]:
    return joblib.load(model_path)

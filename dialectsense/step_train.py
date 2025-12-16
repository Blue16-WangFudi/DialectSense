from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

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


def _build_estimator(model_cfg: dict[str, Any]) -> Any:
    kind = str(model_cfg.get("svm_kind", "linear_svc"))
    class_weight = model_cfg.get("class_weight", None)
    C = float(model_cfg.get("C", 1.0))
    max_iter = int(model_cfg.get("max_iter", 8000))
    if kind == "linear_svc":
        return LinearSVC(class_weight=class_weight, C=C, max_iter=max_iter, dual="auto")
    if kind == "svc_rbf":
        gamma = model_cfg.get("gamma", "scale")
        return SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            class_weight=class_weight,
            probability=False,
            decision_function_shape="ovr",
        )
    raise ValueError(f"Unknown svm_kind: {kind}")


def _fit_and_score_accuracy(
    model_cfg: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[Pipeline, float]:
    est = _build_estimator(model_cfg)
    pipe = Pipeline([("scaler", StandardScaler()), ("svm", est)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_val)
    return pipe, float(accuracy_score(y_val, y_pred))


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

    tune_cfg = model_cfg.get("tune", {}) if isinstance(model_cfg.get("tune"), dict) else {}
    tune_enabled = bool(tune_cfg.get("enabled", False))

    chosen_cfg = dict(model_cfg)
    tune_results: list[dict[str, Any]] = []
    if tune_enabled:
        log.info("train: tuning enabled (objective=val_accuracy)")
        C_grid = tune_cfg.get("C_grid", [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
        kind_grid = tune_cfg.get("svm_kind_grid", ["linear_svc"])
        class_weight_grid = tune_cfg.get("class_weight_grid", [None, "balanced"])
        gamma_grid = tune_cfg.get("gamma_grid", ["scale", "auto"])

        best_acc = -1.0
        best_pipe: Pipeline | None = None
        for kind in kind_grid:
            for cw in class_weight_grid:
                for C in C_grid:
                    if str(kind) == "svc_rbf":
                        for gamma in gamma_grid:
                            cand = dict(chosen_cfg)
                            cand.update({"svm_kind": str(kind), "class_weight": cw, "C": float(C), "gamma": gamma})
                            pipe_c, acc_c = _fit_and_score_accuracy(cand, X_train, y_train, X_val, y_val)
                            tune_results.append({"svm_kind": str(kind), "class_weight": cw, "C": float(C), "gamma": gamma, "val_accuracy": acc_c})
                            if acc_c > best_acc:
                                best_acc, best_pipe, chosen_cfg = acc_c, pipe_c, cand
                    else:
                        cand = dict(chosen_cfg)
                        cand.update({"svm_kind": str(kind), "class_weight": cw, "C": float(C)})
                        pipe_c, acc_c = _fit_and_score_accuracy(cand, X_train, y_train, X_val, y_val)
                        tune_results.append({"svm_kind": str(kind), "class_weight": cw, "C": float(C), "val_accuracy": acc_c})
                        if acc_c > best_acc:
                            best_acc, best_pipe, chosen_cfg = acc_c, pipe_c, cand

        if best_pipe is None:
            raise RuntimeError("train: tuning failed to produce any candidate")

        log.info("train: best val_accuracy=%.4f cfg=%s", best_acc, {k: chosen_cfg.get(k) for k in ["svm_kind", "C", "class_weight", "gamma"] if k in chosen_cfg})
        pipe = best_pipe
    else:
        pipe, _ = _fit_and_score_accuracy(chosen_cfg, X_train, y_train, X_val, y_val)

    final_train = str(tune_cfg.get("final_train", "train_only"))
    if tune_enabled and final_train == "train_val":
        log.info("train: final_train=train_val (refit on train+val for best accuracy)")
        X_tv = np.concatenate([X_train, X_val], axis=0)
        y_tv = np.concatenate([y_train, y_val], axis=0)
        pipe = Pipeline([("scaler", StandardScaler()), ("svm", _build_estimator(chosen_cfg))])
        pipe.fit(X_tv, y_tv)

    calibrated: Any
    classes_ = np.array(sorted(set(y_train.tolist())), dtype=int)
    cal_cfg = model_cfg.get("calibration", {}) if isinstance(model_cfg.get("calibration"), dict) else {}
    cal_enabled = bool(cal_cfg.get("enabled", True))
    if tune_enabled and final_train == "train_val":
        cal_enabled = False
    if cal_enabled and len(set(y_val.tolist())) >= 2:
        present = set(y_val.tolist())
        missing = [int(c) for c in classes_.tolist() if int(c) not in present]
        if missing:
            log.warning("train: calibration skipped (val missing classes=%s)", missing)
            calibrated = DecisionToProbaWrapper(estimator=pipe, classes_=classes_)
        else:
            scores = pipe.decision_function(X_val)
            if scores.ndim == 1:
                scores = scores.reshape(-1, 1)
            lr = LogisticRegression(max_iter=int(cal_cfg.get("max_iter", 2000)))
            lr.fit(scores, y_val)
            calibrated = DecisionLogitCalibrator(estimator=pipe, calibrator=lr, classes_=classes_)
    else:
        calibrated = DecisionToProbaWrapper(estimator=pipe, classes_=classes_)

    models_dir = ensure_dir(artifact_dir / "models")
    out_path = models_dir / "coarse_svm.joblib"
    joblib.dump(
        {
            "model": calibrated,
            "requested_model_cfg": model_cfg,
            "trained_model_cfg": chosen_cfg,
            "label_to_cluster": label_to_cluster,
            "classes": sorted(set(y_train.tolist())),
            "tune": {
                "enabled": tune_enabled,
                "objective": "val_accuracy",
                "final_train": final_train,
                "results": tune_results[: int(tune_cfg.get("max_saved", 2000))],
            }
            if tune_enabled
            else {"enabled": False},
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

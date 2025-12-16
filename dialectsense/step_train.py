from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

from .config import require
from .embeddings_io import load_embeddings_matrix
from .splits_file import read_splits_csv
from .util import ensure_dir, write_json
from .prototypes import PrototypeCosineClassifier
from .stacked import StackedCoarseClassifier


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

@dataclass
class ProbaModelWrapper:
    estimator: Any
    classes_: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(X)


def _class_weight_to_sample_weight(class_weight: Any, y: np.ndarray) -> np.ndarray | None:
    if class_weight is None:
        return None
    if str(class_weight) != "balanced":
        return None
    # Inverse-frequency weights: n / (k * n_c)
    y = np.asarray(y, dtype=int).reshape(-1)
    if y.size == 0:
        return None
    uniq, cnt = np.unique(y, return_counts=True)
    k = float(len(uniq))
    n = float(y.size)
    w_map = {int(c): float(n / (k * float(nc))) for c, nc in zip(uniq.tolist(), cnt.tolist())}
    return np.asarray([w_map[int(v)] for v in y.tolist()], dtype=np.float64)


def _build_estimator(model_cfg: dict[str, Any]) -> Any:
    kind = str(model_cfg.get("kind") or model_cfg.get("svm_kind") or "mlp")
    C = float(model_cfg.get("C", 1.0))
    max_iter = int(model_cfg.get("max_iter", 8000))

    if kind == "linear_svc":
        return LinearSVC(class_weight=None, C=C, max_iter=max_iter, dual="auto")
    if kind == "svc_rbf":
        gamma = model_cfg.get("gamma", "scale")
        return SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            class_weight=None,
            probability=True,
            decision_function_shape="ovr",
        )
    if kind == "logreg":
        lr_C = float(model_cfg.get("logreg_C", 2.0))
        return LogisticRegression(
            max_iter=int(model_cfg.get("logreg_max_iter", 4000)),
            C=lr_C,
            n_jobs=int(model_cfg.get("logreg_n_jobs", -1)),
            solver=str(model_cfg.get("logreg_solver", "lbfgs")),
        )
    if kind == "mlp":
        hidden = model_cfg.get("mlp_hidden", [512, 128])
        if isinstance(hidden, int):
            hidden = [hidden]
        hidden_t = tuple(int(x) for x in hidden)
        return MLPClassifier(
            hidden_layer_sizes=hidden_t,
            activation=str(model_cfg.get("mlp_activation", "relu")),
            alpha=float(model_cfg.get("mlp_alpha", 1e-4)),
            learning_rate_init=float(model_cfg.get("mlp_lr", 1e-3)),
            max_iter=int(model_cfg.get("mlp_max_iter", 80)),
            early_stopping=bool(model_cfg.get("mlp_early_stopping", True)),
            n_iter_no_change=int(model_cfg.get("mlp_n_iter_no_change", 8)),
            validation_fraction=float(model_cfg.get("mlp_validation_fraction", 0.1)),
            random_state=int(model_cfg.get("seed", 42)),
        )
    if kind == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=int(model_cfg.get("trees_n_estimators", 600)),
            max_features=model_cfg.get("trees_max_features", "sqrt"),
            min_samples_leaf=int(model_cfg.get("trees_min_samples_leaf", 2)),
            random_state=int(model_cfg.get("seed", 42)),
            n_jobs=int(model_cfg.get("trees_n_jobs", -1)),
        )
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=float(model_cfg.get("hgb_learning_rate", 0.08)),
            max_iter=int(model_cfg.get("hgb_max_iter", 300)),
            max_depth=model_cfg.get("hgb_max_depth", None),
            max_leaf_nodes=int(model_cfg.get("hgb_max_leaf_nodes", 63)),
            min_samples_leaf=int(model_cfg.get("hgb_min_samples_leaf", 30)),
            l2_regularization=float(model_cfg.get("hgb_l2", 0.0)),
            random_state=int(model_cfg.get("seed", 42)),
        )
    if kind == "proto_cosine":
        return PrototypeCosineClassifier(
            prototypes_per_class=int(model_cfg.get("proto_k", 6)),
            reduce=str(model_cfg.get("proto_reduce", "max")),
            n_init=int(model_cfg.get("proto_n_init", 10)),
            random_state=int(model_cfg.get("seed", 42)),
        )
    raise ValueError(f"Unknown model kind: {kind}")

def _use_scaler(model_cfg: dict[str, Any]) -> bool:
    if "scale" in model_cfg:
        return bool(model_cfg.get("scale"))
    kind = str(model_cfg.get("kind") or model_cfg.get("svm_kind") or "mlp")
    return kind not in {"extra_trees", "hgb", "proto_cosine"}


def _fit_and_score_accuracy(
    model_cfg: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[Pipeline, float]:
    est = _build_estimator(model_cfg)
    if _use_scaler(model_cfg):
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", est)])
    else:
        pipe = Pipeline([("clf", est)])

    sw = _class_weight_to_sample_weight(model_cfg.get("class_weight"), y_train)
    pipe.fit(X_train, y_train, clf__sample_weight=sw)
    y_pred = pipe.predict(X_val)
    return pipe, float(accuracy_score(y_val, y_pred))


def run_train(cfg: dict[str, Any], artifact_dir: Path) -> Path:
    model_cfg = require(cfg, "model")
    model_kind = str(model_cfg.get("kind") or model_cfg.get("svm_kind") or "mlp")

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

    # Special path: stacking for best accuracy.
    if model_kind == "stacked":
        svm_cfg = model_cfg.get("stacked_svm", {}) if isinstance(model_cfg.get("stacked_svm"), dict) else {}
        mlp_cfg = model_cfg.get("stacked_mlp", {}) if isinstance(model_cfg.get("stacked_mlp"), dict) else {}
        meta_cfg = model_cfg.get("stacked_meta", {}) if isinstance(model_cfg.get("stacked_meta"), dict) else {}

        classes_ = np.array(sorted(set(y_train.tolist()) | set(y_val.tolist())), dtype=int)

        base_svm = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LinearSVC(
                        C=float(svm_cfg.get("C", 1.0)),
                        max_iter=int(svm_cfg.get("max_iter", 8000)),
                        dual="auto",
                        class_weight=svm_cfg.get("class_weight", "balanced"),
                    ),
                ),
            ]
        )
        base_svm.fit(X_train, y_train)

        base_mlp = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=tuple(int(x) for x in mlp_cfg.get("hidden", [1024, 256])),
                        alpha=float(mlp_cfg.get("alpha", 1e-4)),
                        learning_rate_init=float(mlp_cfg.get("lr", 8e-4)),
                        max_iter=int(mlp_cfg.get("max_iter", 140)),
                        early_stopping=bool(mlp_cfg.get("early_stopping", True)),
                        n_iter_no_change=int(mlp_cfg.get("n_iter_no_change", 10)),
                        validation_fraction=float(mlp_cfg.get("validation_fraction", 0.1)),
                        random_state=int(cfg.get("seed", 42)),
                    ),
                ),
            ]
        )
        base_mlp.fit(X_train, y_train)

        meta = LogisticRegression(
            C=float(meta_cfg.get("C", 2.0)),
            max_iter=int(meta_cfg.get("max_iter", 4000)),
            solver=str(meta_cfg.get("solver", "lbfgs")),
        )
        stacked = StackedCoarseClassifier(base_svm=base_svm, base_mlp=base_mlp, meta=meta, classes_=classes_)
        meta.fit(stacked._features(X_val), y_val)

        models_dir = ensure_dir(artifact_dir / "models")
        out_path = models_dir / "coarse_model.joblib"
        joblib.dump(
            {
                "model": stacked,
                "requested_model_cfg": model_cfg,
                "trained_model_cfg": {
                    "kind": "stacked",
                    "stacked_svm": svm_cfg,
                    "stacked_mlp": mlp_cfg,
                    "stacked_meta": meta_cfg,
                },
                "label_to_cluster": label_to_cluster,
                "classes": classes_.tolist(),
                "tune": {"enabled": False},
            },
            out_path,
        )
        try:
            import shutil

            shutil.copyfile(out_path, models_dir / "coarse_svm.joblib")
        except Exception:
            pass

        write_json(
            models_dir / "coarse_model_meta.json",
            {
                "n_train": int(X_train.shape[0]),
                "n_val": int(X_val.shape[0]),
                "n_dim": int(X_train.shape[1]),
                "classes": classes_.tolist(),
            },
        )
        log.info("train: wrote %s", out_path)
        return out_path

    tune_cfg = model_cfg.get("tune", {}) if isinstance(model_cfg.get("tune"), dict) else {}
    tune_enabled = bool(tune_cfg.get("enabled", False))

    chosen_cfg = dict(model_cfg)
    tune_results: list[dict[str, Any]] = []
    if tune_enabled:
        log.info("train: tuning enabled (objective=val_accuracy)")
        kind_grid = tune_cfg.get("kind_grid", ["proto_cosine", "mlp", "logreg", "linear_svc"])
        C_grid = tune_cfg.get("C_grid", [0.1, 0.3, 1.0, 3.0])
        class_weight_grid = tune_cfg.get("class_weight_grid", [None, "balanced"])
        mlp_hidden_grid = tune_cfg.get("mlp_hidden_grid", [[512, 128], [768, 256]])
        mlp_alpha_grid = tune_cfg.get("mlp_alpha_grid", [1e-4, 3e-4])
        mlp_lr_grid = tune_cfg.get("mlp_lr_grid", [1e-3])
        logreg_C_grid = tune_cfg.get("logreg_C_grid", [0.5, 1.0, 2.0, 4.0])
        gamma_grid = tune_cfg.get("gamma_grid", ["scale"])
        trees_grid = tune_cfg.get("trees_n_estimators_grid", [400, 800])
        proto_k_grid = tune_cfg.get("proto_k_grid", [2, 4, 6, 8])

        best_acc = -1.0
        best_pipe: Pipeline | None = None
        for kind in kind_grid:
            kind = str(kind)
            if kind == "mlp":
                for cw in class_weight_grid:
                    for hidden in mlp_hidden_grid:
                        for a in mlp_alpha_grid:
                            for lr in mlp_lr_grid:
                                cand = dict(chosen_cfg)
                                cand.update(
                                    {
                                        "kind": "mlp",
                                        "class_weight": cw,
                                        "mlp_hidden": hidden,
                                        "mlp_alpha": float(a),
                                        "mlp_lr": float(lr),
                                        "seed": int(cfg.get("seed", 42)),
                                    }
                                )
                                pipe_c, acc_c = _fit_and_score_accuracy(cand, X_train, y_train, X_val, y_val)
                                tune_results.append(
                                    {
                                        "kind": "mlp",
                                        "class_weight": cw,
                                        "mlp_hidden": list(hidden) if isinstance(hidden, (list, tuple)) else hidden,
                                        "mlp_alpha": float(a),
                                        "mlp_lr": float(lr),
                                        "val_accuracy": acc_c,
                                    }
                                )
                                if acc_c > best_acc:
                                    best_acc, best_pipe, chosen_cfg = acc_c, pipe_c, cand
            elif kind == "logreg":
                for cw in class_weight_grid:
                    for C_lr in logreg_C_grid:
                        cand = dict(chosen_cfg)
                        cand.update({"kind": "logreg", "class_weight": cw, "logreg_C": float(C_lr)})
                        pipe_c, acc_c = _fit_and_score_accuracy(cand, X_train, y_train, X_val, y_val)
                        tune_results.append({"kind": "logreg", "class_weight": cw, "logreg_C": float(C_lr), "val_accuracy": acc_c})
                        if acc_c > best_acc:
                            best_acc, best_pipe, chosen_cfg = acc_c, pipe_c, cand
            elif kind in {"linear_svc", "svc_rbf"}:
                for cw in class_weight_grid:
                    for C in C_grid:
                        if kind == "svc_rbf":
                            for gamma in gamma_grid:
                                cand = dict(chosen_cfg)
                                cand.update({"kind": "svc_rbf", "class_weight": cw, "C": float(C), "gamma": gamma})
                                pipe_c, acc_c = _fit_and_score_accuracy(cand, X_train, y_train, X_val, y_val)
                                tune_results.append({"kind": "svc_rbf", "class_weight": cw, "C": float(C), "gamma": gamma, "val_accuracy": acc_c})
                                if acc_c > best_acc:
                                    best_acc, best_pipe, chosen_cfg = acc_c, pipe_c, cand
                        else:
                            cand = dict(chosen_cfg)
                            cand.update({"kind": "linear_svc", "class_weight": cw, "C": float(C)})
                            pipe_c, acc_c = _fit_and_score_accuracy(cand, X_train, y_train, X_val, y_val)
                            tune_results.append({"kind": "linear_svc", "class_weight": cw, "C": float(C), "val_accuracy": acc_c})
                            if acc_c > best_acc:
                                best_acc, best_pipe, chosen_cfg = acc_c, pipe_c, cand
            elif kind == "extra_trees":
                for n_estimators in trees_grid:
                    cand = dict(chosen_cfg)
                    cand.update(
                        {
                            "kind": "extra_trees",
                            "scale": False,
                            "trees_n_estimators": int(n_estimators),
                            "seed": int(cfg.get("seed", 42)),
                        }
                    )
                    pipe_c, acc_c = _fit_and_score_accuracy(cand, X_train, y_train, X_val, y_val)
                    tune_results.append({"kind": "extra_trees", "trees_n_estimators": int(n_estimators), "val_accuracy": acc_c})
                    if acc_c > best_acc:
                        best_acc, best_pipe, chosen_cfg = acc_c, pipe_c, cand
            elif kind == "proto_cosine":
                for pk in proto_k_grid:
                    cand = dict(chosen_cfg)
                    cand.update({"kind": "proto_cosine", "scale": False, "proto_k": int(pk), "seed": int(cfg.get("seed", 42))})
                    pipe_c, acc_c = _fit_and_score_accuracy(cand, X_train, y_train, X_val, y_val)
                    tune_results.append({"kind": "proto_cosine", "proto_k": int(pk), "val_accuracy": acc_c})
                    if acc_c > best_acc:
                        best_acc, best_pipe, chosen_cfg = acc_c, pipe_c, cand
            else:
                raise ValueError(f"Unknown kind in tune.kind_grid: {kind}")

        if best_pipe is None:
            raise RuntimeError("train: tuning failed to produce any candidate")

        log.info(
            "train: best val_accuracy=%.4f cfg=%s",
            best_acc,
            {k: chosen_cfg.get(k) for k in ["kind", "C", "class_weight", "gamma", "logreg_C", "mlp_hidden", "mlp_alpha", "mlp_lr"] if k in chosen_cfg},
        )
        pipe = best_pipe
    else:
        pipe, _ = _fit_and_score_accuracy(chosen_cfg, X_train, y_train, X_val, y_val)

    final_train = str(tune_cfg.get("final_train", model_cfg.get("final_train", "train_only")))
    if tune_enabled and final_train == "train_val":
        log.info("train: final_train=train_val (refit on train+val for best accuracy)")
        X_tv = np.concatenate([X_train, X_val], axis=0)
        y_tv = np.concatenate([y_train, y_val], axis=0)
        if _use_scaler(chosen_cfg):
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", _build_estimator(chosen_cfg))])
        else:
            pipe = Pipeline([("clf", _build_estimator(chosen_cfg))])
        sw_tv = _class_weight_to_sample_weight(chosen_cfg.get("class_weight"), y_tv)
        pipe.fit(X_tv, y_tv, clf__sample_weight=sw_tv)
    elif (not tune_enabled) and final_train == "train_val":
        log.info("train: final_train=train_val")
        X_tv = np.concatenate([X_train, X_val], axis=0)
        y_tv = np.concatenate([y_train, y_val], axis=0)
        if _use_scaler(chosen_cfg):
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", _build_estimator(chosen_cfg))])
        else:
            pipe = Pipeline([("clf", _build_estimator(chosen_cfg))])
        sw_tv = _class_weight_to_sample_weight(chosen_cfg.get("class_weight"), y_tv)
        pipe.fit(X_tv, y_tv, clf__sample_weight=sw_tv)

    calibrated: Any
    classes_ = np.array(sorted(set(y_train.tolist())), dtype=int)
    cal_cfg = model_cfg.get("calibration", {}) if isinstance(model_cfg.get("calibration"), dict) else {}
    cal_enabled = bool(cal_cfg.get("enabled", True))
    if tune_enabled and final_train == "train_val":
        cal_enabled = False
    if cal_enabled and hasattr(pipe, "decision_function") and len(set(y_val.tolist())) >= 2:
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
        if hasattr(pipe, "predict_proba"):
            calibrated = ProbaModelWrapper(estimator=pipe, classes_=classes_)
        else:
            calibrated = DecisionToProbaWrapper(estimator=pipe, classes_=classes_)

    models_dir = ensure_dir(artifact_dir / "models")
    out_path = models_dir / "coarse_model.joblib"
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
    # Backwards compatibility (older tooling expected this filename).
    try:
        import shutil

        shutil.copyfile(out_path, models_dir / "coarse_svm.joblib")
    except Exception:
        pass

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

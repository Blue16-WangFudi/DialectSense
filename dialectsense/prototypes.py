from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np


def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.clip(n, 1e-12, None)
    return (X / n).astype(np.float32, copy=False)


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


class PrototypeCosineClassifier(BaseEstimator, ClassifierMixin):
    """
    A lightweight, non-linear classifier suitable for realtime inference:
    - Fits KMeans prototypes per class in embedding space
    - Predicts by max cosine similarity to class prototypes
    """

    def __init__(
        self,
        prototypes_per_class: int = 6,
        reduce: str = "max",
        n_init: int = 10,
        random_state: int = 42,
    ) -> None:
        self.prototypes_per_class = int(prototypes_per_class)
        self.reduce = str(reduce)
        self.n_init = int(n_init)
        self.random_state = int(random_state)

        self.classes_: np.ndarray | None = None
        self.proto_: np.ndarray | None = None
        self.proto_class_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Any = None) -> "PrototypeCosineClassifier":
        from sklearn.cluster import KMeans

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y length mismatch")

        classes = np.array(sorted(set(y.tolist())), dtype=int)
        self.classes_ = classes

        Xn = _l2_normalize_rows(X)

        protos: list[np.ndarray] = []
        proto_cls: list[int] = []
        k = int(max(1, int(self.prototypes_per_class)))

        for c in classes.tolist():
            Xc = Xn[y == int(c)]
            if Xc.shape[0] == 0:
                continue
            if Xc.shape[0] <= k:
                p = Xc.mean(axis=0, keepdims=True)
                p = _l2_normalize_rows(p)
                protos.append(p)
                proto_cls.extend([int(c)] * p.shape[0])
                continue
            km = KMeans(n_clusters=k, n_init=int(self.n_init), random_state=int(self.random_state))
            km.fit(Xc)
            centers = np.asarray(km.cluster_centers_, dtype=np.float32)
            centers = _l2_normalize_rows(centers)
            protos.append(centers)
            proto_cls.extend([int(c)] * centers.shape[0])

        if not protos:
            self.proto_ = np.zeros((0, X.shape[1]), dtype=np.float32)
            self.proto_class_ = np.zeros((0,), dtype=int)
            return self

        self.proto_ = np.concatenate(protos, axis=0).astype(np.float32, copy=False)
        self.proto_class_ = np.asarray(proto_cls, dtype=int)
        return self

    def _scores(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None or self.proto_ is None or self.proto_class_ is None:
            raise RuntimeError("Model is not fitted")
        X = np.asarray(X, dtype=np.float32)
        Xn = _l2_normalize_rows(X)
        S = Xn @ self.proto_.T  # cosine similarity, [N, P]
        out = np.full((Xn.shape[0], self.classes_.shape[0]), -1e9, dtype=np.float64)
        for j, c in enumerate(self.classes_.tolist()):
            mask = self.proto_class_ == int(c)
            if not np.any(mask):
                continue
            s = S[:, mask]
            if str(self.reduce) == "mean":
                out[:, j] = np.asarray(s.mean(axis=1), dtype=np.float64)
            else:
                out[:, j] = np.asarray(s.max(axis=1), dtype=np.float64)
        return out

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self._scores(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self._scores(X)
        idx = np.argmax(scores, axis=1)
        return np.asarray(self.classes_[idx], dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self._scores(X)
        return _softmax_rows(scores)

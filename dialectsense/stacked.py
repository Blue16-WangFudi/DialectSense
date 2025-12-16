from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


def _align_columns(x: np.ndarray, src_classes: np.ndarray, dst_classes: np.ndarray) -> np.ndarray:
    """
    Reorders columns of x from src_classes order to dst_classes order.
    """
    src = [int(v) for v in np.asarray(src_classes, dtype=int).tolist()]
    dst = [int(v) for v in np.asarray(dst_classes, dtype=int).tolist()]
    pos = {c: i for i, c in enumerate(src)}
    cols = []
    for c in dst:
        if c not in pos:
            raise ValueError(f"Missing class {c} in base model outputs")
        cols.append(pos[c])
    return np.asarray(x[:, cols], dtype=np.float64)


@dataclass
class StackedCoarseClassifier:
    """
    Two-stage model:
      - base_svm: provides decision_function features
      - base_mlp: provides predict_proba features
      - meta: multinomial logistic regression on concatenated features (trained on val)
    """

    base_svm: Any
    base_mlp: Any
    meta: Any
    classes_: np.ndarray

    def _features(self, X: np.ndarray) -> np.ndarray:
        s = _ensure_2d(self.base_svm.decision_function(X)).astype(np.float64, copy=False)
        p = np.asarray(self.base_mlp.predict_proba(X), dtype=np.float64)
        s = _align_columns(s, getattr(self.base_svm, "classes_", self.classes_), self.classes_)
        p = _align_columns(p, getattr(self.base_mlp, "classes_", self.classes_), self.classes_)
        return np.concatenate([s, p], axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.meta.predict(self._features(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = self.meta.predict_proba(self._features(X))
        # Avoid using np.asarray(copy=...) for compatibility with older NumPy.
        proba = np.asarray(proba, dtype=np.float64)
        # Align meta output columns to classes_ if needed
        meta_classes = getattr(self.meta, "classes_", None)
        if meta_classes is None:
            return proba
        return _align_columns(proba, np.asarray(meta_classes, dtype=int), self.classes_)

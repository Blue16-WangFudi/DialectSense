from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.manifold import TSNE

from .viz import plot_embedding_2d


def _try_umap() -> Any | None:
    try:
        import umap  # type: ignore

        return umap
    except Exception:
        return None


def project_2d_and_save(
    artifact_dir: Path,
    X: np.ndarray,
    y: list[str],
    cluster_cfg: dict[str, Any],
) -> dict[str, Any]:
    figs_dir = artifact_dir / "figs"
    reports_dir = artifact_dir / "reports"
    models_dir = artifact_dir / "models"
    figs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    method = str(cluster_cfg.get("method", "umap_if_available_else_tsne"))
    n_components = int(cluster_cfg.get("n_components", 2))
    max_points = cluster_cfg.get("max_points", None)
    if max_points is not None:
        max_points = int(max_points)

    y_arr = np.asarray(y, dtype=object)
    if max_points is not None and X.shape[0] > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], size=max_points, replace=False)
        idx.sort()
        X = X[idx]
        y_arr = y_arr[idx]

    projector = None
    used = "tsne"
    umap_mod = _try_umap() if "umap" in method else None
    umap_model_path = models_dir / "umap.joblib"
    if umap_mod is not None:
        used = "umap"
        projector = umap_mod.UMAP(
            n_components=n_components,
            random_state=42,
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
        )
        X2 = projector.fit_transform(X)
        joblib.dump(projector, umap_model_path)
    else:
        perplexity = int(cluster_cfg.get("perplexity", 30))
        projector = TSNE(n_components=n_components, perplexity=perplexity, init="pca", random_state=42)
        X2 = projector.fit_transform(X)
        if umap_model_path.exists():
            umap_model_path.unlink()

    plot_embedding_2d(
        out_path=figs_dir / "embedding_2d.png",
        X2=X2,
        y=y_arr.tolist(),
        title=f"Embedding 2D projection ({used})",
        max_points=4000,
    )

    np.savez_compressed(
        str(reports_dir / "projection_2d.npz"),
        X2=X2.astype(np.float32),
        y=y_arr,
    )
    np.savez_compressed(
        str(reports_dir / "projection_assets.npz"),
        X=X.astype(np.float32),
        X2=X2.astype(np.float32),
        y=y_arr,
        method=np.array([used], dtype=object),
    )
    return {"method_used": used, "n_points": int(X2.shape[0]), "max_points": max_points}

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..config import require, resolve_paths
from ..embed.registry import create_embedder
from ..train import load_model


@dataclass
class UIState:
    t_sec: float = 0.0
    conf_t: list[tuple[float, float]] = field(default_factory=list)  # (t, top1_conf)
    emb_hist: list[np.ndarray] = field(default_factory=list)
    baseline: np.ndarray | None = None


def _render_conf_timeline(points: list[tuple[float, float]]) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(7, 2.6))
    if points:
        t = [p[0] for p in points]
        c = [p[1] for p in points]
        ax.plot(t, c, linewidth=2)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("top1 conf")
    ax.set_title("Confidence over time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    img = plt.imread(buf)
    return img


def _render_embedding_overlay(
    X2_bg: np.ndarray,
    y_bg: np.ndarray,
    labels: list[str],
    point2: np.ndarray,
    title: str,
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(6, 5))
    uniq = sorted(set(y_bg.tolist()))
    cmap = plt.get_cmap("tab20")
    color_map = {lab: i for i, lab in enumerate(uniq)}
    c = np.array([color_map[v] for v in y_bg], dtype=int)
    ax.scatter(X2_bg[:, 0], X2_bg[:, 1], c=c, s=6, alpha=0.4, cmap=cmap)
    ax.scatter([point2[0]], [point2[1]], c="red", s=70, marker="x", linewidths=3)
    ax.set_title(title)
    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    img = plt.imread(buf)
    return img


def _format_topk(label_list: list[str], proba: np.ndarray, k: int) -> list[list[Any]]:
    idx = np.argsort(-proba)[:k]
    rows = []
    for rank, i in enumerate(idx, start=1):
        rows.append([rank, label_list[int(i)], float(proba[int(i)])])
    return rows


def _try_load_image(path: str) -> np.ndarray | None:
    try:
        return plt.imread(path)
    except Exception:
        return None


def _knn_map_to_2d(x: np.ndarray, X_ref: np.ndarray, X2_ref: np.ndarray, k: int = 10) -> np.ndarray:
    if X_ref.shape[0] == 0:
        raise ValueError("Empty reference set")
    k = int(min(k, X_ref.shape[0]))
    diff = X_ref - x[None, :]
    d2 = np.sum(diff * diff, axis=1, dtype=np.float64)
    idx = np.argpartition(d2, kth=k - 1)[:k]
    d = np.sqrt(d2[idx]) + 1e-6
    w = 1.0 / d
    w = w / np.sum(w)
    return (X2_ref[idx] * w[:, None]).sum(axis=0)


def launch(cfg: dict[str, Any]) -> None:
    import gradio as gr
    import joblib

    paths = resolve_paths(cfg)
    artifact_dir = paths.artifact_dir

    embed_cfg = require(cfg, "embed")
    ui_cfg = require(cfg, "ui")
    embedder = create_embedder(embed_cfg)

    model_path = artifact_dir / "models" / "svm.joblib"
    bundle = load_model(model_path)
    model = bundle["model"]
    label_list: list[str] = bundle["label_list"]

    chunk_sec = float(ui_cfg.get("chunk_sec", 1.5))
    history_chunks = int(ui_cfg.get("history_chunks", 10))
    vad_thr = float(ui_cfg.get("vad_rms_threshold", 0.01))
    top_k = int(ui_cfg.get("top_k", 3))

    # Optional 2D projection assets
    proj_bg = None  # (X2, y)
    proj_ref = None  # (X_emb, X2, y)
    umap_model = None
    proj_npz = artifact_dir / "reports" / "projection_2d.npz"
    assets_npz = artifact_dir / "reports" / "projection_assets.npz"
    cluster_meta = artifact_dir / "reports" / "cluster.json"
    umap_path = artifact_dir / "models" / "umap.joblib"
    method_used = None
    if cluster_meta.exists():
        try:
            import json

            meta = json.loads(cluster_meta.read_text(encoding="utf-8"))
            method_used = meta.get("method_used")
        except Exception:
            method_used = None

    projection_static_img = _try_load_image(str(artifact_dir / "figs" / "embedding_2d.png"))

    if proj_npz.exists():
        data = np.load(str(proj_npz), allow_pickle=True)
        X2_bg = data["X2"].astype(np.float32)
        y_bg = data["y"]
        proj_bg = (X2_bg, y_bg)

    if assets_npz.exists():
        data = np.load(str(assets_npz), allow_pickle=True)
        X_ref = data["X"].astype(np.float32)
        X2_ref = data["X2"].astype(np.float32)
        y_ref = data["y"]
        proj_ref = (X_ref, X2_ref, y_ref)
        if method_used is None and "method" in data:
            try:
                method_used = str(data["method"][0])
            except Exception:
                method_used = None

    if method_used == "umap" and umap_path.exists():
        try:
            umap_model = joblib.load(umap_path)
        except Exception:
            umap_model = None

    def _reset() -> tuple[UIState, list[list[Any]], np.ndarray, np.ndarray | None]:
        state = UIState()
        top = [[1, "-", 0.0], [2, "-", 0.0], [3, "-", 0.0]]
        timeline = _render_conf_timeline([])
        overlay = projection_static_img
        return state, top, timeline, overlay

    def _set_baseline(state: UIState) -> UIState:
        if state.emb_hist:
            state.baseline = np.mean(np.stack(state.emb_hist, axis=0), axis=0)
        return state

    def _process_chunk(audio: Any, state: UIState) -> tuple[UIState, list[list[Any]], np.ndarray, np.ndarray | None]:
        if state is None:
            state = UIState()

        if audio is None:
            return (
                state,
                _format_topk(label_list, np.zeros(len(label_list), dtype=np.float32), top_k),
                _render_conf_timeline(state.conf_t),
                projection_static_img,
            )

        # gradio Audio(type="numpy") returns (sr, data)
        sr, y = audio
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 2:
            y = y.mean(axis=1)

        if y.size == 0:
            return (
                state,
                _format_topk(label_list, np.zeros(len(label_list), dtype=np.float32), top_k),
                _render_conf_timeline(state.conf_t),
                projection_static_img,
            )

        # Normalize to [-1, 1] if int-like
        if y.dtype.kind in {"i", "u"}:
            y = y.astype(np.float32) / max(1.0, float(np.iinfo(y.dtype).max))

        # Resample path expects embedder.sample_rate; we use a lightweight in-place resample via scipy if needed.
        if int(sr) != int(embedder.sample_rate):
            from scipy.signal import resample_poly

            sr = int(sr)
            target_sr = int(embedder.sample_rate)
            g = np.gcd(sr, target_sr)
            y = resample_poly(y, target_sr // g, sr // g).astype(np.float32, copy=False)
            sr = target_sr

        win_n = max(256, int(chunk_sec * float(sr)))
        proba = np.zeros(len(label_list), dtype=np.float32)
        for start in range(0, y.size, win_n):
            chunk = y[start : start + win_n]
            if chunk.size < int(0.25 * sr):
                continue
            rms = float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64)))
            state.t_sec += float(chunk.size) / float(sr)
            if rms < vad_thr:
                continue

            emb = embedder.embed(y=chunk, sr=int(sr)).astype(np.float32, copy=False)
            if state.baseline is not None and state.baseline.shape == emb.shape:
                emb = emb - state.baseline

            state.emb_hist.append(emb)
            if len(state.emb_hist) > history_chunks:
                state.emb_hist = state.emb_hist[-history_chunks:]

            emb_mean = np.mean(np.stack(state.emb_hist, axis=0), axis=0, dtype=np.float64).astype(np.float32)
            proba = model.predict_proba(emb_mean[None, :])[0].astype(np.float32)
            state.conf_t.append((state.t_sec, float(np.max(proba))))

        top_rows = _format_topk(label_list, proba, top_k)
        timeline = _render_conf_timeline(state.conf_t)

        overlay = projection_static_img
        if proj_bg is not None and state.emb_hist:
            try:
                emb_mean = np.mean(np.stack(state.emb_hist, axis=0), axis=0, dtype=np.float64).astype(np.float32)
                point2 = None
                if umap_model is not None and method_used == "umap":
                    point2 = umap_model.transform(emb_mean[None, :])[0]
                elif proj_ref is not None:
                    X_ref, X2_ref, _ = proj_ref
                    point2 = _knn_map_to_2d(emb_mean, X_ref=X_ref, X2_ref=X2_ref, k=10)

                if point2 is not None:
                    X2_bg, y_bg = proj_bg
                    overlay = _render_embedding_overlay(
                        X2_bg=X2_bg,
                        y_bg=y_bg,
                        labels=label_list,
                        point2=point2,
                        title="Live point on 2D projection",
                    )
            except Exception:
                overlay = projection_static_img

        return state, top_rows, timeline, overlay

    with gr.Blocks(title="DialectSense Demo") as demo:
        gr.Markdown("# DialectSense: Province Dialect Identification (Demo)")
        gr.Markdown(
            f"- Artifacts: `{artifact_dir}`\n"
            f"- Model: `{model_path}`\n"
            f"- Embedder: `{embed_cfg.get('backend')}` (dim={embedder.dim()})\n"
            f"- Chunk: {chunk_sec}s, history: {history_chunks} chunks, VAD RMS>={vad_thr}"
        )

        state = gr.State(UIState())
        with gr.Row():
            audio = gr.Audio(sources=["microphone", "upload"], type="numpy", streaming=True, label="Microphone / Upload")
        with gr.Row():
            pred_table = gr.Dataframe(headers=["rank", "province", "confidence"], datatype=["number", "str", "number"], interactive=False)
        with gr.Row():
            timeline_img = gr.Image(label="Confidence timeline", type="numpy")
        with gr.Row():
            overlay_img = gr.Image(label="2D projection (optional)", type="numpy")

        with gr.Row():
            reset_btn = gr.Button("Reset")
            baseline_btn = gr.Button("Set baseline (optional)")

        reset_btn.click(fn=_reset, outputs=[state, pred_table, timeline_img, overlay_img])
        baseline_btn.click(fn=_set_baseline, inputs=[state], outputs=[state])

        # Streaming updates (best-effort). If streaming is unsupported, users can record a short clip and it will be processed once.
        audio.stream(fn=_process_chunk, inputs=[audio, state], outputs=[state, pred_table, timeline_img, overlay_img])
        audio.change(fn=_process_chunk, inputs=[audio, state], outputs=[state, pred_table, timeline_img, overlay_img])

        demo.load(fn=_reset, outputs=[state, pred_table, timeline_img, overlay_img])

    demo.queue()
    demo.launch()

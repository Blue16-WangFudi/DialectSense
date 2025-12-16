from __future__ import annotations

import csv
import io
import json
import logging
import os
import shutil
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..config import load_config, resolve_paths
from ..pipeline import (
    ensure_coarsen,
    ensure_embed,
    ensure_eval,
    ensure_preprocess,
    ensure_report,
    ensure_split,
    ensure_train,
    prepare_run,
)
from ..step_coarsen import run_coarsen
from ..step_embed import run_embed
from ..step_eval import run_eval
from ..step_preprocess import run_preprocess
from ..step_report import run_report
from ..step_split import run_split
from ..step_train import run_train
from ..util import ensure_dir
from ..infer import CoarsePredictor
from ..streaming import AudioStreamChunker


log = logging.getLogger("dialectsense")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_local_ffmpeg_on_path() -> None:
    repo = _repo_root()
    local = repo / ".cache" / "ffmpeg" / "bin"
    if local.exists():
        os.environ["PATH"] = f"{local}{os.pathsep}{os.environ.get('PATH','')}"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _list_pngs(fig_dir: Path) -> list[tuple[str, str]]:
    if not fig_dir.exists():
        return []
    items = sorted(fig_dir.glob("*.png"))
    # gr.Gallery expects (image, caption) tuples when using tuples.
    return [(str(p), p.name) for p in items]


def _read_csv_rows(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as f:
        dr = csv.DictReader(f)
        for i, r in enumerate(dr):
            if r is None:
                continue
            out.append({k: (v or "") for k, v in r.items()})
            if limit is not None and (i + 1) >= int(limit):
                break
    return out


def _df_from_dict_rows(rows: list[dict[str, Any]], max_cols: int | None = None) -> tuple[list[str], list[list[Any]]]:
    if not rows:
        return [], []
    cols: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in cols:
                cols.append(k)
    if max_cols is not None:
        cols = cols[: int(max_cols)]
    data = [[r.get(c, "") for c in cols] for r in rows]
    return cols, data


@dataclass(frozen=True)
class RunContext:
    config_path: str
    cfg: dict[str, Any]
    artifact_dir: Path


def _load_ctx(config_path: str) -> RunContext:
    cfg = load_config(config_path)
    artifact_dir = resolve_paths(cfg).artifact_dir
    prepare_run(cfg, artifact_dir)
    return RunContext(config_path=config_path, cfg=cfg, artifact_dir=artifact_dir)


def _capture_dialectsense_logs(fn) -> tuple[Any, str]:
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setLevel(logging.INFO)
    logger = logging.getLogger("dialectsense")
    logger.addHandler(h)
    try:
        ret = fn()
        return ret, buf.getvalue()
    except Exception:
        buf.write("\n" + traceback.format_exc())
        raise
    finally:
        logger.removeHandler(h)


def _stage_runner(stage: str, config_path: str) -> tuple[str, str]:
    """
    Returns: (status_markdown, log_text)
    """
    _ensure_local_ffmpeg_on_path()
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found. Run `python bootstrap_ffmpeg.py` or install ffmpeg on PATH.")

    ctx = _load_ctx(config_path)
    t0 = time.time()

    def _run() -> None:
        if stage == "preprocess":
            run_preprocess(ctx.cfg, ctx.artifact_dir)
        elif stage == "embed":
            ensure_preprocess(ctx.cfg, ctx.artifact_dir)
            run_embed(ctx.cfg, ctx.artifact_dir)
        elif stage == "split":
            ensure_preprocess(ctx.cfg, ctx.artifact_dir)
            run_split(ctx.cfg, ctx.artifact_dir)
        elif stage == "coarsen":
            ensure_split(ctx.cfg, ctx.artifact_dir)
            ensure_embed(ctx.cfg, ctx.artifact_dir)
            run_coarsen(ctx.cfg, ctx.artifact_dir)
        elif stage == "train":
            ensure_coarsen(ctx.cfg, ctx.artifact_dir)
            run_train(ctx.cfg, ctx.artifact_dir)
        elif stage == "eval":
            ensure_train(ctx.cfg, ctx.artifact_dir)
            run_eval(ctx.cfg, ctx.artifact_dir)
        elif stage == "report":
            ensure_eval(ctx.cfg, ctx.artifact_dir)
            run_report(ctx.cfg, ctx.artifact_dir)
        elif stage == "all":
            ensure_report(ctx.cfg, ctx.artifact_dir)
        else:
            raise ValueError(f"Unknown stage: {stage}")

    _, logs = _capture_dialectsense_logs(_run)
    dt = time.time() - t0
    status = f"Done: `{stage}` in {dt:.1f}s. Artifacts: `{ctx.artifact_dir}`"
    return status, logs


def _load_results(config_path: str) -> tuple[str, dict[str, Any], str, tuple[list[str], list[list[Any]]], list[tuple[str, str]], str]:
    ctx = _load_ctx(config_path)
    art = ctx.artifact_dir

    overview = [
        f"- Config: `{ctx.config_path}`",
        f"- Artifacts: `{art}`",
        f"- audio_qc.csv: {'✅' if (art/'audio_qc.csv').exists() else '❌'}",
        f"- splits.csv: {'✅' if (art/'splits.csv').exists() else '❌'}",
        f"- label_to_cluster.json: {'✅' if (art/'label_to_cluster.json').exists() else '❌'}",
        f"- models/coarse_model.joblib: {'✅' if (art/'models'/'coarse_model.joblib').exists() else '❌'}",
        f"- report_coarse.json: {'✅' if (art/'report_coarse.json').exists() else '❌'}",
    ]
    overview_md = "\n".join(overview)

    metrics: dict[str, Any] = {}
    report_path = art / "report_coarse.json"
    if report_path.exists():
        metrics = _read_json(report_path)

    cluster_md = ""
    cluster_path = art / "cluster_summary.md"
    if cluster_path.exists():
        cluster_md = _read_text(cluster_path)

    conf_rows = _read_csv_rows(art / "top_confusions.csv", limit=200)
    conf_headers, conf_data = _df_from_dict_rows(conf_rows)

    figs = _list_pngs(art / "figures")

    warn = ""
    if not shutil.which("ffmpeg"):
        warn = "Warning: `ffmpeg` not found on PATH; audio preprocessing/prediction will fail."
    return overview_md, metrics, cluster_md, (conf_headers, conf_data), figs, warn


def _predict_one(
    config_path: str,
    audio: Any,
    top_k: int,
    timeline: bool,
) -> tuple[str, list[list[Any]], np.ndarray | None]:
    if audio is None:
        return "No audio provided.", [], None

    _ensure_local_ffmpeg_on_path()

    ctx = _load_ctx(config_path)
    art = ctx.artifact_dir
    ensure_train(ctx.cfg, art)
    ensure_coarsen(ctx.cfg, art)

    predictor = _get_predictor(config_path)
    model = predictor.model

    cluster_to_labels: dict[str, list[str]] = {}
    c2l_path = art / "cluster_to_labels.json"
    if c2l_path.exists():
        cluster_to_labels = _read_json(c2l_path)

    embed_cfg = ctx.cfg.get("embed", {}) if isinstance(ctx.cfg.get("embed"), dict) else {}
    chunk_cfg = embed_cfg.get("chunk", {}) if isinstance(embed_cfg.get("chunk"), dict) else {}
    chunks = []
    emb = None

    # Preferred: use numpy audio (avoids Gradio file-cache restrictions).
    if isinstance(audio, tuple) and len(audio) == 2:
        sr, y = audio
        sr, y = predictor.preprocess_audio_array(y, sr=int(sr))
        chunks = predictor.embedder.embed_audio_chunks(y, sr=sr, chunk_cfg=chunk_cfg)
        emb = predictor.embedder.embed_audio_chunked(y, sr=sr, chunk_cfg=chunk_cfg).embedding
    else:
        # Fallback: treat as filepath.
        audio_path = str(audio)
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg not found. Run `python bootstrap_ffmpeg.py` or install ffmpeg on PATH.")
        tmp_dir = ensure_dir(art / "tmp" / "ui_predict")
        out_wav = tmp_dir / f"input_{uuid.uuid4().hex}.wav"
        from ..audio_preprocess import preprocess_audio

        audio_cfg = ctx.cfg.get("audio", {}) if isinstance(ctx.cfg.get("audio"), dict) else {}
        preprocess_audio(audio_path, out_wav, audio_cfg=audio_cfg)
        chunks = predictor.embedder.embed_wav_path_chunks(out_wav, chunk_cfg=chunk_cfg)
        emb = predictor.embedder.embed_wav_path(out_wav, chunk_cfg=chunk_cfg).embedding

    if emb is None:
        return "No audio provided.", [], None

    proba = model.predict_proba(emb.reshape(1, -1))[0].astype(np.float64)
    classes = getattr(model, "classes_", None)
    if classes is None:
        classes = np.arange(len(proba), dtype=int)
    classes = np.asarray(classes, dtype=int)

    idx = np.argsort(-proba)[: int(top_k)]
    rows: list[list[Any]] = []
    best_cluster = int(classes[idx[0]])
    for rank, j in enumerate(idx.tolist(), start=1):
        cid = int(classes[j])
        rows.append([rank, str(cid), float(proba[j])])

    detail = cluster_to_labels.get(str(best_cluster), [])
    md = (
        f"Predicted coarse cluster: `{best_cluster}`\n\n"
        f"- Chunks used: {len(chunks)}\n"
        f"- Cluster labels: {', '.join(detail) if detail else '(missing cluster_to_labels.json)'}"
    )

    timeline_img: np.ndarray | None = None
    if timeline and chunks:
        t_end = [c.end_sec for c in chunks]
        top1 = []
        for c in chunks:
            p = model.predict_proba(c.embedding.reshape(1, -1))[0].astype(np.float64)
            top1.append(float(np.max(p)))

        fig, ax = plt.subplots(figsize=(7, 2.6))
        ax.plot(t_end, top1, linewidth=2)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("top1 prob")
        ax.set_title("Chunk-level top1 confidence")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160)
        plt.close(fig)
        buf.seek(0)
        timeline_img = plt.imread(buf)

    return md, rows, timeline_img


def _predict_one_safe(
    config_path: str,
    audio: Any,
    top_k: int,
    timeline: bool,
) -> tuple[str, list[list[Any]], np.ndarray | None]:
    try:
        return _predict_one(config_path, audio, top_k, timeline)
    except Exception:
        return ("```text\n" + traceback.format_exc() + "\n```"), [], None


_PREDICTOR_CACHE: dict[str, tuple[float, CoarsePredictor]] = {}


def _get_predictor(config_path: str) -> CoarsePredictor:
    cfg_path = str(Path(config_path))
    ctx = _load_ctx(cfg_path)
    model_path = ctx.artifact_dir / "models" / "coarse_model.joblib"
    mtime = float(model_path.stat().st_mtime) if model_path.exists() else 0.0
    cached = _PREDICTOR_CACHE.get(cfg_path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    pred = CoarsePredictor.from_artifacts(ctx.artifact_dir, cfg=ctx.cfg)
    _PREDICTOR_CACHE[cfg_path] = (mtime, pred)
    return pred


@dataclass
class RealtimeState:
    last_input_len: int | None = None
    chunker: AudioStreamChunker | None = None
    times: list[float] = None  # type: ignore[assignment]
    probas: list[list[float]] = None  # type: ignore[assignment]
    classes: list[int] = None  # type: ignore[assignment]
    ema: list[float] | None = None

    def __post_init__(self) -> None:
        if self.times is None:
            self.times = []
        if self.probas is None:
            self.probas = []
        if self.classes is None:
            self.classes = []


def _format_topk_clusters(
    cluster_to_labels: dict[str, list[str]], classes: np.ndarray, proba: np.ndarray, k: int
) -> list[list[Any]]:
    k = int(max(1, k))
    idx = np.argsort(-proba)[:k]
    rows: list[list[Any]] = []
    for rank, j in enumerate(idx.tolist(), start=1):
        cid = int(classes[j])
        detail = cluster_to_labels.get(str(cid), [])
        rows.append([rank, str(cid), float(proba[j]), "、".join(detail[:6]) + ("…" if len(detail) > 6 else "")])
    return rows


def _render_proba_lines(times: list[float], probas: list[list[float]], classes: list[int]) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(8.2, 3.1))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("confidence")
    ax.set_title("Per-label confidence over time (fixed chunks)")
    ax.grid(True, alpha=0.3)

    if times and probas and classes:
        t = np.asarray(times, dtype=np.float64)
        P = np.asarray(probas, dtype=np.float64)  # [T, C]
        cmap = plt.get_cmap("tab20")
        for j, cid in enumerate(classes):
            ax.plot(t, P[:, j], linewidth=1.8, alpha=0.9, color=cmap(j % 20), label=str(cid))
        ax.legend(ncol=6, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


def _rt_reset(cfg_path: str, chunk_sec: float, hop_sec: float) -> tuple[RealtimeState, str, list[list[Any]], np.ndarray]:
    pred = _get_predictor(cfg_path)
    state = RealtimeState()
    state.chunker = AudioStreamChunker(sr=int(pred.target_sr), chunk_sec=float(chunk_sec), hop_sec=float(hop_sec))
    state.classes = [int(c) for c in pred.classes.tolist()]
    state.times = []
    state.probas = []
    state.ema = None
    status = "Ready. Start speaking; the chart updates after each fixed-length chunk."
    table = [[1, "-", 0.0, "-"]]
    img = _render_proba_lines([], [], state.classes)
    return state, status, table, img


def _rt_step(
    audio: Any,
    cfg_path: str,
    state: RealtimeState,
    chunk_sec: float,
    hop_sec: float,
    top_k: int,
    max_points: int,
    ema_alpha: float,
) -> tuple[RealtimeState, str, list[list[Any]], np.ndarray]:
    if state is None:
        state = RealtimeState()

    pred = _get_predictor(cfg_path)
    if state.chunker is None or abs(float(state.chunker.chunk_sec) - float(chunk_sec)) > 1e-6 or abs(
        float(state.chunker.hop_sec) - float(hop_sec)
    ) > 1e-6:
        state.chunker = AudioStreamChunker(sr=int(pred.target_sr), chunk_sec=float(chunk_sec), hop_sec=float(hop_sec))
        state.last_input_len = None
        state.times = []
        state.probas = []
        state.ema = None

    if audio is None:
        img = _render_proba_lines(state.times, state.probas, state.classes)
        return state, "Waiting for audio…", [[1, "-", 0.0, "-"]], img

    sr_in, y_in = audio
    y_in = np.asarray(y_in)
    if y_in.size == 0:
        img = _render_proba_lines(state.times, state.probas, state.classes)
        return state, "Waiting for audio…", [[1, "-", 0.0, "-"]], img

    # Gradio streaming may provide cumulative audio or per-chunk audio; handle both.
    y_mono = y_in.mean(axis=1) if y_in.ndim == 2 else y_in
    y_mono = np.asarray(y_mono, dtype=np.float32).reshape(-1)
    if state.last_input_len is None:
        y_delta = y_mono
    else:
        if y_mono.size > state.last_input_len:
            y_delta = y_mono[state.last_input_len :]
        else:
            y_delta = y_mono
    state.last_input_len = int(y_mono.size)

    if y_delta.size == 0:
        img = _render_proba_lines(state.times, state.probas, state.classes)
        return state, "Listening…", [[1, "-", 0.0, "-"]], img

    updates = pred.stream_predict(state.chunker, y_new=y_delta, sr=int(sr_in))
    if not updates:
        img = _render_proba_lines(state.times, state.probas, state.classes)
        return state, "Listening… (buffering until next chunk)", [[1, "-", 0.0, "-"]], img

    max_points = int(max(1, max_points))
    alpha = float(np.clip(float(ema_alpha), 0.0, 1.0))

    last_classes = None
    last_proba = None
    last_t = None
    for t_end, classes, proba in updates:
        last_t = float(t_end)
        last_classes = classes
        last_proba = proba.astype(np.float64, copy=False)
        if state.ema is None or len(state.ema) != int(last_proba.size):
            state.ema = last_proba.tolist()
        else:
            ema = np.asarray(state.ema, dtype=np.float64)
            ema = (1.0 - alpha) * ema + alpha * last_proba
            state.ema = ema.tolist()

        state.times.append(last_t)
        state.probas.append(state.ema)

    if len(state.times) > max_points:
        state.times = state.times[-max_points:]
        state.probas = state.probas[-max_points:]

    classes_list = [int(c) for c in (last_classes.tolist() if last_classes is not None else pred.classes.tolist())]
    state.classes = classes_list

    proba_now = np.asarray(state.probas[-1], dtype=np.float64)
    classes_now = np.asarray(state.classes, dtype=int)
    top_rows = _format_topk_clusters(pred.cluster_to_labels, classes_now, proba_now, int(top_k))

    best = int(classes_now[int(np.argmax(proba_now))])
    best_detail = pred.cluster_to_labels.get(str(best), [])
    status = f"Top1: `{best}` (conf={float(np.max(proba_now)):.3f}) · {('、'.join(best_detail[:6]) + ('…' if len(best_detail)>6 else '')) if best_detail else ''}"
    img = _render_proba_lines(state.times, state.probas, state.classes)
    return state, status, top_rows, img


def _rt_step_safe(
    audio: Any,
    cfg_path: str,
    state: RealtimeState,
    chunk_sec: float,
    hop_sec: float,
    top_k: int,
    max_points: int,
    ema_alpha: float,
) -> tuple[RealtimeState, str, list[list[Any]], np.ndarray]:
    try:
        return _rt_step(audio, cfg_path, state, chunk_sec, hop_sec, top_k, max_points, ema_alpha)
    except Exception:
        if state is None:
            state = RealtimeState()
        status = "```text\n" + traceback.format_exc() + "\n```"
        img = _render_proba_lines(state.times, state.probas, state.classes)
        return state, status, [], img


def launch(default_config_path: str = "configs/smoke.json") -> None:
    _ensure_local_ffmpeg_on_path()
    import gradio as gr
    import inspect

    repo = _repo_root()
    # IMPORTANT: keep Gradio upload/cache dir stable across config reloads.
    # This avoids errors like: "Cannot move ... because it was not uploaded by a user."
    gradio_tmp = Path(os.environ.get("GRADIO_TEMP_DIR") or (repo / ".cache" / "gradio")).resolve()
    gradio_tmp.mkdir(parents=True, exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = str(gradio_tmp)
    # Align Python tempdir to avoid surprises in other libs.
    tempfile.tempdir = str(gradio_tmp)

    config_choices = []
    for p in [repo / "configs" / "smoke.json", repo / "configs" / "full.json"]:
        if p.exists():
            config_choices.append(str(p))
    if str(Path(default_config_path)) not in config_choices and Path(default_config_path).exists():
        config_choices.insert(0, str(Path(default_config_path)))

    with gr.Blocks(title="DialectSense UI") as demo:
        gr.Markdown("# DialectSense: Training/Eval Explorer + Prediction UI")

        with gr.Row():
            config_path = gr.Dropdown(
                choices=config_choices,
                value=str(Path(default_config_path)) if Path(default_config_path).exists() else (config_choices[0] if config_choices else ""),
                label="Config file",
                allow_custom_value=True,
            )
            reload_btn = gr.Button("Reload artifacts")

        warn = gr.Markdown()
        overview = gr.Markdown()

        with gr.Tabs():
            with gr.Tab("Run"):
                gr.Markdown("Run stages for the selected config (writes to `artifacts/<run_name>/`).")
                with gr.Row():
                    run_all = gr.Button("Run All (cached)", variant="primary")
                    b_pre = gr.Button("preprocess")
                    b_emb = gr.Button("embed")
                    b_split = gr.Button("split")
                    b_coarsen = gr.Button("coarsen")
                    b_train = gr.Button("train")
                    b_eval = gr.Button("eval")
                    b_rep = gr.Button("report")
                status = gr.Markdown()
                logs = gr.Textbox(label="Logs", lines=18)

            with gr.Tab("Results"):
                metrics = gr.JSON(label="report_coarse.json")
                cluster_md = gr.Markdown(label="cluster_summary.md")
                conf_pair = gr.State(([], []))
                conf_table = gr.Dataframe(
                    headers=["true", "pred", "count", "rate"],
                    datatype=["str", "str", "number", "number"],
                    interactive=False,
                    label="top_confusions.csv",
                )

            with gr.Tab("Figures"):
                figs = gr.Gallery(label="artifacts/<run>/figures/*.png", columns=3, height=560, format="png")

            with gr.Tab("Audio QC"):
                with gr.Row():
                    qc_limit = gr.Slider(50, 2000, value=300, step=50, label="Rows to show")
                    kept_only = gr.Checkbox(value=False, label="kept only")
                qc_table = gr.Dataframe(interactive=False, label="audio_qc.csv (sample)")

            with gr.Tab("Predict"):
                gr.Markdown("Upload/record audio to run WavLM-Large → coarse model prediction.")
                with gr.Row():
                    audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Audio input")
                with gr.Row():
                    topk = gr.Slider(1, 10, value=5, step=1, label="Top-K")
                    show_timeline = gr.Checkbox(value=True, label="Chunk timeline (slow)")
                    pred_btn = gr.Button("Predict", variant="primary")
                pred_md = gr.Markdown()
                pred_table = gr.Dataframe(headers=["rank", "cluster_id", "prob"], interactive=False)
                pred_timeline = gr.Image(label="Timeline", type="numpy")

            with gr.Tab("Realtime"):
                gr.Markdown(
                    "Realtime fixed-chunk inference from microphone streaming. "
                    "Shows per-cluster confidence over time (line chart)."
                )
                rt_state = gr.State(RealtimeState())
                with gr.Row():
                    rt_chunk = gr.Slider(0.8, 6.0, value=3.0, step=0.1, label="Chunk length (sec)")
                    rt_hop = gr.Slider(0.2, 3.0, value=1.0, step=0.1, label="Hop (sec)")
                    rt_points = gr.Slider(10, 240, value=60, step=5, label="History points")
                with gr.Row():
                    rt_topk = gr.Slider(1, 12, value=5, step=1, label="Top-K table")
                    rt_ema = gr.Slider(0.0, 1.0, value=0.35, step=0.05, label="EMA smoothing α (0=off)")
                    rt_reset_btn = gr.Button("Reset", variant="secondary")
                rt_audio = gr.Audio(sources=["microphone"], type="numpy", streaming=True, label="Microphone (streaming)")
                rt_status = gr.Markdown()
                rt_table = gr.Dataframe(headers=["rank", "cluster_id", "conf", "cluster_labels"], interactive=False)
                rt_chart = gr.Image(label="Confidence lines", type="numpy")

        def _do_reload(cfg_path: str):
            return _load_results(cfg_path)

        def _do_qc(cfg_path: str, limit: int, kept: bool):
            ctx = _load_ctx(cfg_path)
            rows = _read_csv_rows(ctx.artifact_dir / "audio_qc.csv", limit=int(limit))
            if kept:
                rows = [r for r in rows if (r.get("kept") or "") == "1"]
            headers, data = _df_from_dict_rows(rows, max_cols=20)
            return gr.update(headers=headers, value=data)

        reload_btn.click(
            fn=_do_reload,
            inputs=[config_path],
            outputs=[overview, metrics, cluster_md, conf_pair, figs, warn],
        )

        def _set_conf_table(conf_pair_value):
            headers, data = conf_pair_value
            return gr.update(headers=headers, value=data)

        reload_btn.click(fn=_set_conf_table, inputs=[conf_pair], outputs=[conf_table])

        qc_limit.change(fn=_do_qc, inputs=[config_path, qc_limit, kept_only], outputs=[qc_table])
        kept_only.change(fn=_do_qc, inputs=[config_path, qc_limit, kept_only], outputs=[qc_table])

        def _run_stage(stage: str, cfg_path: str):
            s, l = _stage_runner(stage, cfg_path)
            return s, l

        for btn, stage in [
            (run_all, "all"),
            (b_pre, "preprocess"),
            (b_emb, "embed"),
            (b_split, "split"),
            (b_coarsen, "coarsen"),
            (b_train, "train"),
            (b_eval, "eval"),
            (b_rep, "report"),
        ]:
            btn.click(fn=lambda cfg_path, st=stage: _run_stage(st, cfg_path), inputs=[config_path], outputs=[status, logs])
            btn.click(fn=_do_reload, inputs=[config_path], outputs=[overview, metrics, cluster_md, conf_pair, figs, warn])
            btn.click(fn=_set_conf_table, inputs=[conf_pair], outputs=[conf_table])

        pred_btn.click(
            fn=_predict_one_safe,
            inputs=[config_path, audio_in, topk, show_timeline],
            outputs=[pred_md, pred_table, pred_timeline],
            time_limit=1800,
        )

        rt_reset_btn.click(fn=_rt_reset, inputs=[config_path, rt_chunk, rt_hop], outputs=[rt_state, rt_status, rt_table, rt_chart])
        rt_audio.stream(
            fn=_rt_step_safe,
            inputs=[rt_audio, config_path, rt_state, rt_chunk, rt_hop, rt_topk, rt_points, rt_ema],
            outputs=[rt_state, rt_status, rt_table, rt_chart],
            stream_every=1.2,
            trigger_mode="always_last",
            time_limit=1800,
        )

        demo.load(fn=_do_reload, inputs=[config_path], outputs=[overview, metrics, cluster_md, conf_pair, figs, warn])
        demo.load(fn=_set_conf_table, inputs=[conf_pair], outputs=[conf_table])
        demo.load(fn=_do_qc, inputs=[config_path, qc_limit, kept_only], outputs=[qc_table])
        demo.load(fn=_rt_reset, inputs=[config_path, rt_chunk, rt_hop], outputs=[rt_state, rt_status, rt_table, rt_chart])

    queue_sig = inspect.signature(demo.queue)
    if "concurrency_count" in queue_sig.parameters:
        demo.queue(concurrency_count=1)
    elif "default_concurrency_limit" in queue_sig.parameters:
        demo.queue(default_concurrency_limit=1)
    else:
        demo.queue()
    demo.launch(allowed_paths=[str(repo)], show_error=True)

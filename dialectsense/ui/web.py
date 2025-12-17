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

def _configure_matplotlib_fonts() -> None:
    # Best-effort CJK font setup so plot legends can display Chinese labels.
    try:
        from matplotlib import font_manager

        candidates = [
            "Noto Sans CJK SC",
            "Source Han Sans SC",
            "WenQuanYi Zen Hei",
            "SimHei",
            "Microsoft YaHei",
            "Arial Unicode MS",
        ]
        available = {f.name for f in font_manager.fontManager.ttflist}
        for name in candidates:
            if name in available:
                cur = list(matplotlib.rcParams.get("font.sans-serif", []))
                matplotlib.rcParams["font.sans-serif"] = [name, *cur]
                matplotlib.rcParams["axes.unicode_minus"] = False
                break
    except Exception:
        return


_configure_matplotlib_fonts()

try:
    # Used only for Gradio special-args injection (request.session_hash).
    from gradio.routes import Request as GradioRequest
except Exception:  # pragma: no cover
    GradioRequest = Any  # type: ignore[misc,assignment]

_RT_SESSION_CACHE: dict[str, tuple[float, "RealtimeState"]] = {}


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


def _df_value(headers: list[str], rows: list[list[Any]]) -> dict[str, Any]:
    """
    Gradio Dataframe outputs accept a dict with keys: {headers, data}.
    Returning this form avoids Gradio trying to infer/reshape unexpected types.
    """
    def _cell(v: Any) -> Any:
        try:
            if isinstance(v, np.generic):
                return v.item()
        except Exception:
            pass
        return v

    return {"headers": list(headers), "data": [[_cell(v) for v in r] for r in rows]}


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
) -> tuple[str, dict[str, Any], np.ndarray | None]:
    if audio is None:
        return "No audio provided.", _df_value(["rank", "cluster_id", "prob"], []), None

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
        return "No audio provided.", _df_value(["rank", "cluster_id", "prob"], []), None

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

    return md, _df_value(["rank", "cluster_id", "prob"], rows), timeline_img


def _predict_one_safe(
    config_path: str,
    audio: Any,
    top_k: int,
    timeline: bool,
) -> tuple[str, dict[str, Any], np.ndarray | None]:
    try:
        return _predict_one(config_path, audio, top_k, timeline)
    except Exception:
        return ("```text\n" + traceback.format_exc() + "\n```"), _df_value(["rank", "cluster_id", "prob"], []), None


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
    pending: list[tuple[float, np.ndarray]] = None  # type: ignore[assignment]
    session_id: str | None = None

    def __post_init__(self) -> None:
        if self.times is None:
            self.times = []
        if self.probas is None:
            self.probas = []
        if self.classes is None:
            self.classes = []
        if self.pending is None:
            self.pending = []


def _rt_prune_cache(now: float, max_age_sec: float = 1800.0) -> None:
    if not _RT_SESSION_CACHE:
        return
    dead = [k for k, (ts, _) in _RT_SESSION_CACHE.items() if (now - float(ts)) > float(max_age_sec)]
    for k in dead:
        _RT_SESSION_CACHE.pop(k, None)


def _rt_get_state(session_id: str | None, provided: RealtimeState | None) -> RealtimeState:
    """
    Some Gradio/proxy deployments don't reliably persist `gr.State` updates for `.stream()` events.
    To make buffering robust, we also keep a server-side per-session cache keyed by a stable session id.
    """
    now = time.time()
    _rt_prune_cache(now)
    sid = session_id or "default"
    cached = _RT_SESSION_CACHE.get(sid)
    cached_state = cached[1] if cached is not None else None

    def _score(st: RealtimeState | None) -> tuple[int, int, int]:
        if st is None:
            return (0, 0, 0)
        buf = 0
        try:
            if st.chunker is not None:
                buf = int(st.chunker.buffered_samples)
        except Exception:
            buf = 0
        t = len(st.times) if st.times is not None else 0
        p = len(st.pending) if st.pending is not None else 0
        return (buf, t, p)

    # If Gradio provides a "fresh" state each callback (buggy persistence), do NOT overwrite
    # a more-advanced cached state. Prefer whichever has progressed further.
    chosen: RealtimeState
    if cached_state is None and provided is None:
        chosen = RealtimeState()
    elif cached_state is None and provided is not None:
        chosen = provided
    elif cached_state is not None and provided is None:
        chosen = cached_state
    else:
        assert cached_state is not None and provided is not None
        chosen = provided if _score(provided) > _score(cached_state) else cached_state

    chosen.session_id = sid
    _RT_SESSION_CACHE[sid] = (now, chosen)
    return chosen


def _rt_put_state(st: RealtimeState) -> None:
    now = time.time()
    _rt_prune_cache(now)
    sid = st.session_id or "default"
    _RT_SESSION_CACHE[sid] = (now, st)


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


def _cluster_legend(cluster_to_labels: dict[str, list[str]], cid: int) -> str:
    labels = cluster_to_labels.get(str(int(cid)), [])
    if not labels:
        return str(int(cid))
    short = "、".join(labels[:2]) + ("…" if len(labels) > 2 else "")
    return f"{int(cid)}:{short}"


def _render_proba_lines(
    times: list[float],
    probas: list[list[float]],
    classes: list[int],
    plot_class_ids: list[int] | None = None,
    legend_map: dict[int, str] | None = None,
    max_lines: int = 12,
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(8.2, 3.1))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("confidence")
    ax.set_title("Per-label confidence over time (fixed chunks)")
    ax.grid(True, alpha=0.3)

    if times and probas and classes:
        t = np.asarray(times, dtype=np.float64)
        P = np.asarray(probas, dtype=np.float64)  # [T, C]
        cls = np.asarray(classes, dtype=int).reshape(-1)

        max_lines = int(min(len(cls), max(1, int(max_lines))))
        keep_idx: np.ndarray
        if plot_class_ids:
            want = [int(c) for c in plot_class_ids]
            pos = {int(c): i for i, c in enumerate(cls.tolist())}
            idx = [pos[c] for c in want if c in pos]
            if idx:
                keep_idx = np.asarray(idx[:max_lines], dtype=int)
            else:
                keep_idx = np.argsort(-P[-1])[:max_lines]
        else:
            # Default: Plot only top-N classes (by max confidence over history) for readability.
            if P.shape[1] > max_lines:
                score = P.max(axis=0)
                keep_idx = np.argsort(-score)[:max_lines]
            else:
                keep_idx = np.arange(P.shape[1])

        n_lines = int(keep_idx.size)
        use_hsv = n_lines > 20
        cmap = plt.get_cmap("hsv" if use_hsv else "tab20")
        for k, j in enumerate(keep_idx.tolist()):
            cid = int(cls[int(j)])
            if use_hsv:
                c = cmap(float(k) / float(max(1, n_lines - 1)))
            else:
                c = cmap(k % 20)
            lab = legend_map.get(cid, str(cid)) if legend_map is not None else str(cid)
            ax.plot(t, P[:, j], linewidth=1.9, alpha=0.9, color=c, label=lab)
        if n_lines <= 12:
            ax.legend(ncol=6, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


def _rt_reset(
    cfg_path: str,
    chunk_sec: float,
    hop_sec: float,
    session_id: str | None = None,
    request: Any = None,
) -> tuple[RealtimeState, str, dict[str, Any], np.ndarray]:
    pred = _get_predictor(cfg_path)
    if not session_id:
        try:
            session_id = getattr(request, "session_hash", None)
        except Exception:
            session_id = None
    state = _rt_get_state(session_id, None)
    state.chunker = AudioStreamChunker(sr=int(pred.target_sr), chunk_sec=float(chunk_sec), hop_sec=float(hop_sec))
    state.classes = [int(c) for c in pred.classes.tolist()]
    state.times = []
    state.probas = []
    state.ema = None
    state.pending = []
    state.last_input_len = None
    status = "Ready. Start speaking; the chart updates after each fixed-length chunk."
    table = _df_value(["rank", "cluster_id", "conf", "cluster_labels"], [[1, "-", 0.0, "-"]])
    legend = {int(cid): _cluster_legend(pred.cluster_to_labels, int(cid)) for cid in state.classes}
    img = _render_proba_lines([], [], state.classes, plot_class_ids=None, legend_map=legend, max_lines=6)
    _rt_put_state(state)
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
    debug: bool = False,
) -> tuple[RealtimeState, str, list[list[Any]], np.ndarray]:
    raise RuntimeError("_rt_step is deprecated; use _rt_step_safe (generator) directly.")


def _rt_step_safe(
    audio: Any,
    cfg_path: str,
    state: RealtimeState,
    chunk_sec: float,
    hop_sec: float,
    top_k: int,
    max_points: int,
    ema_alpha: float,
    session_id: str | None = None,
    debug: bool = False,
    request: Any = None,
):
    try:
        if not session_id:
            try:
                session_id = getattr(request, "session_hash", None)
            except Exception:
                session_id = None
        st = _rt_get_state(session_id, state if isinstance(state, RealtimeState) else None)
        pred = _get_predictor(cfg_path)

        # (Re)initialize chunker if params changed.
        # Be tolerant to float jitter from UI sliders (otherwise buffering can keep resetting).
        chunk_sec_r = round(float(chunk_sec), 3)
        hop_sec_r = round(float(hop_sec), 3)
        if st.chunker is None or abs(round(float(st.chunker.chunk_sec), 3) - chunk_sec_r) > 5e-3 or abs(
            round(float(st.chunker.hop_sec), 3) - hop_sec_r
        ) > 5e-3:
            st.chunker = AudioStreamChunker(sr=int(pred.target_sr), chunk_sec=float(chunk_sec), hop_sec=float(hop_sec))
            st.last_input_len = None
            st.times = []
            st.probas = []
            st.ema = None
            st.pending = []
            st.classes = [int(c) for c in pred.classes.tolist()]

        legend = {int(cid): _cluster_legend(pred.cluster_to_labels, int(cid)) for cid in (st.classes or [])}
        plot_k = int(max(1, top_k))
        plot_ids: list[int] | None = None
        if st.probas and st.classes and len(st.probas[-1]) == len(st.classes):
            p_last = np.asarray(st.probas[-1], dtype=np.float64)
            c_last = np.asarray(st.classes, dtype=int)
            idx_last = np.argsort(-p_last)[:plot_k]
            plot_ids = [int(c_last[i]) for i in idx_last.tolist()]

        if audio is None:
            img = _render_proba_lines(
                st.times, st.probas, st.classes, plot_class_ids=plot_ids, legend_map=legend, max_lines=plot_k
            )
            status = "Waiting for audio…"
            if debug:
                status += (
                    f"\n\n`debug`: session={st.session_id} audio=None pending={len(st.pending)} "
                    f"buffered={float(st.chunker.buffered_sec) if st.chunker else 0.0:.2f}s"
                )
            _rt_put_state(st)
            yield st, status, _df_value(["rank", "cluster_id", "conf", "cluster_labels"], [[1, "-", 0.0, "-"]]), img
            return

        sr_in, y_in = audio
        y_in = np.asarray(y_in)
        if y_in.size == 0:
            img = _render_proba_lines(
                st.times, st.probas, st.classes, plot_class_ids=plot_ids, legend_map=legend, max_lines=plot_k
            )
            status = "Waiting for audio…"
            if debug:
                status += f"\n\n`debug`: session={st.session_id} sr={sr_in} y.size=0 pending={len(st.pending)}"
            _rt_put_state(st)
            yield st, status, _df_value(["rank", "cluster_id", "conf", "cluster_labels"], [[1, "-", 0.0, "-"]]), img
            return

        # Gradio streaming may provide cumulative audio (growing) or per-chunk audio (fixed size).
        # Slice deltas along the *sample* axis without destroying dtype (int16 scaling happens later).
        y_arr = y_in
        sample_axis = 0
        n_samples = int(y_arr.shape[0]) if y_arr.ndim >= 1 else 0
        if y_arr.ndim == 2:
            # Accept both (samples, channels) and (channels, samples)
            if y_arr.shape[0] <= 4 and y_arr.shape[1] > y_arr.shape[0]:
                sample_axis = 1
                n_samples = int(y_arr.shape[1])
            else:
                sample_axis = 0
                n_samples = int(y_arr.shape[0])

        # New recording / stream restart: Gradio may reset the cumulative buffer length.
        # If we don't reset our own buffered state, we'll keep "buffering" forever.
        if st.last_input_len is not None:
            last_n = int(st.last_input_len)
            # Only treat as restart when the previous buffer was "meaningfully large"
            # to avoid false positives from small chunk-size jitter.
            if last_n >= 4096 and n_samples + 256 < last_n and n_samples < int(0.75 * last_n):
                st.last_input_len = None
                if st.chunker is not None:
                    st.chunker.reset()
                st.times = []
                st.probas = []
                st.ema = None
                st.pending = []

        if st.last_input_len is None:
            y_delta_raw = y_arr
            delta_samples = int(n_samples)
        else:
            if n_samples > int(st.last_input_len):
                if sample_axis == 0:
                    y_delta_raw = y_arr[int(st.last_input_len) :] if y_arr.ndim == 1 else y_arr[int(st.last_input_len) :, :]
                else:
                    y_delta_raw = y_arr[:, int(st.last_input_len) :]
                delta_samples = int(n_samples - int(st.last_input_len))
            else:
                y_delta_raw = y_arr
                delta_samples = int(n_samples)
        st.last_input_len = int(n_samples)

        if y_delta_raw.size == 0:
            img = _render_proba_lines(
                st.times, st.probas, st.classes, plot_class_ids=plot_ids, legend_map=legend, max_lines=plot_k
            )
            status = "Listening…"
            if debug:
                status += (
                    f"\n\n`debug`: session={st.session_id} sr={sr_in} recv={int(n_samples)} delta=0 "
                    f"pending={len(st.pending)} buffered={float(st.chunker.buffered_sec) if st.chunker else 0.0:.2f}s"
                )
            _rt_put_state(st)
            yield st, status, _df_value(["rank", "cluster_id", "conf", "cluster_labels"], [[1, "-", 0.0, "-"]]), img
            return

        # Push audio into chunker (after resampling to target sr). Inference can be slow; we surface a quick
        # "processing…" update before doing WavLM forward pass when a chunk becomes available.
        _, y_delta_16k = pred.preprocess_audio_array(y_delta_raw, sr=int(sr_in))
        for t_end, chunk in st.chunker.push(y_delta_16k):
            st.pending.append((t_end, chunk))
        if len(st.pending) > 20:
            st.pending = st.pending[-20:]

        if not st.pending:
            img = _render_proba_lines(
                st.times, st.probas, st.classes, plot_class_ids=plot_ids, legend_map=legend, max_lines=plot_k
            )
            buf = float(st.chunker.buffered_sec) if st.chunker is not None else 0.0
            need = float(st.chunker.chunk_sec) if st.chunker is not None else float(chunk_sec)
            status = f"Listening… (buffering: {buf:.2f}s / {need:.2f}s)"
            if debug:
                status += (
                    f"\n\n`debug`: session={st.session_id} sr={sr_in} recv={int(n_samples)} delta={int(delta_samples)} "
                    f"delta@16k={int(y_delta_16k.size)} pending={len(st.pending)}"
                )
            _rt_put_state(st)
            yield st, status, _df_value(["rank", "cluster_id", "conf", "cluster_labels"], [[1, "-", 0.0, "-"]]), img
            return

        # Yield quick progress update before heavy inference.
        t_preview = float(st.pending[0][0])
        status = f"Processing chunk ending at {t_preview:.2f}s… (CPU may be slow; first run may download WavLM)"
        if debug:
            buf = float(st.chunker.buffered_sec) if st.chunker is not None else 0.0
            status += (
                f"\n\n`debug`: session={st.session_id} sr={sr_in} recv={int(n_samples)} delta={int(delta_samples)} "
                f"pending={len(st.pending)} buffered={buf:.2f}s"
            )
        img = _render_proba_lines(
            st.times, st.probas, st.classes, plot_class_ids=plot_ids, legend_map=legend, max_lines=plot_k
        )
        _rt_put_state(st)
        yield st, status, _df_value(["rank", "cluster_id", "conf", "cluster_labels"], [[1, "-", 0.0, "-"]]), img

        # Do inference for one chunk and yield final update.
        max_points = int(max(1, max_points))
        alpha = float(np.clip(float(ema_alpha), 0.0, 1.0))

        t_end, chunk = st.pending.pop(0)
        classes, proba = pred.predict_chunk_proba(chunk, sr=int(pred.target_sr))
        proba = proba.astype(np.float64, copy=False)

        if st.ema is None or len(st.ema) != int(proba.size):
            st.ema = proba.tolist()
        else:
            ema = np.asarray(st.ema, dtype=np.float64)
            ema = (1.0 - alpha) * ema + alpha * proba
            st.ema = ema.tolist()

        st.times.append(float(t_end))
        st.probas.append(st.ema)

        if len(st.times) > max_points:
            st.times = st.times[-max_points:]
            st.probas = st.probas[-max_points:]

        st.classes = [int(c) for c in classes.tolist()]

        proba_now = np.asarray(st.probas[-1], dtype=np.float64)
        classes_now = np.asarray(st.classes, dtype=int)
        top_rows = _format_topk_clusters(pred.cluster_to_labels, classes_now, proba_now, int(top_k))
        idx_plot = np.argsort(-proba_now)[:plot_k]
        plot_ids = [int(classes_now[i]) for i in idx_plot.tolist()]
        legend = {int(cid): _cluster_legend(pred.cluster_to_labels, int(cid)) for cid in st.classes}

        best = int(classes_now[int(np.argmax(proba_now))])
        best_detail = pred.cluster_to_labels.get(str(best), [])
        status = (
            f"Top1: `{best}` (conf={float(np.max(proba_now)):.3f})"
            f" · pending_chunks={len(st.pending)}"
            f" · {('、'.join(best_detail[:6]) + ('…' if len(best_detail)>6 else '')) if best_detail else ''}"
        )
        if debug:
            buf = float(st.chunker.buffered_sec) if st.chunker is not None else 0.0
            status += (
                f"\n\n`debug`: session={st.session_id} sr={sr_in} recv={int(n_samples)} delta={int(delta_samples)} "
                f"buffered={buf:.2f}s points={len(st.times)}"
            )
        img = _render_proba_lines(
            st.times, st.probas, st.classes, plot_class_ids=plot_ids, legend_map=legend, max_lines=plot_k
        )
        _rt_put_state(st)
        yield st, status, _df_value(["rank", "cluster_id", "conf", "cluster_labels"], top_rows), img
    except Exception:
        st = state if state is not None else RealtimeState()
        status = "```text\n" + traceback.format_exc() + "\n```"
        img = _render_proba_lines(st.times, st.probas, st.classes, plot_class_ids=None, legend_map=None, max_lines=6)
        _rt_put_state(st)
        yield st, status, _df_value(["rank", "cluster_id", "conf", "cluster_labels"], []), img


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
    # Avoid slow/blocked network calls to Gradio's analytics/version check in restricted environments.
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

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
                    audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy", format="wav", label="Audio input")
                with gr.Row():
                    topk = gr.Slider(1, 10, value=5, step=1, label="Top-K")
                    show_timeline = gr.Checkbox(value=True, label="Chunk timeline (slow)")
                    warmup_btn = gr.Button("Warmup WavLM (first run may take minutes)")
                    pred_btn = gr.Button("Predict", variant="primary")
                pred_md = gr.Markdown()
                pred_table = gr.Dataframe(headers=["rank", "cluster_id", "prob"], interactive=False)
                pred_timeline = gr.Image(label="Timeline", type="numpy")

            with gr.Tab("Realtime"):
                gr.Markdown(
                    "Realtime fixed-chunk inference from microphone streaming. "
                    "Shows Top-K per-cluster confidence over time (line chart). "
                    "Tip: first run may need to download/load WavLM; click Warmup and wait."
                )
                rt_state = gr.State(RealtimeState())
                rt_sid = gr.State("")
                with gr.Row():
                    rt_chunk = gr.Slider(0.6, 6.0, value=1.5, step=0.1, label="Chunk length (sec)")
                    rt_hop = gr.Slider(0.2, 3.0, value=0.75, step=0.05, label="Hop (sec)")
                    rt_points = gr.Slider(10, 240, value=60, step=5, label="History points")
                with gr.Row():
                    rt_topk = gr.Slider(1, 12, value=5, step=1, label="Top-K table")
                    rt_ema = gr.Slider(0.0, 1.0, value=0.35, step=0.05, label="EMA smoothing α (0=off)")
                    rt_debug = gr.Checkbox(value=False, label="Debug status")
                    rt_warmup_btn = gr.Button("Warmup WavLM")
                    rt_reset_btn = gr.Button("Reset", variant="secondary")
                rt_audio = gr.Audio(sources=["microphone"], type="numpy", format="wav", streaming=True, label="Microphone (streaming)")
                rt_status = gr.Markdown()
                rt_table = gr.Dataframe(headers=["rank", "cluster_id", "conf", "cluster_labels"], interactive=False)
                rt_chart = gr.Image(label="Top-K confidence lines", type="numpy")

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

        def _warmup(cfg_path: str) -> str:
            try:
                pred = _get_predictor(cfg_path)
                # Trigger model load/download
                d = pred.embedder.dim
                return f"✅ WavLM ready (dim={d})."
            except Exception:
                return "```text\n" + traceback.format_exc() + "\n```"

        def _stable_sid(sid: str | None, request: Any) -> str:
            if sid:
                return str(sid)
            try:
                s = getattr(request, "session_hash", None)
                if s:
                    return str(s)
            except Exception:
                pass
            return str(uuid.uuid4())

        def _rt_init_wrapped(cfg_path: str, chunk_sec: float, hop_sec: float, request: GradioRequest):
            sid = _stable_sid(None, request)
            state, status, table, img = _rt_reset(cfg_path, chunk_sec, hop_sec, session_id=sid, request=request)
            return sid, state, status, table, img

        def _rt_reset_wrapped(cfg_path: str, chunk_sec: float, hop_sec: float, sid: str, request: GradioRequest):
            sid = _stable_sid(sid, request)
            state, status, table, img = _rt_reset(cfg_path, chunk_sec, hop_sec, session_id=sid, request=request)
            return sid, state, status, table, img

        def _rt_step_wrapped(
            audio: Any,
            cfg_path: str,
            state: RealtimeState,
            chunk_sec: float,
            hop_sec: float,
            top_k: int,
            max_points: int,
            ema_alpha: float,
            sid: str,
            debug: bool,
            request: GradioRequest,
        ):
            sid = _stable_sid(sid, request)
            for st, status, table, img in _rt_step_safe(
                audio,
                cfg_path,
                state,
                chunk_sec,
                hop_sec,
                top_k,
                max_points,
                ema_alpha,
                session_id=sid,
                debug=debug,
                request=request,
            ):
                yield sid, st, status, table, img

        pred_btn.click(
            fn=_predict_one_safe,
            inputs=[config_path, audio_in, topk, show_timeline],
            outputs=[pred_md, pred_table, pred_timeline],
            time_limit=1800,
        )
        warmup_btn.click(fn=_warmup, inputs=[config_path], outputs=[pred_md], time_limit=1800)

        rt_reset_btn.click(
            fn=_rt_reset_wrapped,
            inputs=[config_path, rt_chunk, rt_hop, rt_sid],
            outputs=[rt_sid, rt_state, rt_status, rt_table, rt_chart],
        )
        rt_warmup_btn.click(fn=_warmup, inputs=[config_path], outputs=[rt_status], time_limit=1800)
        rt_audio.stream(
            fn=_rt_step_wrapped,
            inputs=[rt_audio, config_path, rt_state, rt_chunk, rt_hop, rt_topk, rt_points, rt_ema, rt_sid, rt_debug],
            outputs=[rt_sid, rt_state, rt_status, rt_table, rt_chart],
            stream_every=0.1,
            trigger_mode="once",
            time_limit=1800,
        )

        demo.load(fn=_do_reload, inputs=[config_path], outputs=[overview, metrics, cluster_md, conf_pair, figs, warn])
        demo.load(fn=_set_conf_table, inputs=[conf_pair], outputs=[conf_table])
        demo.load(fn=_do_qc, inputs=[config_path, qc_limit, kept_only], outputs=[qc_table])
        demo.load(
            fn=_rt_init_wrapped,
            inputs=[config_path, rt_chunk, rt_hop],
            outputs=[rt_sid, rt_state, rt_status, rt_table, rt_chart],
        )

    queue_sig = inspect.signature(demo.queue)
    if "concurrency_count" in queue_sig.parameters:
        demo.queue(concurrency_count=1)
    elif "default_concurrency_limit" in queue_sig.parameters:
        demo.queue(default_concurrency_limit=1)
    else:
        demo.queue()
    demo.launch(allowed_paths=[str(repo)], show_error=True)

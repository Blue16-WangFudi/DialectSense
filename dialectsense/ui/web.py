from __future__ import annotations

import csv
import io
import json
import logging
import os
import shutil
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

from ..audio_preprocess import preprocess_audio
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
from ..wavlm import WavLMEmbedder


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
    return [(p.name, str(p)) for p in items]


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
        f"- models/coarse_svm.joblib: {'✅' if (art/'models'/'coarse_svm.joblib').exists() else '❌'}",
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
    audio_path: str | None,
    top_k: int,
    timeline: bool,
) -> tuple[str, list[list[Any]], np.ndarray | None]:
    if not audio_path:
        return "No audio provided.", ([], []), None

    _ensure_local_ffmpeg_on_path()
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found. Run `python bootstrap_ffmpeg.py` or install ffmpeg on PATH.")

    ctx = _load_ctx(config_path)
    art = ctx.artifact_dir
    ensure_train(ctx.cfg, art)
    ensure_coarsen(ctx.cfg, art)

    model_path = art / "models" / "coarse_svm.joblib"
    bundle = __import__("joblib").load(model_path)
    model = bundle["model"]

    cluster_to_labels: dict[str, list[str]] = {}
    c2l_path = art / "cluster_to_labels.json"
    if c2l_path.exists():
        cluster_to_labels = _read_json(c2l_path)

    audio_cfg = ctx.cfg.get("audio", {}) if isinstance(ctx.cfg.get("audio"), dict) else {}
    embed_cfg = ctx.cfg.get("embed", {}) if isinstance(ctx.cfg.get("embed"), dict) else {}
    chunk_cfg = embed_cfg.get("chunk", {}) if isinstance(embed_cfg.get("chunk"), dict) else {}

    tmp_dir = ensure_dir(art / "tmp" / "ui_predict")
    out_wav = tmp_dir / f"input_{uuid.uuid4().hex}.wav"
    preprocess_audio(audio_path, out_wav, audio_cfg=audio_cfg)

    embedder = WavLMEmbedder(embed_cfg)
    chunks = embedder.embed_wav_path_chunks(out_wav, chunk_cfg=chunk_cfg)
    emb = embedder.embed_wav_path(out_wav, chunk_cfg=chunk_cfg).embedding

    proba = model.predict_proba(emb.reshape(1, -1))[0].astype(np.float64)
    # Try to recover class order.
    classes = None
    if hasattr(model, "classes_"):
        try:
            classes = np.asarray(getattr(model, "classes_"), dtype=int)
        except Exception:
            classes = None
    if classes is None and hasattr(model, "calibrator"):
        try:
            classes = np.asarray(model.calibrator.classes_, dtype=int)
        except Exception:
            classes = None
    if classes is None:
        classes = np.arange(len(proba), dtype=int)

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


def launch(default_config_path: str = "configs/smoke.json") -> None:
    _ensure_local_ffmpeg_on_path()
    import gradio as gr

    repo = _repo_root()
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
                figs = gr.Gallery(label="artifacts/<run>/figures/*.png", columns=3, height=560)

            with gr.Tab("Audio QC"):
                with gr.Row():
                    qc_limit = gr.Slider(50, 2000, value=300, step=50, label="Rows to show")
                    kept_only = gr.Checkbox(value=False, label="kept only")
                qc_table = gr.Dataframe(interactive=False, label="audio_qc.csv (sample)")

            with gr.Tab("Predict"):
                gr.Markdown("Upload/record audio to run WavLM-Large → coarse model prediction.")
                with gr.Row():
                    audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio input")
                with gr.Row():
                    topk = gr.Slider(1, 10, value=5, step=1, label="Top-K")
                    show_timeline = gr.Checkbox(value=True, label="Chunk timeline (slow)")
                    pred_btn = gr.Button("Predict", variant="primary")
                pred_md = gr.Markdown()
                pred_table = gr.Dataframe(headers=["rank", "cluster_id", "prob"], interactive=False)
                pred_timeline = gr.Image(label="Timeline", type="numpy")

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

        pred_btn.click(fn=_predict_one, inputs=[config_path, audio_in, topk, show_timeline], outputs=[pred_md, pred_table, pred_timeline])

        demo.load(fn=_do_reload, inputs=[config_path], outputs=[overview, metrics, cluster_md, conf_pair, figs, warn])
        demo.load(fn=_set_conf_table, inputs=[conf_pair], outputs=[conf_table])
        demo.load(fn=_do_qc, inputs=[config_path, qc_limit, kept_only], outputs=[qc_table])

    demo.queue(concurrency_count=1)
    demo.launch()

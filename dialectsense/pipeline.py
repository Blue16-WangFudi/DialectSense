from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import resolve_paths, save_resolved_config
from .step_coarsen import run_coarsen
from .step_embed import run_embed
from .step_eval import run_eval
from .step_preprocess import run_preprocess
from .step_report import run_report
from .step_split import run_split
from .step_train import run_train
from .util import configure_temp_dir, ensure_dir, set_global_seed


log = logging.getLogger("dialectsense")


def artifact_dir_for_cfg(cfg: dict[str, Any]) -> Path:
    return resolve_paths(cfg).artifact_dir


def prepare_run(cfg: dict[str, Any], artifact_dir: Path) -> None:
    ensure_dir(artifact_dir)
    set_global_seed(int(cfg.get("seed", 42)))
    configure_temp_dir(artifact_dir)
    save_resolved_config(cfg, artifact_dir)


def ensure_preprocess(cfg: dict[str, Any], artifact_dir: Path) -> Path:
    p = artifact_dir / "audio_qc.csv"
    if not p.exists():
        log.info("ensure: preprocess")
        run_preprocess(cfg, artifact_dir)
    return p


def ensure_embed(cfg: dict[str, Any], artifact_dir: Path) -> Path:
    ensure_preprocess(cfg, artifact_dir)
    p = artifact_dir / "embeddings" / "meta.json"
    if not p.exists():
        log.info("ensure: embed")
        run_embed(cfg, artifact_dir)
    return p


def ensure_split(cfg: dict[str, Any], artifact_dir: Path) -> Path:
    ensure_preprocess(cfg, artifact_dir)
    p = artifact_dir / "splits.csv"
    if not p.exists():
        log.info("ensure: split")
        run_split(cfg, artifact_dir)
    return p


def ensure_coarsen(cfg: dict[str, Any], artifact_dir: Path) -> Path:
    ensure_split(cfg, artifact_dir)
    ensure_embed(cfg, artifact_dir)
    p = artifact_dir / "label_to_cluster.json"
    if not p.exists():
        log.info("ensure: coarsen")
        run_coarsen(cfg, artifact_dir)
    return p


def ensure_train(cfg: dict[str, Any], artifact_dir: Path) -> Path:
    ensure_coarsen(cfg, artifact_dir)
    p = artifact_dir / "models" / "coarse_svm.joblib"
    if not p.exists():
        log.info("ensure: train")
        run_train(cfg, artifact_dir)
    return p


def ensure_eval(cfg: dict[str, Any], artifact_dir: Path) -> Path:
    ensure_train(cfg, artifact_dir)
    p = artifact_dir / "report_coarse.json"
    if not p.exists():
        log.info("ensure: eval")
        run_eval(cfg, artifact_dir)
    return p


def ensure_report(cfg: dict[str, Any], artifact_dir: Path) -> Path:
    ensure_eval(cfg, artifact_dir)
    p = artifact_dir / "report_index.json"
    if not p.exists():
        log.info("ensure: report")
        run_report(cfg, artifact_dir)
    return p


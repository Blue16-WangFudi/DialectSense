from __future__ import annotations

import logging
import json
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


def _stable_json(x: Any) -> str:
    try:
        return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return repr(x)


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
    p = artifact_dir / "models" / "coarse_model.joblib"
    need = not p.exists()
    if not need:
        try:
            import joblib

            bundle = joblib.load(p)
            requested = bundle.get("requested_model_cfg") or bundle.get("model_cfg") or bundle.get("trained_model_cfg")
            current = cfg.get("model")
            if requested is not None and current is not None:
                need = _stable_json(requested) != _stable_json(current)
        except Exception:
            need = False
    if need:
        log.info("ensure: train (rebuild)")
        run_train(cfg, artifact_dir)
    return p


def ensure_eval(cfg: dict[str, Any], artifact_dir: Path) -> Path:
    ensure_train(cfg, artifact_dir)
    p = artifact_dir / "report_coarse.json"
    model_p = artifact_dir / "models" / "coarse_model.joblib"
    need = not p.exists()
    if not need and model_p.exists():
        try:
            need = model_p.stat().st_mtime > p.stat().st_mtime
        except Exception:
            need = False
    if need:
        log.info("ensure: eval (rebuild)")
        run_eval(cfg, artifact_dir)
    return p


def ensure_report(cfg: dict[str, Any], artifact_dir: Path) -> Path:
    ensure_eval(cfg, artifact_dir)
    p = artifact_dir / "report_index.json"
    coarse_p = artifact_dir / "report_coarse.json"
    need = not p.exists()
    if not need and coarse_p.exists():
        try:
            need = coarse_p.stat().st_mtime > p.stat().st_mtime
        except Exception:
            need = False
    if need:
        log.info("ensure: report (rebuild)")
        run_report(cfg, artifact_dir)
    return p

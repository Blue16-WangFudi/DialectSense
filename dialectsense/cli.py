from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from .config import load_config, resolve_paths, save_resolved_config
from .step_coarsen import run_coarsen
from .step_embed import run_embed
from .step_eval import run_eval
from .step_preprocess import run_preprocess
from .step_split import run_split
from .step_train import run_train
from .util import configure_temp_dir, ensure_dir, set_global_seed


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dialectsense")


def _artifact_dir(cfg: dict[str, Any]) -> Path:
    return resolve_paths(cfg).artifact_dir


def _ensure_preprocess(cfg: dict[str, Any], artifact_dir: Path) -> None:
    if not (artifact_dir / "audio_qc.csv").exists():
        run_preprocess(cfg, artifact_dir)


def _ensure_embed(cfg: dict[str, Any], artifact_dir: Path) -> None:
    _ensure_preprocess(cfg, artifact_dir)
    if not (artifact_dir / "embeddings" / "meta.json").exists():
        run_embed(cfg, artifact_dir)


def _ensure_split(cfg: dict[str, Any], artifact_dir: Path) -> None:
    _ensure_preprocess(cfg, artifact_dir)
    if not (artifact_dir / "splits.csv").exists():
        run_split(cfg, artifact_dir)


def _ensure_coarsen(cfg: dict[str, Any], artifact_dir: Path) -> None:
    _ensure_split(cfg, artifact_dir)
    _ensure_embed(cfg, artifact_dir)
    if not (artifact_dir / "label_to_cluster.json").exists():
        run_coarsen(cfg, artifact_dir)


def _ensure_train(cfg: dict[str, Any], artifact_dir: Path) -> None:
    _ensure_coarsen(cfg, artifact_dir)
    if not (artifact_dir / "models" / "coarse_svm.joblib").exists():
        run_train(cfg, artifact_dir)


def _ensure_eval(cfg: dict[str, Any], artifact_dir: Path) -> None:
    _ensure_train(cfg, artifact_dir)
    if not (artifact_dir / "report_coarse.json").exists():
        run_eval(cfg, artifact_dir)


def cmd_preprocess(cfg: dict[str, Any]) -> None:
    artifact_dir = _artifact_dir(cfg)
    ensure_dir(artifact_dir)
    set_global_seed(int(cfg.get("seed", 42)))
    configure_temp_dir(artifact_dir)
    save_resolved_config(cfg, artifact_dir)
    run_preprocess(cfg, artifact_dir)


def cmd_embed(cfg: dict[str, Any]) -> None:
    artifact_dir = _artifact_dir(cfg)
    ensure_dir(artifact_dir)
    set_global_seed(int(cfg.get("seed", 42)))
    configure_temp_dir(artifact_dir)
    save_resolved_config(cfg, artifact_dir)
    _ensure_preprocess(cfg, artifact_dir)
    run_embed(cfg, artifact_dir)


def cmd_split(cfg: dict[str, Any]) -> None:
    artifact_dir = _artifact_dir(cfg)
    ensure_dir(artifact_dir)
    set_global_seed(int(cfg.get("seed", 42)))
    configure_temp_dir(artifact_dir)
    save_resolved_config(cfg, artifact_dir)
    _ensure_preprocess(cfg, artifact_dir)
    run_split(cfg, artifact_dir)


def cmd_coarsen(cfg: dict[str, Any]) -> None:
    artifact_dir = _artifact_dir(cfg)
    ensure_dir(artifact_dir)
    set_global_seed(int(cfg.get("seed", 42)))
    configure_temp_dir(artifact_dir)
    save_resolved_config(cfg, artifact_dir)
    _ensure_split(cfg, artifact_dir)
    _ensure_embed(cfg, artifact_dir)
    run_coarsen(cfg, artifact_dir)


def cmd_train(cfg: dict[str, Any]) -> None:
    artifact_dir = _artifact_dir(cfg)
    ensure_dir(artifact_dir)
    set_global_seed(int(cfg.get("seed", 42)))
    configure_temp_dir(artifact_dir)
    save_resolved_config(cfg, artifact_dir)
    _ensure_coarsen(cfg, artifact_dir)
    run_train(cfg, artifact_dir)


def cmd_eval(cfg: dict[str, Any]) -> None:
    artifact_dir = _artifact_dir(cfg)
    ensure_dir(artifact_dir)
    set_global_seed(int(cfg.get("seed", 42)))
    configure_temp_dir(artifact_dir)
    save_resolved_config(cfg, artifact_dir)
    _ensure_train(cfg, artifact_dir)
    run_eval(cfg, artifact_dir)


def cmd_report(cfg: dict[str, Any]) -> None:
    from .step_report import run_report

    artifact_dir = _artifact_dir(cfg)
    ensure_dir(artifact_dir)
    set_global_seed(int(cfg.get("seed", 42)))
    configure_temp_dir(artifact_dir)
    save_resolved_config(cfg, artifact_dir)
    _ensure_eval(cfg, artifact_dir)
    run_report(cfg, artifact_dir)


def main() -> None:
    ap = argparse.ArgumentParser(prog="dialectsense")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["preprocess", "embed", "split", "coarsen", "train", "eval", "report"]:
        sp = sub.add_parser(name)
        sp.add_argument("--config", required=True, help="Path to JSON config (e.g., configs/smoke.json)")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.cmd == "preprocess":
        cmd_preprocess(cfg)
    elif args.cmd == "embed":
        cmd_embed(cfg)
    elif args.cmd == "split":
        cmd_split(cfg)
    elif args.cmd == "coarsen":
        cmd_coarsen(cfg)
    elif args.cmd == "train":
        cmd_train(cfg)
    elif args.cmd == "eval":
        cmd_eval(cfg)
    elif args.cmd == "report":
        cmd_report(cfg)
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import logging
from typing import Any

from .config import load_config
from .pipeline import (
    artifact_dir_for_cfg,
    ensure_coarsen,
    ensure_embed,
    ensure_eval,
    ensure_preprocess,
    ensure_report,
    ensure_split,
    ensure_train,
    prepare_run,
 )


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dialectsense")


def cmd_preprocess(cfg: dict[str, Any]) -> None:
    artifact_dir = artifact_dir_for_cfg(cfg)
    prepare_run(cfg, artifact_dir)
    ensure_preprocess(cfg, artifact_dir)


def cmd_embed(cfg: dict[str, Any]) -> None:
    artifact_dir = artifact_dir_for_cfg(cfg)
    prepare_run(cfg, artifact_dir)
    ensure_preprocess(cfg, artifact_dir)
    ensure_embed(cfg, artifact_dir)


def cmd_split(cfg: dict[str, Any]) -> None:
    artifact_dir = artifact_dir_for_cfg(cfg)
    prepare_run(cfg, artifact_dir)
    ensure_preprocess(cfg, artifact_dir)
    ensure_split(cfg, artifact_dir)


def cmd_coarsen(cfg: dict[str, Any]) -> None:
    artifact_dir = artifact_dir_for_cfg(cfg)
    prepare_run(cfg, artifact_dir)
    ensure_split(cfg, artifact_dir)
    ensure_embed(cfg, artifact_dir)
    ensure_coarsen(cfg, artifact_dir)


def cmd_train(cfg: dict[str, Any]) -> None:
    artifact_dir = artifact_dir_for_cfg(cfg)
    prepare_run(cfg, artifact_dir)
    ensure_coarsen(cfg, artifact_dir)
    ensure_train(cfg, artifact_dir)


def cmd_eval(cfg: dict[str, Any]) -> None:
    artifact_dir = artifact_dir_for_cfg(cfg)
    prepare_run(cfg, artifact_dir)
    ensure_train(cfg, artifact_dir)
    ensure_eval(cfg, artifact_dir)


def cmd_report(cfg: dict[str, Any]) -> None:
    artifact_dir = artifact_dir_for_cfg(cfg)
    prepare_run(cfg, artifact_dir)
    ensure_report(cfg, artifact_dir)


def cmd_ui(cfg: dict[str, Any], config_path: str) -> None:
    from .ui.web import launch

    artifact_dir = artifact_dir_for_cfg(cfg)
    prepare_run(cfg, artifact_dir)
    launch(default_config_path=config_path)


def main() -> None:
    ap = argparse.ArgumentParser(prog="dialectsense")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["preprocess", "embed", "split", "coarsen", "train", "eval", "report", "ui"]:
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
    elif args.cmd == "ui":
        cmd_ui(cfg, config_path=str(args.config))
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()

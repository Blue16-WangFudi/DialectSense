from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResolvedPaths:
    repo_root: Path
    artifact_dir: Path


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a JSON object")
    return cfg


def resolve_paths(cfg: dict[str, Any], repo_root: str | Path | None = None) -> ResolvedPaths:
    root = Path(repo_root) if repo_root else Path(".")
    run_name = str(cfg.get("run_name") or "run")
    artifact_dir = root / "artifacts" / run_name
    return ResolvedPaths(repo_root=root, artifact_dir=artifact_dir)


def save_resolved_config(cfg: dict[str, Any], artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out = artifact_dir / "config.resolved.json"
    with out.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def require(cfg: dict[str, Any], dotted_key: str) -> Any:
    cur: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Missing config key: {dotted_key}")
        cur = cur[part]
    return cur

from __future__ import annotations

import hashlib
import json
import random
import tempfile
import os
from pathlib import Path
from typing import Any

import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def stable_json_hash(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def configure_temp_dir(base_dir: str | Path) -> Path:
    """
    Ensure Python (and deps) have a writable temp dir.

    Some environments (sandboxed/WSL mounts) may not allow /tmp; using an artifacts-local
    temp directory makes the pipeline more reliable.
    """
    # Gradio relies on a stable temp/upload dir; if it's set, do not override it per-run.
    gradio_tmp = os.environ.get("GRADIO_TEMP_DIR")
    if gradio_tmp:
        p = ensure_dir(Path(gradio_tmp))
        tempfile.tempdir = str(p)
        return p

    p = ensure_dir(Path(base_dir) / "tmp")
    tempfile.tempdir = str(p)
    return p


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

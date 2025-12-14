from __future__ import annotations

import platform
import shutil
import stat
import tarfile
import urllib.request
from pathlib import Path


def _chmod_x(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    base = repo_root / ".cache" / "ffmpeg"
    bin_dir = base / "bin"
    ffmpeg_path = bin_dir / "ffmpeg"
    ffprobe_path = bin_dir / "ffprobe"

    if ffmpeg_path.exists() and ffprobe_path.exists():
        print(f"ffmpeg already present: {ffmpeg_path}")
        return 0

    arch = platform.machine().lower()
    if arch not in ("x86_64", "amd64"):
        raise RuntimeError(f"Unsupported arch for bundled ffmpeg bootstrap: {arch}")

    # Static builds from https://johnvansickle.com/ffmpeg/ (Linux x86_64).
    url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
    tmp_dir = base / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tar_path = tmp_dir / "ffmpeg-static.tar.xz"

    print(f"Downloading ffmpeg static build: {url}")
    urllib.request.urlretrieve(url, tar_path)  # nosec - tooling bootstrap only

    extract_dir = tmp_dir / "extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, mode="r:xz") as tf:
        tf.extractall(extract_dir)  # nosec - extracting trusted archive

    candidates = list(extract_dir.glob("ffmpeg-*-amd64-static/ffmpeg"))
    candidates_probe = list(extract_dir.glob("ffmpeg-*-amd64-static/ffprobe"))
    if not candidates or not candidates_probe:
        raise RuntimeError("Unexpected ffmpeg archive layout; ffmpeg/ffprobe not found")

    bin_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidates[0], ffmpeg_path)
    shutil.copy2(candidates_probe[0], ffprobe_path)
    _chmod_x(ffmpeg_path)
    _chmod_x(ffprobe_path)

    # Best-effort cleanup.
    try:
        shutil.rmtree(extract_dir)
    except Exception:
        pass
    try:
        tar_path.unlink(missing_ok=True)
    except Exception:
        pass

    print(f"Installed: {ffmpeg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


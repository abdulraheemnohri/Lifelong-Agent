"""Hugging Face GGUF model downloader.

Implements (or emulates) this exact command::

    huggingface-cli download tencent/Youtu-LLM-2B \\
      --revision GGUF \\
      --include "*.gguf" \\
      --local-dir Youtu-LLM-2B-GGUF

Strategy:
1. If GGUF files already exist locally, skip the download.
2. Try the exact ``huggingface-cli`` command.
3. Fall back to a ``python -c`` snapshot_download subprocess.
4. Final fallback: in-process ``huggingface_hub.snapshot_download``.

``HF_TOKEN`` is honoured for gated repositories. The best GGUF file is
selected automatically (common quants such as q4_k_m are preferred).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from ..utils.logging import get_logger

logger = get_logger("downloader")

DEFAULT_REPO = "tencent/Youtu-LLM-2B"
DEFAULT_REVISION = "GGUF"
DEFAULT_LOCAL_DIR = "Youtu-LLM-2B-GGUF"

_PREFERRED_QUANTS = ("q4_k_m", "q4_0", "q5_k_m", "q4_k_s", "q5_0", "q8_0")


def find_gguf_files(model_dir: Path) -> list[Path]:
    """Return all ``*.gguf`` files under *model_dir* (recursive)."""
    model_dir = Path(model_dir)
    if not model_dir.exists():
        return []
    return sorted(p for p in model_dir.rglob("*.gguf") if p.is_file())


def select_best_gguf(files: list[Path]) -> Path:
    """Pick the best GGUF file.

    Preference order:
    1. Common quantizations (q4_k_m, q4_0, q5_k_m, ...).
    2. Otherwise the smallest reasonable file (fastest to load).
    """
    if not files:
        raise ValueError("No GGUF files provided")

    lowered = [(p, p.name.lower()) for p in files]
    for quant in _PREFERRED_QUANTS:
        for path, name in lowered:
            if quant in name:
                return path

    # Smallest file that is still "reasonable" (> 1 MB guard against stubs)
    sized = [(p.stat().st_size, p) for p in files if p.stat().st_size > 1_000_000]
    if sized:
        return min(sized)[1]
    return min(files, key=lambda p: p.stat().st_size)


def _build_cli_command(repo: str, revision: str, local_dir: Path) -> list[str]:
    """The exact huggingface-cli command required by the specification."""
    return [
        "huggingface-cli",
        "download",
        repo,
        "--revision",
        revision,
        "--include",
        "*.gguf",
        "--local-dir",
        str(local_dir),
    ]


def _try_huggingface_cli(repo: str, revision: str, local_dir: Path, token: str | None) -> bool:
    """Attempt the exact huggingface-cli command. Returns True on success."""
    command = _build_cli_command(repo, revision, local_dir)
    env = dict(os.environ)
    if token:
        env["HF_TOKEN"] = token
    try:
        result = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=3600
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.info("huggingface-cli unavailable: %s", exc)
        return False
    if result.returncode == 0:
        logger.info("Downloaded via huggingface-cli")
        return True
    logger.info("huggingface-cli failed: %s", (result.stderr or "").strip()[:200])
    return False


def _try_python_module(repo: str, revision: str, local_dir: Path, token: str | None) -> bool:
    """Fallback: run snapshot_download through a python subprocess."""
    snippet = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id={repo!r}, revision={revision!r}, "
        f"allow_patterns=['*.gguf'], local_dir={str(local_dir)!r})"
    )
    command = [sys.executable, "-c", snippet]
    env = dict(os.environ)
    if token:
        env["HF_TOKEN"] = token
    try:
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=3600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.info("python module fallback unavailable: %s", exc)
        return False
    if result.returncode == 0:
        logger.info("Downloaded via python -c snapshot_download")
        return True
    logger.info("python module fallback failed: %s", (result.stderr or "").strip()[:200])
    return False


def _try_snapshot_download(repo: str, revision: str, local_dir: Path, token: str | None) -> bool:
    """Final fallback: in-process huggingface_hub.snapshot_download."""
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as exc:
        logger.info("huggingface_hub not installed: %s", exc)
        return False
    try:
        snapshot_download(
            repo_id=repo,
            revision=revision,
            allow_patterns=["*.gguf"],
            local_dir=str(local_dir),
            token=token,
        )
        logger.info("Downloaded via in-process snapshot_download")
        return True
    except Exception as exc:  # noqa: BLE001 - network/model errors are expected
        logger.warning("snapshot_download failed: %s", exc)
        return False


def ensure_model(
    repo: str = DEFAULT_REPO,
    revision: str = DEFAULT_REVISION,
    local_dir: str | Path = DEFAULT_LOCAL_DIR,
    force: bool = False,
    token: str | None = None,
) -> Path:
    """Ensure the GGUF model exists locally and return the best GGUF path.

    Raises ``RuntimeError`` when no GGUF file can be obtained.
    """
    local_dir = Path(local_dir)
    token = token or os.environ.get("HF_TOKEN")

    existing = find_gguf_files(local_dir)
    if existing and not force:
        logger.info("Model already present (%d GGUF files); skipping download", len(existing))
        return select_best_gguf(existing)

    local_dir.mkdir(parents=True, exist_ok=True)

    attempts = (
        ("huggingface-cli", _try_huggingface_cli),
        ("python module", _try_python_module),
        ("snapshot_download", _try_snapshot_download),
    )
    for label, attempt in attempts:
        logger.info("Trying downloader: %s", label)
        if attempt(repo, revision, local_dir, token):
            break

    files = find_gguf_files(local_dir)
    if not files:
        raise RuntimeError(
            f"No GGUF files found in {local_dir!r} after trying all downloaders. "
            "Check your internet connection, HF_TOKEN, and that the repository "
            f"'{repo}' (revision '{revision}') publishes GGUF files."
        )
    return select_best_gguf(files)

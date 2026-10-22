"""Tiny JSON persistence helpers used for all agent state files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .fs import atomic_write_text


def read_json(path: Path, default: Any = None) -> Any:
    """Read JSON from *path*; return *default* when missing or invalid."""
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def write_json(path: Path, data: Any) -> Path:
    """Atomically write *data* as pretty-printed JSON."""
    text = json.dumps(data, indent=2, sort_keys=False, default=str)
    return atomic_write_text(Path(path), text + "\n")

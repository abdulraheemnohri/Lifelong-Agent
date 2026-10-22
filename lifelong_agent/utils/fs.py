"""Filesystem helpers with workspace-safety checks."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """Create a directory (and parents) if missing; return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_within(child: Path, parent: Path) -> bool:
    """True if *child* is *parent* or located underneath it."""
    child = Path(child).resolve()
    parent = Path(parent).resolve()
    return child == parent or parent in child.parents


def atomic_write_text(path: Path, text: str) -> Path:
    """Write text atomically (write to a temp file, then replace)."""
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def slugify(text: str, max_words: int = 4, separator: str = "_") -> str:
    """Turn arbitrary text into a safe lowercase slug."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return separator.join(words[:max_words]) or "item"


def short_hash(text: str, length: int = 6) -> str:
    """Deterministic short hash of a string (stable across runs/platforms)."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:length]

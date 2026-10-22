"""Embedding backends.

- ``SentenceTransformerEmbedder``: local ``sentence-transformers/all-MiniLM-L6-v2``
  when the library is installed.
- ``HashEmbedder``: a fully deterministic, dependency-free hash embedder used
  as the offline fallback.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Sequence

from ..utils.logging import get_logger

logger = get_logger("embedders")

ST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class BaseEmbedder:
    """Common interface for embedding backends."""

    name: str = "base"

    @property
    def dim(self) -> int:
        raise NotImplementedError

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError


class HashEmbedder(BaseEmbedder):
    """Deterministic bag-of-hashed-tokens embedder.

    Uses hashlib (not Python's salted ``hash()``) so vectors are identical
    across processes, machines, and runs.
    """

    name = "hash"

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _bucket(self, token: str) -> int:
        # MD5 is the classic "hashing trick" bucketing hash: deterministic
        # across processes/machines and non-cryptographic by design.
        digest = hashlib.md5(token.encode("utf-8")).digest()  # noqa: S324
        return int.from_bytes(digest[:8], "big") % self._dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for token in tokens:
            vector[self._bucket(token)] += 1.0
        norm = math.sqrt(sum(component * component for component in vector))
        if norm > 0:
            vector = [component / norm for component in vector]
        return vector


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local sentence-transformers embedder (all-MiniLM-L6-v2, 384 dims)."""

    name = "sentence_transformer"

    def __init__(self, model_name: str = ST_MODEL_NAME) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        return [float(x) for x in self._model.encode(text)]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors (0.0 for empty vectors)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def get_embedder(backend: str = "auto", dim: int = 256) -> BaseEmbedder:
    """Resolve an embedder backend with graceful fallback.

    ``auto`` and ``sentence-transformer`` use the local model when the
    library is installed and fall back to the deterministic hash embedder
    otherwise (with a warning).
    """
    backend = (backend or "auto").lower()
    if backend == "hash":
        return HashEmbedder(dim)
    if backend in {"auto", "sentence-transformer", "sentence_transformer", "st"}:
        try:
            return SentenceTransformerEmbedder()
        except ImportError:
            logger.info(
                "sentence-transformers unavailable; using deterministic hash embedder"
            )
            return HashEmbedder(dim)
    logger.warning("Unknown embedder backend %r; using hash embedder", backend)
    return HashEmbedder(dim)

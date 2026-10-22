"""Vector memory with two backends.

- ``ChromaVectorMemory``: persistent ChromaDB collection (when installed).
- ``JSONVectorMemory``: dependency-free JSON file with cosine similarity.

Both store the full experience record (task, action, feedback, reflection,
score, strategy version, timestamp, tool usage, metadata) and support
semantic search.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..utils.json_utils import read_json, write_json
from ..utils.logging import get_logger
from .embedders import BaseEmbedder, cosine_similarity

logger = get_logger("memory")


class BaseVectorMemory:
    """Interface shared by vector memory backends."""

    backend_name = "base"

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        raise NotImplementedError

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError


class JSONVectorMemory(BaseVectorMemory):
    """Persistent JSON-file vector memory with cosine similarity search."""

    backend_name = "json"

    def __init__(self, path: Path, embedder: BaseEmbedder) -> None:
        self.path = Path(path)
        self.embedder = embedder
        self._records: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        data = read_json(self.path, default=[]) or []
        self._records = data if isinstance(data, list) else []

    def _persist(self) -> None:
        write_json(self.path, self._records)

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        record_id = uuid.uuid4().hex
        record = {
            "id": record_id,
            "text": text,
            "metadata": metadata or {},
            "vector": self.embedder.embed(text),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._records.append(record)
        self._persist()
        return record_id

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        if not self._records:
            return []
        query_vector = self.embedder.embed(query)
        scored = [
            (cosine_similarity(query_vector, record["vector"]), record)
            for record in self._records
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        results = []
        for score, record in scored[: max(1, k)]:
            if score <= 0.0:
                continue
            results.append(
                {
                    "id": record["id"],
                    "text": record["text"],
                    "metadata": record.get("metadata", {}),
                    "score": round(score, 4),
                }
            )
        return results

    def count(self) -> int:
        return len(self._records)


class ChromaVectorMemory(BaseVectorMemory):
    """Persistent ChromaDB-backed vector memory."""

    backend_name = "chroma"

    def __init__(self, db_dir: Path, embedder: BaseEmbedder) -> None:
        import chromadb  # type: ignore  # optional dependency

        self.embedder = embedder
        self.client = chromadb.PersistentClient(
            path=str(db_dir),
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="agent_memory",
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        record_id = uuid.uuid4().hex
        clean_metadata = {
            str(key): str(value) for key, value in (metadata or {}).items()
        }
        self.collection.add(
            ids=[record_id],
            documents=[text],
            embeddings=[self.embedder.embed(text)],
            metadatas=[clean_metadata],
        )
        return record_id

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        if self.collection.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=[self.embedder.embed(query)],
            n_results=min(max(1, k), max(1, self.collection.count())),
        )
        hits = []
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for record_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            score = 1.0 - float(distance)  # cosine distance -> similarity
            hits.append(
                {
                    "id": record_id,
                    "text": document,
                    "metadata": metadata or {},
                    "score": round(max(0.0, min(1.0, score)), 4),
                }
            )
        return hits

    def count(self) -> int:
        return self.collection.count()


def get_vector_memory(
    backend: str, workspace: Path, embedder: BaseEmbedder
) -> BaseVectorMemory:
    """Resolve a vector memory backend.

    ``auto`` uses ChromaDB when installed, otherwise the JSON fallback.
    """
    backend = (backend or "auto").lower()
    workspace = Path(workspace)
    if backend in {"chroma", "auto"}:
        try:
            memory = ChromaVectorMemory(workspace / "vector_db", embedder)
            if backend == "chroma":
                return memory
            logger.info("Using ChromaDB vector memory backend")
            return memory
        except ImportError:
            if backend == "chroma":
                logger.warning("chromadb not installed; falling back to JSON memory")
            else:
                logger.info("chromadb unavailable; using JSON vector memory")
        except Exception as exc:  # noqa: BLE001 - any chroma failure -> fallback
            logger.warning("ChromaDB init failed (%s); using JSON memory", exc)

    json_path = workspace / f"vector_memory_{embedder.name}.json"
    return JSONVectorMemory(json_path, embedder)

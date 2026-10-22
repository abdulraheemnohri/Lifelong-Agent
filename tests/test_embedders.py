"""Embedder tests: determinism, normalization, fallbacks, cosine similarity."""

from __future__ import annotations

import math

from lifelong_agent.memory.embedders import (
    BaseEmbedder,
    HashEmbedder,
    cosine_similarity,
    get_embedder,
)


def test_hash_embedder_deterministic():
    embedder = HashEmbedder(dim=64)
    assert embedder.embed("hello world") == embedder.embed("hello world")


def test_hash_embedder_dimension_and_normalization():
    embedder = HashEmbedder(dim=64)
    vector = embedder.embed("some sample text")
    assert len(vector) == 64
    norm = math.sqrt(sum(x * x for x in vector))
    assert abs(norm - 1.0) < 1e-6


def test_hash_embedder_different_texts_differ():
    embedder = HashEmbedder(dim=64)
    assert embedder.embed("alpha beta") != embedder.embed("gamma delta")


def test_hash_embedder_empty_text_is_zero_vector():
    embedder = HashEmbedder(dim=64)
    assert embedder.embed("") == [0.0] * 64
    assert cosine_similarity(embedder.embed(""), embedder.embed("text")) == 0.0


def test_hash_embedder_stable_across_instances():
    # hashlib (not salted hash()) guarantees cross-process stability.
    a = HashEmbedder(32).embed("lifelong agent")
    b = HashEmbedder(32).embed("lifelong agent")
    assert a == b


def test_cosine_similarity_identical_vectors():
    vector = [0.5, 0.5, 0.0]
    assert abs(cosine_similarity(vector, vector) - 1.0) < 1e-9


def test_cosine_similarity_orthogonal_vectors():
    assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-9


def test_get_embedder_explicit_hash():
    embedder = get_embedder("hash", dim=16)
    assert isinstance(embedder, HashEmbedder)
    assert embedder.dim == 16


def test_get_embedder_auto_returns_working_embedder():
    embedder = get_embedder("auto", dim=32)
    assert isinstance(embedder, BaseEmbedder)
    assert embedder.dim > 0
    assert len(embedder.embed("test")) == embedder.dim


def test_get_embedder_missing_st_falls_back_to_hash(monkeypatch):
    import lifelong_agent.memory.embedders as module

    def _fail(name=module.ST_MODEL_NAME):
        raise ImportError("sentence-transformers not installed")

    monkeypatch.setattr(module, "SentenceTransformerEmbedder", _fail)
    embedder = module.get_embedder("sentence-transformer", dim=32)
    assert isinstance(embedder, HashEmbedder)

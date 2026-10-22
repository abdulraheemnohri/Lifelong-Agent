"""Vector memory tests: JSON backend add/search/persist + factory."""

from __future__ import annotations

from pathlib import Path

from lifelong_agent.memory.embedders import HashEmbedder
from lifelong_agent.memory.vector_memory import (
    BaseVectorMemory,
    JSONVectorMemory,
    get_vector_memory,
)


def test_json_memory_add_and_search_ranking(tmp_path: Path):
    memory = JSONVectorMemory(tmp_path / "mem.json", HashEmbedder(64))
    memory.add("task: count words in a sentence", {"task": "count words"})
    memory.add("task: reverse a string", {"task": "reverse"})
    memory.add("task: count characters carefully", {"task": "count chars"})

    hits = memory.search("count words", k=2)
    assert hits, "expected at least one hit"
    assert hits[0]["text"].startswith("task: count words")
    assert all(hits[i]["score"] >= hits[i + 1]["score"] for i in range(len(hits) - 1))


def test_json_memory_metadata_roundtrip(tmp_path: Path):
    memory = JSONVectorMemory(tmp_path / "mem.json", HashEmbedder(64))
    memory.add("experience about testing", {"score": 0.9, "tool_usage": "echo"})
    hits = memory.search("testing", k=1)
    assert hits[0]["metadata"]["score"] == 0.9
    assert hits[0]["metadata"]["tool_usage"] == "echo"


def test_json_memory_persistence_across_instances(tmp_path: Path):
    path = tmp_path / "mem.json"
    memory = JSONVectorMemory(path, HashEmbedder(64))
    memory.add("remember this", {"key": "value"})
    reloaded = JSONVectorMemory(path, HashEmbedder(64))
    assert reloaded.count() == 1
    assert reloaded.search("remember", k=1)[0]["text"] == "remember this"


def test_json_memory_empty_search(tmp_path: Path):
    memory = JSONVectorMemory(tmp_path / "mem.json", HashEmbedder(64))
    assert memory.search("anything") == []
    assert memory.count() == 0


def test_factory_json_backend(tmp_path: Path):
    memory = get_vector_memory("json", tmp_path, HashEmbedder(32))
    assert isinstance(memory, JSONVectorMemory)
    assert memory.path.name == "vector_memory_hash.json"


def test_factory_auto_returns_a_working_memory(tmp_path: Path):
    memory = get_vector_memory("auto", tmp_path, HashEmbedder(32))
    assert isinstance(memory, BaseVectorMemory)
    record_id = memory.add("hello", {"a": 1})
    assert record_id
    assert memory.count() >= 1

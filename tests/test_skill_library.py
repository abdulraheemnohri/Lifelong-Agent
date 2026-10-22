"""Skill library tests: upsert, retrieve ranking, prune, persistence."""

from __future__ import annotations

from pathlib import Path

from lifelong_agent.memory.skill_library import Skill, SkillLibrary


def _skill(skill_id: str, name: str, description: str, score: float = 0.5,
           uses: int = 0, keywords: list[str] | None = None) -> Skill:
    return Skill(id=skill_id, name=name, description=description,
                 score=score, uses=uses, keywords=keywords or [])


def test_upsert_and_get(tmp_path: Path):
    library = SkillLibrary(tmp_path / "skills.json")
    skill = _skill("skill_a", "Word Counting", "Count words in text", 0.9)
    library.upsert(skill)
    assert library.get("skill_a") is not None
    assert library.get("skill_a").name == "Word Counting"


def test_upsert_merges_usage(tmp_path: Path):
    library = SkillLibrary(tmp_path / "skills.json")
    library.upsert(_skill("skill_a", "A", "desc", 0.8))
    library.upsert(_skill("skill_a", "A", "desc updated", 0.6))
    merged = library.get("skill_a")
    assert merged.uses == 1
    assert 0.6 < merged.score < 0.8


def test_retrieve_relevance_ranking(tmp_path: Path):
    library = SkillLibrary(tmp_path / "skills.json")
    library.upsert(_skill("skill_words", "Words", "count words in text", 0.5,
                          keywords=["count", "words"]))
    library.upsert(_skill("skill_reverse", "Reverse", "reverse a string", 0.9))
    hits = library.retrieve("count words", k=2)
    assert hits, "expected ranked hits"
    assert hits[0].id == "skill_words"


def test_prune_removes_low_quality(tmp_path: Path):
    library = SkillLibrary(tmp_path / "skills.json")
    library.upsert(_skill("good", "Good", "high quality lesson", 0.9))
    library.upsert(_skill("bad", "Bad", "weak lesson", 0.05))
    pruned = library.prune(min_score=0.2)
    assert pruned == 1
    assert library.get("bad") is None
    assert library.get("good") is not None


def test_prune_caps_library_size(tmp_path: Path):
    library = SkillLibrary(tmp_path / "skills.json")
    for index in range(10):
        library.upsert(_skill(f"s{index}", f"S{index}", f"lesson {index}", 0.5))
    library.prune(min_score=0.0, max_skills=4)
    assert len(library.list()) <= 4


def test_persistence_roundtrip(tmp_path: Path):
    path = tmp_path / "skills.json"
    library = SkillLibrary(path)
    library.upsert(_skill("skill_x", "X", "persisted lesson", 0.7))
    reloaded = SkillLibrary(path)
    assert reloaded.get("skill_x").description == "persisted lesson"


def test_list_sorted_by_score(tmp_path: Path):
    library = SkillLibrary(tmp_path / "skills.json")
    library.upsert(_skill("low", "Low", "l", 0.1))
    library.upsert(_skill("high", "High", "h", 0.9))
    assert library.list()[0].id == "high"

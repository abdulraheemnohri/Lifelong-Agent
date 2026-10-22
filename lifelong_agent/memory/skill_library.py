"""Skill library: distilled, reusable lessons with ranking and pruning."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..utils.json_utils import read_json, write_json
from ..utils.logging import get_logger

logger = get_logger("skills")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Skill:
    """A distilled reusable lesson."""

    id: str
    name: str
    description: str
    score: float = 0.5
    uses: int = 0
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Skill":
        return cls(
            id=str(data["id"]),
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            score=float(data.get("score", 0.5)),
            uses=int(data.get("uses", 0)),
            created_at=str(data.get("created_at", _now())),
            updated_at=str(data.get("updated_at", _now())),
            keywords=[str(k) for k in data.get("keywords", [])],
        )


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class SkillLibrary:
    """Persistent, ranked collection of skills."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.skills: dict[str, Skill] = {}
        self.reload()

    def reload(self) -> None:
        data = read_json(self.path, default={"skills": {}}) or {}
        raw = data.get("skills", {}) if isinstance(data, dict) else {}
        self.skills = {key: Skill.from_dict(value) for key, value in raw.items()}

    def save(self) -> None:
        payload = {"skills": {key: skill.to_dict() for key, skill in self.skills.items()}}
        write_json(self.path, payload)

    def upsert(self, skill: Skill) -> Skill:
        """Insert or update a skill; merges usage and score on update."""
        existing = self.skills.get(skill.id)
        if existing is not None:
            existing.uses += 1
            existing.score = (existing.score + skill.score) / 2.0
            existing.description = skill.description or existing.description
            existing.keywords = sorted(set(existing.keywords) | set(skill.keywords))
            existing.updated_at = _now()
            merged = existing
        else:
            merged = skill
            self.skills[skill.id] = skill
        self.save()
        return merged

    def get(self, skill_id: str) -> Skill | None:
        return self.skills.get(skill_id)

    def retrieve(self, query: str, k: int = 5) -> list[Skill]:
        """Rank skills by relevance, score, and usage."""
        query_tokens = _tokens(query)
        scored: list[tuple[float, Skill]] = []
        for skill in self.skills.values():
            skill_tokens = (
                _tokens(skill.name) | _tokens(skill.description) | set(skill.keywords)
            )
            if query_tokens and skill_tokens:
                overlap = len(query_tokens & skill_tokens) / len(
                    query_tokens | skill_tokens
                )
            else:
                overlap = 0.0
            relevance = 0.5 * overlap + 0.3 * min(max(skill.score, 0.0), 1.0)
            relevance += 0.2 * (min(skill.uses, 10) / 10.0)
            scored.append((relevance, skill))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [skill for _, skill in scored[: max(1, k)]]

    def prune(self, min_score: float = 0.2, max_skills: int = 100) -> int:
        """Drop low-quality skills and cap library size. Returns pruned count."""
        before = len(self.skills)
        kept = {
            key: skill
            for key, skill in self.skills.items()
            if skill.score >= min_score and skill.uses >= 0
        }
        if len(kept) > max_skills:
            ranked = sorted(kept.values(), key=lambda s: s.score, reverse=True)
            keep_ids = {skill.id for skill in ranked[:max_skills]}
            kept = {key: skill for key, skill in kept.items() if skill.id in keep_ids}
        pruned = before - len(kept)
        if pruned:
            self.skills = kept
            self.save()
            logger.info("Pruned %d low-quality skills", pruned)
        return pruned

    def list(self) -> list[Skill]:
        return sorted(self.skills.values(), key=lambda s: s.score, reverse=True)

"""Reflector: produces post-task reflections and distills reusable skills."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from ..memory.skill_library import Skill
from ..utils.fs import slugify
from ..llm.base import BaseLLM

REFLECT_PROMPT = """Reflect on this completed task in 2-3 sentences: what worked,
what to repeat, and what to avoid in future similar tasks.

Task: {task}
Result: {result}
Score: {score}
"""

LESSON_PROMPT = """Distill a single reusable lesson (one sentence) for tasks
similar to the following. Reply with the lesson only.

Task: {task}
Reflection: {reflection}
"""


class Reflector:
    """Self-reflection and skill distillation."""

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    def reflect(self, task: str, result: dict[str, Any], score: float) -> str:
        """Return a short reflection about how the task went."""
        summary = {k: result.get(k) for k in ("ok", "action", "output", "error")}
        prompt = REFLECT_PROMPT.format(task=task, result=summary, score=round(score, 3))
        try:
            return self.llm.generate(prompt)
        except Exception:  # noqa: BLE001
            return f"Task scored {round(score, 3)}; verify and record the lesson."

    def distill_skill(
        self,
        task: str,
        reflection: str,
        score: float,
        threshold: float = 0.75,
    ) -> Skill | None:
        """Create a reusable skill when the outcome was good enough."""
        if score < threshold:
            return None
        prompt = LESSON_PROMPT.format(task=task, reflection=reflection)
        try:
            lesson = self.llm.generate(prompt)
        except Exception:  # noqa: BLE001
            lesson = reflection
        keywords = re.findall(r"[a-z0-9]+", task.lower())[:8]
        skill_id = f"skill_{slugify(task, 4)}_{slugify(lesson, 3)}"
        return Skill(
            id=skill_id,
            name=slugify(task, 4).replace("_", " ").title(),
            description=lesson.strip()[:300],
            score=round(min(score, 1.0), 4),
            keywords=keywords,
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

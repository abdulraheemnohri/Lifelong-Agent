"""Planner: composes an action plan informed by memory, skills, and tools."""

from __future__ import annotations

from typing import Any

from ..llm.base import BaseLLM

PLAN_PROMPT = """You are a lifelong agent. Produce a short, numbered plan to complete the task.

Active strategy:
{strategy}

Relevant memories:
{memories}

Relevant skills:
{skills}

Available tools:
{tools}

Task: {task}

Reply with a numbered plan (3-6 steps) only."""


class Planner:
    """Turns a task + context into an actionable plan."""

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    def _format_memories(self, memories: list[dict[str, Any]]) -> str:
        if not memories:
            return "(none)"
        return "\n".join(f"- {m.get('text', '')[:160]}" for m in memories[:3])

    def _format_skills(self, skills: list[Any]) -> str:
        if not skills:
            return "(none)"
        return "\n".join(f"- {s.name}: {s.description[:160]}" for s in skills[:3])

    def _format_tools(self, tools: list[Any]) -> str:
        if not tools:
            return "(none)"
        return "\n".join(f"- {t.name}: {t.description[:120]}" for t in tools)

    def plan(
        self,
        task: str,
        memories: list[dict[str, Any]] | None = None,
        skills: list[Any] | None = None,
        tools: list[Any] | None = None,
        strategy: str = "",
    ) -> str:
        """Return a plan for *task* under the current strategy."""
        prompt = PLAN_PROMPT.format(
            strategy=strategy or "(default)",
            memories=self._format_memories(memories or []),
            skills=self._format_skills(skills or []),
            tools=self._format_tools(tools or []),
            task=task,
        )
        try:
            return self.llm.generate(prompt)
        except Exception:  # noqa: BLE001 - planning must never crash the loop
            return "1. Retrieve context.\n2. Act with the safest option.\n3. Verify."

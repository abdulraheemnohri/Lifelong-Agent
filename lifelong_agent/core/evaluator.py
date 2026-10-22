"""Evaluator: converts an action result into a 0..1 quality score."""

from __future__ import annotations

from typing import Any


class Evaluator:
    """Heuristic scoring of task results.

    The score rewards: a successful ``ok`` flag, non-empty output, and the
    absence of errors. A base score is always awarded for attempting.
    """

    def evaluate(self, task: str, result: dict[str, Any]) -> float:
        score = 0.1  # attempted
        if result.get("ok") is True:
            score += 0.4
        output = result.get("output")
        if isinstance(output, str) and output.strip():
            score += 0.3
        elif isinstance(output, dict) and output:
            score += 0.3
        if not result.get("error"):
            score += 0.2
        return round(min(score, 1.0), 4)

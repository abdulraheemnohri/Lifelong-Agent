"""Controlled self-modification pipeline.

Pipeline: proposal -> static validation -> testing -> evaluation ->
commit -> monitoring -> rollback.

The agent never freely overwrites its own core source; every proposal goes
through this engine, is committed to Git, is monitored afterwards, and can
be rolled back if performance degrades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..utils.logging import get_logger
from .agent import LifelongAgent

logger = get_logger("self_modification")


@dataclass
class Proposal:
    """A proposed self-modification."""

    kind: str  # "tool" | "strategy" | "config"
    description: str = ""
    payload: dict[str, Any] | None = None
    source: str | None = None
    name: str | None = None


@dataclass
class ApplyResult:
    """Outcome of applying a proposal."""

    accepted: bool
    kind: str
    detail: dict[str, Any] = field(default_factory=dict)
    commit: str | None = None
    rolled_back: bool = False

    def summary(self) -> str:
        state = "accepted" if self.accepted else "rejected"
        extra = self.detail.get("reason", "")
        return f"{self.kind}: {state} {extra}".strip()


class SelfModificationEngine:
    """Safe, auditable self-modification for the lifelong agent."""

    def __init__(self, agent: LifelongAgent) -> None:
        self.agent = agent
        self._monitoring: dict[str, Any] = {"proposals": 0, "accepted": 0, "rolled_back": 0}

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    def apply(self, proposal: Proposal) -> ApplyResult:
        """Run the full controlled pipeline for a proposal."""
        self._monitoring["proposals"] += 1
        config = self.agent.config

        if proposal.kind == "tool":
            if not config.enable_self_writing_tools:
                return ApplyResult(False, proposal.kind, {"reason": "tools disabled"})
            description = proposal.description or "unnamed capability"
            source = proposal.source
            if source:
                name = proposal.name or self.agent.forge.tool_name_for(description)
                forge_result = self.agent.forge.forge_from_source(name, source, description)
            else:
                forge_result = self.agent.forge.forge(description)
            accepted = forge_result.accepted
            if accepted:
                self._monitoring["accepted"] += 1
                self.agent.state["tools_written"] = (
                    int(self.agent.state.get("tools_written", 0)) + 1
                )
                self.agent.save_state()
                commit = self.agent.git.commit(
                    f"feat: self-modification accepted tool {forge_result.name}",
                    paths=[forge_result.path, forge_result.test_path,
                           config.state_path]
                    if forge_result.path and forge_result.test_path
                    else [config.state_path],
                )
            else:
                commit = None
            return ApplyResult(
                accepted=accepted,
                kind=proposal.kind,
                detail={
                    "name": forge_result.name,
                    "reason": forge_result.reason,
                    "violations": forge_result.violations,
                },
                commit=commit,
            )

        if proposal.kind == "strategy":
            if not config.enable_genetic_evolution:
                return ApplyResult(False, proposal.kind, {"reason": "evolution disabled"})
            if not self.agent.safety.approve_high_risk("strategy evolution"):
                return ApplyResult(False, proposal.kind, {"reason": "approval denied"})
            summary = self.agent.meta_evolve()
            return ApplyResult(
                True, proposal.kind, detail=summary,
                commit=self.agent.git.commit(
                    "feat: self-modification evolved strategy",
                    paths=[config.genomes_path, config.state_path],
                ),
            )

        if proposal.kind == "config":
            if not proposal.payload:
                return ApplyResult(False, proposal.kind, {"reason": "empty payload"})
            if not self.agent.safety.approve_high_risk("configuration change"):
                return ApplyResult(False, proposal.kind, {"reason": "approval denied"})
            applied = config.apply_overrides(proposal.payload)
            if not applied:
                return ApplyResult(False, proposal.kind, {"reason": "no valid keys"})
            self.agent.save_state()
            commit = self.agent.git.commit(
                f"chore: self-modification config change {applied}",
                paths=[config.state_path],
            )
            return ApplyResult(True, proposal.kind, {"applied": applied}, commit=commit)

        return ApplyResult(False, proposal.kind, {"reason": f"unknown kind {proposal.kind!r}"})

    # ------------------------------------------------------------------
    # Monitoring and rollback
    # ------------------------------------------------------------------
    def baseline_score(self) -> float:
        """Average score across all completed tasks."""
        attempted = max(1, int(self.agent.state.get("tasks_attempted", 0)))
        total = float(self.agent.state.get("total_score", 0.0))
        return total / attempted

    def monitor(self, recent_scores: list[float], margin: float = 0.1) -> bool:
        """Roll back when recent performance degrades beyond *margin*."""
        if not recent_scores:
            return False
        recent = sum(recent_scores) / len(recent_scores)
        baseline = self.baseline_score()
        if recent < baseline - margin:
            logger.warning(
                "Performance degraded (recent=%.3f baseline=%.3f); rolling back",
                recent, baseline,
            )
            return self.rollback()
        return False

    def rollback(self) -> bool:
        """Git rollback + full agent reload."""
        target = self.agent.git.rollback()
        if target is None:
            return False
        self.agent.reload()
        self._monitoring["rolled_back"] += 1
        return True

    def status(self) -> dict[str, Any]:
        return dict(self._monitoring)

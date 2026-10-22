"""SafetyGuard: static validation gate and human-approval hook.

The guard enforces that no generated tool is executed before passing AST
validation and pytest, and provides a hook for human approval of
high-risk actions.
"""

from __future__ import annotations

from typing import Callable, Optional

from ..config import AgentConfig
from ..tools.validator import ValidationResult, validate_source
from ..utils.logging import get_logger

logger = get_logger("safety")


class SafetyGuard:
    """Central safety gate for the self-modification pipeline."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._approval_callback: Optional[Callable[[str], bool]] = None

    def set_approval_callback(self, callback: Callable[[str], bool]) -> None:
        """Register a human-approval callback (e.g. wired to the TUI)."""
        self._approval_callback = callback

    def validate_tool_source(self, source: str) -> ValidationResult:
        """Static AST validation of generated tool code."""
        return validate_source(source)

    def approve_high_risk(self, description: str) -> bool:
        """Decide whether a high-risk action may proceed.

        Order: human approval callback (if set) -> human approval mode
        (deny by default) -> default allow for non-high-risk flows.
        """
        if self._approval_callback is not None:
            decision = bool(self._approval_callback(description))
            logger.info("High-risk action %r approved=%s by callback", description, decision)
            return decision
        if self.config.human_approval_mode:
            logger.warning(
                "High-risk action %r DENIED (human approval mode is on)", description
            )
            return False
        return True

    def may_modify_core(self) -> bool:
        """Core source self-modification requires an explicit opt-in."""
        return bool(self.config.enable_core_self_modification)

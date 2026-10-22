"""Abstract LLM interface shared by every backend."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Unified text-in / text-out interface for all LLM backends."""

    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> str:
        """Generate a completion for *prompt* with an optional *system* message."""
        raise NotImplementedError

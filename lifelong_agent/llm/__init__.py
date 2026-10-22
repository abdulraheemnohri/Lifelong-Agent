"""LLM backends: base interface, mock, offline GGUF, and OpenAI-compatible."""

from __future__ import annotations

from ..config import AgentConfig
from .base import BaseLLM
from .mock_llm import MockLLM

__all__ = ["BaseLLM", "MockLLM", "get_llm"]


def get_llm(config: AgentConfig):
    """Resolve the configured LLM backend with safe fallbacks.

    - ``mock``                -> MockLLM (deterministic, offline)
    - ``gguf``                -> OfflineGGUFLLM, MockLLM fallback
    - ``openai_compatible``   -> OpenAICompatibleLLM (local server)
    - ``auto``                -> GGUF if a model is present, else MockLLM
    """
    from ..utils.logging import get_logger

    logger = get_logger("llm")
    backend = (config.llm_backend or "auto").lower()

    if backend == "mock":
        return MockLLM()

    if backend in {"gguf", "auto"}:
        from .offline_gguf_llm import OfflineGGUFLLM

        llm = OfflineGGUFLLM.try_load(config)
        if llm is not None:
            return llm
        if backend == "gguf":
            logger.warning("GGUF model unavailable; falling back to MockLLM")
        return MockLLM()

    if backend == "openai_compatible":
        from .openai_compatible_llm import OpenAICompatibleLLM

        return OpenAICompatibleLLM(
            base_url=config.openai_base_url,
            api_key=config.openai_api_key,
            model=config.openai_model,
            max_tokens=config.gguf_max_tokens,
            temperature=config.gguf_temperature,
        )

    logger.warning("Unknown LLM backend %r; falling back to MockLLM", backend)
    return MockLLM()

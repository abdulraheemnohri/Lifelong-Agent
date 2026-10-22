"""Offline GGUF inference through llama-cpp-python.

The backend loads a local GGUF file (downloaded by ``downloader.py``),
tries chat completion first and falls back to plain completion. If the
model or the library is unavailable the caller falls back to MockLLM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import AgentConfig
from ..utils.logging import get_logger
from .base import BaseLLM

logger = get_logger("gguf")


class OfflineGGUFLLM(BaseLLM):
    """Run a local GGUF model fully offline."""

    name = "gguf"

    def __init__(
        self,
        model_path: Path,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        n_threads: int = 4,
        max_tokens: int = 512,
        temperature: float = 0.7,
        verbose: bool = False,
    ) -> None:
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "llama-cpp-python is required for GGUF inference. "
                "Install it with: pip install llama-cpp-python"
            ) from exc

        self.model_path = Path(model_path)
        self.max_tokens = max_tokens
        self.temperature = temperature
        logger.info("Loading GGUF model from %s", self.model_path)
        self._llm: Any = Llama(
            model_path=str(self.model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            verbose=verbose,
        )

    def generate(self, prompt: str, system: str = "") -> str:
        """Chat completion first; plain completion as fallback."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # 1) chat completion
        try:
            result = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            content = result["choices"][0]["message"].get("content", "")
            if content:
                return content
        except Exception as exc:  # noqa: BLE001 - fall back on any failure
            logger.warning("Chat completion failed (%s); trying plain completion", exc)

        # 2) plain completion fallback
        result = self._llm(
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return result["choices"][0].get("text", "")

    @classmethod
    def try_load(cls, config: AgentConfig) -> "OfflineGGUFLLM | None":
        """Return an instance if a usable GGUF model exists, else ``None``."""
        from .downloader import find_gguf_files, select_best_gguf

        model_dir = config.model_path
        files = find_gguf_files(model_dir)
        if not files:
            return None
        gguf_path = select_best_gguf(files)
        try:
            return cls(
                model_path=gguf_path,
                n_ctx=config.gguf_n_ctx,
                n_gpu_layers=config.gguf_n_gpu_layers,
                n_threads=config.gguf_n_threads,
                max_tokens=config.gguf_max_tokens,
                temperature=config.gguf_temperature,
            )
        except Exception as exc:  # noqa: BLE001 - any load failure -> fallback
            logger.warning("Could not load GGUF model %s: %s", gguf_path, exc)
            return None

"""Optional OpenAI-compatible local-server LLM (e.g. llama.cpp server, vLLM).

Uses only the standard library. If the server cannot be reached, callers
get a ``ConnectionError``; the agent's ``auto`` mode only selects this
backend when it is explicitly configured.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from ..utils.logging import get_logger
from .base import BaseLLM

logger = get_logger("openai_compatible")


class OpenAICompatibleLLM(BaseLLM):
    """Minimal client for an OpenAI-compatible chat/completions endpoint."""

    name = "openai_compatible"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str = "",
        model: str = "local-model",
        max_tokens: int = 512,
        temperature: float = 0.7,
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

    def _post(self, url: str, payload: dict) -> dict:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def generate(self, prompt: str, system: str = "") -> str:
        # 1) chat completion
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            result = self._post(
                f"{self.base_url}/chat/completions",
                {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
            )
            return result["choices"][0]["message"].get("content", "")
        except (urllib.error.URLError, OSError, KeyError, IndexError) as exc:
            logger.warning("chat/completions failed (%s); trying /completions", exc)

        # 2) plain completion fallback
        result = self._post(
            f"{self.base_url}/completions",
            {
                "model": self.model,
                "prompt": (system + "\n\n" + prompt) if system else prompt,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
        )
        return result["choices"][0].get("text", "")

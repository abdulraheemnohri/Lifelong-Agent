"""Deterministic, fully offline mock LLM used for tests and fallbacks."""

from __future__ import annotations

from .base import BaseLLM

_PLAN = (
    "1. Restate the task clearly.\n"
    "2. Retrieve relevant memories and skills.\n"
    "3. Choose the safest available tool when it reduces risk.\n"
    "4. Execute the action and verify the result.\n"
    "5. Record the lesson for future tasks."
)

_REFLECTION = (
    "The task followed the plan; the result was verified. "
    "Keep this approach and store it as a reusable lesson."
)

_MOCK_TOOL = '''TOOL_NAME = "generated_tool_mock"
TOOL_DESCRIPTION = "Mock generated tool."

def run(payload: str = "", **kwargs):
    return {"ok": True, "output": "mock tool: " + str(payload)}
'''


class MockLLM(BaseLLM):
    """A deterministic stand-in LLM.

    Responses depend only on the prompt content, never on randomness or
    the network, which keeps agent runs and pytest suites reproducible.
    """

    name = "mock"

    def generate(self, prompt: str, system: str = "") -> str:
        text = f"{system}\n{prompt}".lower()
        if "plan" in text:
            return _PLAN
        if "reflect" in text or "lesson" in text:
            return _REFLECTION
        if "tool code" in text or "write a python tool" in text:
            return _MOCK_TOOL
        first_line = prompt.strip().splitlines()[0] if prompt.strip() else ""
        return f"[mock] Processed: {first_line[:120]}"

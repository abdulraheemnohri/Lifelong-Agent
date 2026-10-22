"""LLM backend tests: MockLLM determinism and factory fallbacks."""

from __future__ import annotations

from pathlib import Path

from lifelong_agent.config import AgentConfig
from lifelong_agent.llm import get_llm
from lifelong_agent.llm.base import BaseLLM
from lifelong_agent.llm.mock_llm import MockLLM


def test_mock_llm_is_deterministic_and_offline():
    llm = MockLLM()
    first = llm.generate("plan the work", system="be careful")
    second = llm.generate("plan the work", system="be careful")
    assert first == second
    assert isinstance(first, str) and first.strip()


def test_mock_llm_plan_branch():
    llm = MockLLM()
    assert "1." in llm.generate("Please plan this task")


def test_mock_llm_tool_branch_returns_contract_code():
    llm = MockLLM()
    code = llm.generate("Write a Python tool for greeting")
    assert "TOOL_NAME" in code and "def run(" in code


def test_base_llm_is_abstract():
    try:
        BaseLLM()  # type: ignore[abstract]
    except TypeError:
        pass
    else:
        raise AssertionError("BaseLLM must be abstract")


def test_factory_mock_backend(tmp_path: Path):
    config = AgentConfig(workspace=tmp_path)
    config.llm_backend = "mock"
    assert isinstance(get_llm(config), MockLLM)


def test_factory_auto_falls_back_to_mock_without_model(tmp_path: Path):
    config = AgentConfig(workspace=tmp_path)
    config.llm_backend = "auto"
    llm = get_llm(config)
    assert isinstance(llm, BaseLLM)
    # No GGUF model exists in the empty workspace -> MockLLM fallback.
    assert isinstance(llm, MockLLM)

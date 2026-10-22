"""AutoGen adapter tests: fallback mode without pyautogen, JSON replies."""

from __future__ import annotations

import json

from lifelong_agent.autogen_integration import adapter as adapter_module
from lifelong_agent.autogen_integration.adapter import LifelongAgentAdapter


def test_module_imports_without_pyautogen():
    # The module must import whether or not pyautogen is installed.
    assert isinstance(adapter_module.AUTOGEN_AVAILABLE, bool)


def test_adapter_delegates_to_agent_and_returns_json(agent):
    wrapper = LifelongAgentAdapter(agent, name="TestAdapter")
    reply = wrapper.generate_reply(
        [{"role": "user", "content": "count the words in hello world"}]
    )
    data = json.loads(reply)
    assert "ok" in data
    assert "score" in data
    assert data["ok"] is True


def test_adapter_empty_messages_return_error(agent):
    wrapper = LifelongAgentAdapter(agent)
    data = json.loads(wrapper.generate_reply([]))
    assert data["ok"] is False
    assert "error" in data


def test_group_chat_degrades_gracefully(agent):
    result = adapter_module.build_group_chat(agent, "do something")
    if adapter_module.AUTOGEN_AVAILABLE:
        assert result is not None
        participants, manager = result
        assert len(participants) >= 6  # adapter + 5 roles
    else:
        assert result is None

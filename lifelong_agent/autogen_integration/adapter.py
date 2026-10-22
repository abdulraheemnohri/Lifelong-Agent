"""AutoGen adapter: lifelong agent as a conversable agent.

Examples
--------
Single agent chat::

    from lifelong_agent.autogen_integration.adapter import LifelongAgentAdapter
    adapter = LifelongAgentAdapter(agent)
    reply = adapter.generate_reply([{"role": "user", "content": "Debug a failing test"}])

Group chat (requires pyautogen)::

    from lifelong_agent.autogen_integration.adapter import build_group_chat
    agents, manager = build_group_chat(agent, "Refactor the memory module")
"""

from __future__ import annotations

import json
from typing import Any

from ..utils.logging import get_logger

logger = get_logger("autogen")

try:  # optional dependency
    from autogen import ConversableAgent, GroupChat, GroupChatManager  # type: ignore

    AUTOGEN_AVAILABLE = True
except Exception:  # noqa: BLE001 - any import failure -> graceful degradation
    ConversableAgent = None  # type: ignore[assignment]
    GroupChat = None  # type: ignore[assignment]
    GroupChatManager = None  # type: ignore[assignment]
    AUTOGEN_AVAILABLE = False
    logger.info("pyautogen not installed; adapter runs in fallback mode")

GROUP_ROLES: list[tuple[str, str]] = [
    ("Planner", "Break the task into small, verifiable steps."),
    ("Coder", "Propose the minimal concrete implementation."),
    ("Tester", "Define pytest cases that prove correctness."),
    ("Critic", "Point out risks, gaps, and simpler alternatives."),
    ("Safety_Officer", "Reject unsafe or irreversible operations."),
    ("Evolution_Manager", "Record lessons and update the strategy."),
]


def _task_from_messages(messages: list[Any]) -> str:
    """Extract the last user message content from an AutoGen-style list."""
    for message in reversed(messages or []):
        if isinstance(message, dict):
            content = message.get("content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


class LifelongAgentAdapter:
    """ConversableAgent-compatible wrapper around the lifelong agent.

    Incoming task messages are delegated to the lifelong agent core; the
    reply is JSON-formatted so downstream agents can parse the result.
    """

    def __init__(self, agent: Any, name: str = "LifelongAgent") -> None:
        self.agent = agent
        self.name = name
        self._conversable: Any = None
        if AUTOGEN_AVAILABLE:
            self._conversable = ConversableAgent(
                name=name,
                llm_config=False,
                human_input_mode="NEVER",
            )
            self._conversable.register_reply(
                [ConversableAgent, None], self._autogen_reply, position=0
            )

    @property
    def conversable(self) -> Any:
        """The underlying pyautogen ConversableAgent (or ``None``)."""
        return self._conversable

    def _autogen_reply(
        self,
        recipient: Any = None,
        messages: list[Any] | None = None,
        sender: Any = None,
        config: Any = None,
    ) -> tuple[bool, str]:
        reply = self.generate_reply(messages or [])
        return True, reply

    def generate_reply(self, messages: list[Any]) -> str:
        """Run a task through the lifelong agent; return a JSON result."""
        task = _task_from_messages(messages)
        if not task:
            return json.dumps({"ok": False, "error": "no task message found"})
        try:
            result = self.agent.run_task(task)
        except Exception as exc:  # noqa: BLE001 - surface errors as JSON
            result = {"ok": False, "error": str(exc)}
        return json.dumps(
            {
                "ok": bool(result.get("ok")),
                "score": result.get("score"),
                "action": result.get("action"),
                "reflection": result.get("reflection"),
            }
        )


def build_group_chat(agent: Any, task: str) -> tuple[list[Any], Any] | None:
    """Build an optional multi-agent group chat around the lifelong agent.

    Roles: Planner, Coder, Tester, Critic, Safety Officer, Evolution Manager.
    Returns ``(agents, manager)`` or ``None`` when pyautogen is unavailable.
    """
    if not AUTOGEN_AVAILABLE:
        logger.warning("Group chat requested but pyautogen is not installed")
        return None

    adapter = LifelongAgentAdapter(agent)
    participants: list[Any] = [adapter.conversable]
    for role_name, duty in GROUP_ROLES:
        participants.append(
            ConversableAgent(
                name=role_name,
                system_message=f"You are the {role_name.replace('_', ' ')}. {duty}",
                llm_config=False,
                human_input_mode="NEVER",
            )
        )
    group_chat = GroupChat(
        agents=[a for a in participants if a is not None],
        messages=[{"role": "user", "content": task}],
        max_round=6,
    )
    manager = GroupChatManager(groupchat=group_chat, llm_config=False)
    return participants, manager

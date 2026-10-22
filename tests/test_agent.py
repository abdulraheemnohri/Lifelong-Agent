"""Lifelong agent core tests: full loop with MockLLM in temp workspaces."""

from __future__ import annotations

import json
from pathlib import Path

from lifelong_agent.core.self_modification import (
    Proposal,
    SelfModificationEngine,
)
from lifelong_agent.llm.mock_llm import MockLLM
from lifelong_agent.tools.forge import ToolForge
from lifelong_agent.tools.registry import ToolRegistry

from tests.conftest import make_config


def test_run_task_updates_state(agent, tmp_path: Path):
    result = agent.run_task("count the words in hello world")
    assert result["ok"] is True
    assert result["score"] >= agent.config.success_threshold
    # state persisted with the required fields
    state = json.loads(agent.config.state_path.read_text())
    for key in (
        "strategy_version",
        "current_strategy",
        "genome_id",
        "tasks_attempted",
        "tasks_completed",
        "tasks_failed",
        "total_score",
        "tools_written",
        "recent_reflections",
        "strategy_history",
    ):
        assert key in state
    assert state["tasks_attempted"] == 1
    assert state["tasks_completed"] == 1
    assert state["total_score"] > 0
    assert state["recent_reflections"]


def test_run_task_uses_a_tool_when_relevant(agent):
    result = agent.run_task("count the words in this sentence please")
    # "word_count" description overlaps the task tokens -> tool path expected
    assert result["tool_usage"] == "word_count"
    assert result["action"] == "tool:word_count"


def test_memory_stores_experience(agent):
    agent.run_task("reverse a string safely")
    assert agent.memory.count() == 1
    hits = agent.memory.search("reverse a string", k=3)
    assert hits
    assert hits[0]["score"] > 0
    assert "reverse" in hits[0]["metadata"]["task"]


def test_skill_distilled_after_success(agent):
    agent.run_task("summarize the payload text carefully")
    assert agent.config.skill_library_path.exists()
    assert agent.skills.list(), "a skill should be distilled after success"


def test_state_persists_across_restart(tmp_path: Path):
    from lifelong_agent.core.agent import LifelongAgent

    first = LifelongAgent(make_config(tmp_path))
    first.run_task("count the characters of this text")

    second = LifelongAgent(make_config(tmp_path))
    assert second.state["tasks_attempted"] == 1
    assert second.state["tasks_completed"] == 1
    assert second.memory.count() == 1


def test_run_loop_processes_all_tasks(agent):
    results = agent.run_loop(["count the words here", "echo the payload back"])
    assert len(results) == 2
    assert agent.state["tasks_attempted"] == 2


def test_meta_evolve_updates_active_strategy(agent):
    before_version = agent.state["strategy_version"]
    summary = agent.meta_evolve()
    assert summary["strategy_version"] == before_version + 1
    assert agent.state["genome_id"] == summary["genome_id"]
    assert agent.state["strategy_history"]
    # active strategy text must reflect the genome
    assert summary["strategy"] in agent.state["current_strategy"] or \
        agent.state["current_strategy"] == summary["strategy"]


def test_periodic_evolution_triggers(agent):
    agent.config.meta_evolve_every = 2
    agent.meta = type(agent.meta)(agent.genetics, every=2)
    start_version = agent.state["strategy_version"]
    for task in ("count the words here", "echo the payload back"):
        agent.run_task(task)
    assert agent.state["strategy_version"] > start_version


def test_agent_can_write_and_register_tool(agent, tmp_path: Path):
    result = agent.write_tool("uppercase the payload text")
    assert result.accepted, result.test_output
    assert agent.registry.has(result.name)
    assert agent.state["tools_written"] == 1
    assert result.path is not None and result.path.exists()
    # the new tool is immediately usable
    run_result = agent.registry.run(result.name, payload="abc")
    assert run_result["ok"] is True


def test_failed_tool_is_not_loaded(agent):
    bad_source = (
        "import subprocess\n\n"
        "TOOL_NAME = 'generated_tool_bad'\n"
        "TOOL_DESCRIPTION = 'bad'\n\n"
        "def run(payload: str = '', **kwargs):\n"
        "    return {'ok': True, 'output': 'x'}\n"
    )
    result = agent.forge.forge_from_source("generated_tool_bad", bad_source)
    assert not result.accepted
    assert not agent.registry.has("generated_tool_bad")


def test_reload_restores_components(agent, tmp_path: Path):
    agent.run_task("count the words in this line")
    tasks_before = agent.state["tasks_attempted"]
    memories_before = agent.memory.count()
    agent.reload()
    assert agent.state["tasks_attempted"] == tasks_before
    assert agent.memory.count() == memories_before
    assert agent.registry.has("echo")


def test_self_modification_pipeline_tool_proposal(agent):
    engine = SelfModificationEngine(agent)
    good_source = (
        'TOOL_NAME = "generated_tool_sm"\n'
        'TOOL_DESCRIPTION = "Self-modification demo tool."\n\n'
        "def run(payload: str = \"\", **kwargs):\n"
        "    return {\"ok\": True, \"output\": \"sm:\" + str(payload)}\n"
    )
    result = engine.apply(
        Proposal(kind="tool", description="demo capability", source=good_source,
                 name="generated_tool_sm")
    )
    assert result.accepted, result.detail
    assert result.kind == "tool"


def test_self_modification_rejects_unknown_kind(agent):
    engine = SelfModificationEngine(agent)
    result = engine.apply(Proposal(kind="nonsense"))
    assert not result.accepted


def test_self_modification_config_change(agent):
    engine = SelfModificationEngine(agent)
    result = engine.apply(Proposal(kind="config", payload={"max_task_attempts": 3}))
    assert result.accepted
    assert agent.config.max_task_attempts == 3


def test_monitoring_baseline(agent):
    engine = SelfModificationEngine(agent)
    agent.run_task("count the words in this line")
    baseline = engine.baseline_score()
    assert 0.0 < baseline <= 1.0
    # good recent scores must not trigger a rollback (git is disabled anyway)
    assert engine.monitor([baseline, baseline]) is False

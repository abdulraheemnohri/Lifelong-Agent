"""Shared pytest fixtures for the lifelong agent test suite.

All tests run fully offline: MockLLM, hash embeddings, JSON vector memory,
and temporary workspaces. No GPU, no GGUF model, no internet required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lifelong_agent.config import AgentConfig


def make_config(tmp_path: Path, **overrides) -> AgentConfig:
    """Deterministic, offline-friendly configuration in a temp workspace."""
    config = AgentConfig(workspace=tmp_path)
    config.llm_backend = "mock"
    config.embedder_backend = "hash"
    config.vector_backend = "json"
    config.enable_git_self_modification = False
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


@pytest.fixture
def config(tmp_path: Path) -> AgentConfig:
    return make_config(tmp_path)


@pytest.fixture
def agent(tmp_path: Path):
    from lifelong_agent.core.agent import LifelongAgent

    return LifelongAgent(make_config(tmp_path))

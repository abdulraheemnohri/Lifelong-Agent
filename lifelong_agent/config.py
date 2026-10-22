"""Central configuration for the lifelong agent.

All settings can be overridden through environment variables so that the
project stays offline-first and dependency-free.  Heavy optional
dependencies (llama-cpp-python, chromadb, sentence-transformers, pyautogen)
are never imported at configuration time.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value is not None and value != "" else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in _TRUE_VALUES


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_str(name, str(default)))
    except ValueError:
        return default


@dataclass
class AgentConfig:
    """Runtime configuration for the agent and all its subsystems."""

    workspace: Path

    # Model / LLM
    model_repo: str = "tencent/Youtu-LLM-2B"
    model_revision: str = "GGUF"
    model_dir: str = "Youtu-LLM-2B-GGUF"
    llm_backend: str = "auto"  # auto | gguf | mock | openai_compatible
    gguf_n_ctx: int = 4096
    gguf_n_gpu_layers: int = 0
    gguf_n_threads: int = 4
    gguf_max_tokens: int = 512
    gguf_temperature: float = 0.7
    openai_base_url: str = "http://127.0.0.1:8000/v1"
    openai_api_key: str = ""
    openai_model: str = "local-model"

    # Embeddings / vector memory
    embedder_backend: str = "auto"  # auto | sentence-transformer | hash
    vector_backend: str = "auto"  # auto | chroma | json
    hash_embedder_dim: int = 256

    # Behaviour switches
    enable_self_writing_tools: bool = True
    enable_genetic_evolution: bool = True
    enable_git_self_modification: bool = True
    enable_tui: bool = True
    enable_core_self_modification: bool = False  # never enable blindly
    human_approval_mode: bool = False

    # Agent loop
    success_threshold: float = 0.75
    max_task_attempts: int = 2
    meta_evolve_every: int = 3
    genetic_evolve_every: int = 3
    population_size: int = 8
    evolution_seed: int = 42
    pytest_timeout: int = 120
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, workspace: str | Path | None = None) -> "AgentConfig":
        """Build a configuration honouring environment variables."""
        workspace = Path(workspace or os.environ.get("WORKSPACE") or Path.cwd())
        config = cls(workspace=Path(workspace))
        config.model_repo = _env_str("MODEL_REPO", config.model_repo)
        config.model_revision = _env_str("MODEL_REVISION", config.model_revision)
        config.model_dir = _env_str("MODEL_DIR", config.model_dir)
        config.llm_backend = _env_str("LLM_BACKEND", config.llm_backend)
        config.gguf_n_ctx = _env_int("GGUF_N_CTX", config.gguf_n_ctx)
        config.gguf_n_gpu_layers = _env_int("GGUF_N_GPU_LAYERS", config.gguf_n_gpu_layers)
        config.gguf_n_threads = _env_int("GGUF_N_THREADS", config.gguf_n_threads)
        config.gguf_max_tokens = _env_int("GGUF_MAX_TOKENS", config.gguf_max_tokens)
        config.gguf_temperature = _env_float("GGUF_TEMPERATURE", config.gguf_temperature)
        config.openai_base_url = _env_str("OPENAI_BASE_URL", config.openai_base_url)
        config.openai_api_key = _env_str("OPENAI_API_KEY", config.openai_api_key)
        config.openai_model = _env_str("OPENAI_MODEL", config.openai_model)
        config.embedder_backend = _env_str("EMBEDDER_BACKEND", config.embedder_backend)
        config.vector_backend = _env_str("VECTOR_BACKEND", config.vector_backend)
        config.hash_embedder_dim = _env_int("HASH_EMBEDDER_DIM", config.hash_embedder_dim)
        config.enable_self_writing_tools = _env_bool(
            "ENABLE_SELF_WRITING_TOOLS", config.enable_self_writing_tools
        )
        config.enable_genetic_evolution = _env_bool(
            "ENABLE_GENETIC_EVOLUTION", config.enable_genetic_evolution
        )
        config.enable_git_self_modification = _env_bool(
            "ENABLE_GIT_SELF_MODIFICATION", config.enable_git_self_modification
        )
        config.enable_tui = _env_bool("ENABLE_TUI", config.enable_tui)
        config.enable_core_self_modification = _env_bool(
            "ENABLE_CORE_SELF_MODIFICATION", config.enable_core_self_modification
        )
        config.human_approval_mode = _env_bool("HUMAN_APPROVAL_MODE", config.human_approval_mode)
        config.success_threshold = _env_float("SUCCESS_THRESHOLD", config.success_threshold)
        config.max_task_attempts = _env_int("MAX_TASK_ATTEMPTS", config.max_task_attempts)
        config.meta_evolve_every = _env_int("META_EVOLVE_EVERY", config.meta_evolve_every)
        config.genetic_evolve_every = _env_int("GENETIC_EVOLVE_EVERY", config.genetic_evolve_every)
        config.population_size = _env_int("POPULATION_SIZE", config.population_size)
        config.evolution_seed = _env_int("EVOLUTION_SEED", config.evolution_seed)
        config.pytest_timeout = _env_int("PYTEST_TIMEOUT", config.pytest_timeout)
        config.log_level = _env_str("LOG_LEVEL", config.log_level)
        return config

    # ------------------------------------------------------------------
    # Derived paths (all inside the workspace)
    # ------------------------------------------------------------------
    @property
    def state_path(self) -> Path:
        return self.workspace / "agent_state.json"

    @property
    def skill_library_path(self) -> Path:
        return self.workspace / "skill_library.json"

    @property
    def genomes_path(self) -> Path:
        return self.workspace / "genomes.json"

    @property
    def tools_dir(self) -> Path:
        return self.workspace / "tools_generated"

    @property
    def failed_tools_dir(self) -> Path:
        return self.tools_dir / "failed"

    @property
    def vector_db_dir(self) -> Path:
        return self.workspace / "vector_db"

    @property
    def model_path(self) -> Path:
        return self.workspace / self.model_dir

    @property
    def agent_cache_dir(self) -> Path:
        return self.workspace / ".agent_cache"

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["workspace"] = str(self.workspace)
        return data

    def apply_overrides(self, overrides: dict[str, Any]) -> list[str]:
        """Safely apply a dict of config overrides; returns applied keys."""
        allowed = {f for f in self.__dataclass_fields__ if f != "workspace"}
        applied: list[str] = []
        for key, value in overrides.items():
            if key not in allowed:
                continue
            current = getattr(self, key)
            try:
                setattr(self, key, type(current)(value))
                applied.append(key)
            except (TypeError, ValueError):
                continue
        return applied

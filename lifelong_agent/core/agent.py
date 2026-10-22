"""The lifelong agent core.

Implements the lifelong loop::

    task -> retrieve memory/skills/tools -> plan -> act -> feedback ->
    reflect -> distill skill -> update strategy -> store experience ->
    evolve periodically

State persists across restarts in ``agent_state.json``; meaningful changes
are committed to Git and every component can be reloaded after a rollback.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..config import AgentConfig
from ..evolution.genetic import GeneticEvolution
from ..evolution.meta_evolution import MetaEvolution
from ..git.git_manager import GitManager
from ..llm import get_llm
from ..llm.base import BaseLLM
from ..memory.embedders import get_embedder
from ..memory.skill_library import Skill, SkillLibrary
from ..memory.vector_memory import BaseVectorMemory, get_vector_memory
from ..tools.forge import ForgeResult, ToolForge
from ..tools.registry import ToolRegistry, ToolSpec
from ..utils.json_utils import read_json, write_json
from ..utils.logging import get_logger, setup_logging
from .evaluator import Evaluator
from .planner import Planner
from .reflector import Reflector
from .safety import SafetyGuard

logger = get_logger("agent")

MAX_REFLECTIONS = 20


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LifelongAgent:
    """A lifelong, self-evolving, offline-first autonomous agent."""

    def __init__(self, config: AgentConfig, llm: BaseLLM | None = None) -> None:
        setup_logging(config.log_level)
        self.config = config
        self.workspace = Path(config.workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.log_sink: Callable[[str], None] | None = None

        # Embeddings + memory
        self.embedder = get_embedder(config.embedder_backend, dim=config.hash_embedder_dim)
        self.memory: BaseVectorMemory = get_vector_memory(
            config.vector_backend, self.workspace, self.embedder
        )
        self.skills = SkillLibrary(config.skill_library_path)

        # Tools
        self.registry = ToolRegistry()
        self.registry.load_builtin_tools()
        self.registry.load_generated_tools(config.tools_dir)

        # LLM + reasoning modules
        self.llm = llm or get_llm(config)
        self.planner = Planner(self.llm)
        self.evaluator = Evaluator()
        self.reflector = Reflector(self.llm)
        self.safety = SafetyGuard(config)

        # Evolution
        self.genetics = GeneticEvolution(
            config.genomes_path,
            population_size=config.population_size,
            seed=config.evolution_seed,
        )
        self.meta = MetaEvolution(self.genetics, every=config.genetic_evolve_every)

        # Git
        self.git = GitManager(self.workspace, enabled=config.enable_git_self_modification)
        self.git.init()

        # Tool forge
        self.forge = ToolForge(
            self.llm, config.tools_dir, self.registry, pytest_timeout=config.pytest_timeout
        )

        # Persistent state
        self.state: dict[str, Any] = self._load_state()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def log(self, message: str) -> None:
        logger.info(message)
        if self.log_sink is not None:
            try:
                self.log_sink(message)
            except Exception:  # noqa: BLE001 - a UI sink must never break the agent
                pass

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------
    def _default_state(self) -> dict[str, Any]:
        best = self.genetics.best()
        return {
            "strategy_version": 1,
            "current_strategy": best.strategy_text(),
            "genome_id": best.id,
            "tasks_attempted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_score": 0.0,
            "tools_written": 0,
            "recent_reflections": [],
            "strategy_history": [],
            "tasks_since_evolve": 0,
            "created_at": _now(),
            "updated_at": _now(),
        }

    def _load_state(self) -> dict[str, Any]:
        state = read_json(self.config.state_path, default={}) or {}
        defaults = self._default_state()
        if not isinstance(state, dict):
            state = {}
        for key, value in defaults.items():
            state.setdefault(key, value)
        # Validate the genome reference after a possible rollback.
        if self.genetics.get(str(state.get("genome_id"))) is None:
            state["genome_id"] = self.genetics.best().id
        return state

    def save_state(self) -> None:
        self.state["updated_at"] = _now()
        write_json(self.config.state_path, self.state)

    # ------------------------------------------------------------------
    # Lifelong loop
    # ------------------------------------------------------------------
    def _select_tool(self, task: str) -> ToolSpec | None:
        """Pick the most relevant registered tool for a task (token overlap)."""
        import re

        task_tokens = set(re.findall(r"[a-z0-9]+", task.lower()))
        if not task_tokens:
            return None
        best_spec: ToolSpec | None = None
        best_overlap = 0
        for spec in self.registry.list_tools():
            tool_tokens = set(
                re.findall(r"[a-z0-9]+", f"{spec.name} {spec.description}".lower())
            )
            overlap = len(task_tokens & tool_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_spec = spec
        return best_spec if best_overlap >= 2 else None

    def _execute(self, task: str, plan: str) -> dict[str, Any]:
        """Execute the chosen action: a registered tool or the LLM itself."""
        tool = self._select_tool(task)
        if tool is not None:
            self.log(f"Using tool: {tool.name}")
            tool_result = self.registry.run(tool.name, payload=task)
            answer = self.llm.generate(
                f"Task: {task}\nTool output: {tool_result.get('output')}\n"
                "Produce the final answer."
            )
            return {
                "ok": bool(tool_result.get("ok")) and bool(answer),
                "action": f"tool:{tool.name}",
                "output": answer,
                "tool_usage": tool.name,
                "error": tool_result.get("error"),
            }
        answer = self.llm.generate(
            f"Complete the task: {task}\nPlan:\n{plan}",
            system=self.state.get("current_strategy", ""),
        )
        return {"ok": bool(answer), "action": "llm", "output": answer,
                "tool_usage": None, "error": None}

    def run_task(self, task: str) -> dict[str, Any]:
        """Run one full lifelong learning cycle for *task*."""
        task = task.strip()
        if not task:
            return {"ok": False, "score": 0.0, "error": "empty task"}

        self.log(f"Task received: {task}")
        self.state["tasks_attempted"] += 1

        # 1-2. Retrieve memory, skills, tools
        memories = self.memory.search(task, k=3)
        skills = self.skills.retrieve(task, k=3)
        tools = self.registry.list_tools()

        # 3-5. Plan, act, feedback (retry up to max_task_attempts)
        result: dict[str, Any] = {}
        score = 0.0
        for attempt in range(1, max(1, self.config.max_task_attempts) + 1):
            plan = self.planner.plan(
                task, memories=memories, skills=skills, tools=tools,
                strategy=self.state.get("current_strategy", ""),
            )
            result = self._execute(task, plan)
            result["plan"] = plan
            result["attempt"] = attempt
            score = self.evaluator.evaluate(task, result)
            self.log(f"Attempt {attempt} score: {score}")
            if score >= self.config.success_threshold:
                break

        ok = score >= self.config.success_threshold

        # 6. Reflect
        reflection = self.reflector.reflect(task, result, score)
        self.state["recent_reflections"].append(
            {"task": task, "reflection": reflection, "score": score, "at": _now()}
        )
        self.state["recent_reflections"] = self.state["recent_reflections"][-MAX_REFLECTIONS:]

        # 7. Distill a skill
        skill: Skill | None = self.reflector.distill_skill(
            task, reflection, score, threshold=self.config.success_threshold
        )
        if skill is not None:
            self.skills.upsert(skill)
            self.log(f"Skill distilled: {skill.name}")

        # 8-9. Update strategy fitness and store experience
        self.genetics.record_task(str(self.state.get("genome_id")), score)
        self.memory.add(
            text=(
                f"task: {task}\naction: {result.get('action')}\n"
                f"feedback: ok={ok}, score={round(score, 3)}\n"
                f"reflection: {reflection}"
            ),
            metadata={
                "task": task,
                "action": result.get("action"),
                "feedback": f"ok={ok}; score={round(score, 3)}",
                "reflection": reflection,
                "score": round(score, 4),
                "strategy_version": self.state.get("strategy_version"),
                "timestamp": _now(),
                "tool_usage": result.get("tool_usage"),
                "metadata": {"attempt": result.get("attempt")},
            },
        )

        # Bookkeeping
        self.state["total_score"] = round(
            float(self.state.get("total_score", 0.0)) + score, 4
        )
        if ok:
            self.state["tasks_completed"] += 1
        else:
            self.state["tasks_failed"] += 1
        self.state["tasks_since_evolve"] = int(self.state.get("tasks_since_evolve", 0)) + 1
        self.save_state()

        # Autosave milestone to Git
        self.git.commit(
            f"chore: agent completed task '{task[:40]}' (score {round(score, 2)})",
            paths=[self.config.state_path, self.config.skill_library_path,
                   self.config.genomes_path],
        )

        # 10. Evolve periodically
        if self.config.enable_genetic_evolution:
            evolved = self.meta.maybe_evolve(self.state["tasks_since_evolve"])
            if evolved is not None:
                self._activate_genome(evolved)

        self.log(f"Task finished: ok={ok} score={score}")
        return {
            "task": task,
            "ok": ok,
            "score": score,
            "action": result.get("action"),
            "tool_usage": result.get("tool_usage"),
            "reflection": reflection,
            "skill": skill.id if skill else None,
        }

    def run_loop(self, tasks: list[str]) -> list[dict[str, Any]]:
        """Run a sequence of tasks through the lifelong loop."""
        return [self.run_task(task) for task in tasks if task.strip()]

    # ------------------------------------------------------------------
    # Evolution and strategy management
    # ------------------------------------------------------------------
    def _activate_genome(self, genome: Any) -> None:
        """Make *genome* the active strategy and record history."""
        self.state["current_strategy"] = genome.strategy_text()
        self.state["strategy_version"] = int(self.state.get("strategy_version", 1)) + 1
        self.state["genome_id"] = genome.id
        self.state["tasks_since_evolve"] = 0
        self.state["strategy_history"].append(
            {
                "version": self.state["strategy_version"],
                "genome_id": genome.id,
                "fitness": round(genome.fitness, 4),
                "at": _now(),
            }
        )
        self.save_state()
        self.git.commit(
            f"feat: evolve strategy to genome {genome.id} "
            f"(v{self.state['strategy_version']})",
            paths=[self.config.genomes_path, self.config.state_path],
        )
        self.log(
            f"Strategy evolved: v{self.state['strategy_version']} "
            f"(genome {genome.id}, fitness {round(genome.fitness, 3)})"
        )

    def meta_evolve(self) -> dict[str, Any]:
        """Force a genetic evolution step and activate the best genome."""
        genome = self.genetics.evolve()
        self._activate_genome(genome)
        return {
            "genome_id": genome.id,
            "strategy_version": self.state["strategy_version"],
            "fitness": round(genome.fitness, 4),
            "strategy": genome.strategy_text(),
        }

    # ------------------------------------------------------------------
    # Tool writing
    # ------------------------------------------------------------------
    def write_tool(self, description: str) -> ForgeResult:
        """Generate + validate + pytest + register a new tool."""
        if not self.config.enable_self_writing_tools:
            raise RuntimeError("Self-writing tools are disabled in configuration")
        result = self.forge.forge(description)
        if result.accepted:
            self.state["tools_written"] = int(self.state.get("tools_written", 0)) + 1
            self.save_state()
            self.git.commit(
                f"feat: add generated tool {result.name}",
                paths=[result.path, result.test_path, self.config.state_path]
                if result.path and result.test_path
                else [self.config.state_path],
            )
            self.log(f"Tool written and accepted: {result.name}")
        else:
            self.git.commit(
                f"chore: record rejected tool {result.name}",
                paths=[self.config.failed_tools_dir] if self.config.failed_tools_dir.exists() else None,
            )
            self.log(f"Tool rejected: {result.name} ({result.reason})")
        return result

    # ------------------------------------------------------------------
    # Reload after rollback
    # ------------------------------------------------------------------
    def reload(self) -> None:
        """Reload every persisted component (used after Git rollback)."""
        self.skills = SkillLibrary(self.config.skill_library_path)
        self.genetics = GeneticEvolution(
            self.config.genomes_path,
            population_size=self.config.population_size,
            seed=self.config.evolution_seed,
        )
        self.meta = MetaEvolution(self.genetics, every=self.config.genetic_evolve_every)
        self.memory = get_vector_memory(
            self.config.vector_backend, self.workspace, self.embedder
        )
        self.registry = ToolRegistry()
        self.registry.load_builtin_tools()
        self.registry.load_generated_tools(self.config.tools_dir)
        self.state = self._load_state()
        self.log("Agent state, skills, tools, and genomes reloaded")

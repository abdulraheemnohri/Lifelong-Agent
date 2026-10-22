"""Genetic algorithm that evolves agent strategy genomes.

Each genome carries a list of strategy principles, a running-average
fitness updated from task scores, and optional personality traits
(caution, curiosity, tool_usage, verification, minimalism, memory_usage).
The best genome becomes the agent's active strategy.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..utils.json_utils import read_json, write_json
from ..utils.logging import get_logger

logger = get_logger("genetic")

PRINCIPLES: list[str] = [
    "Verify every result.",
    "Prefer reversible actions.",
    "Use tools when they reduce risk.",
    "Retrieve relevant memory before acting.",
    "Break tasks into smaller steps.",
    "Prefer minimal changes.",
    "Test before and after changes.",
    "Avoid destructive operations.",
    "Ask for clarification when uncertain.",
    "Record lessons after every task.",
]

DEFAULT_TRAITS: dict[str, float] = {
    "caution": 0.5,
    "curiosity": 0.5,
    "tool_usage": 0.5,
    "verification": 0.5,
    "minimalism": 0.5,
    "memory_usage": 0.5,
}

MAX_PRINCIPLES = 6


@dataclass
class Genome:
    """A candidate strategy genome."""

    id: str
    strategy: list[str]
    fitness: float = 0.0
    tasks: int = 0
    generation: int = 0
    traits: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TRAITS))

    def strategy_text(self) -> str:
        return "\n".join(f"- {principle}" for principle in self.strategy)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Genome":
        traits = dict(DEFAULT_TRAITS)
        traits.update({k: float(v) for k, v in (data.get("traits") or {}).items()})
        return cls(
            id=str(data["id"]),
            strategy=[str(s) for s in data.get("strategy", [])],
            fitness=float(data.get("fitness", 0.0)),
            tasks=int(data.get("tasks", 0)),
            generation=int(data.get("generation", 0)),
            traits=traits,
        )


class GeneticEvolution:
    """Evolves a population of strategy genomes across generations."""

    def __init__(
        self,
        path: Path,
        population_size: int = 8,
        seed: int = 42,
        elite_fraction: float = 0.25,
        mutation_rate: float = 0.7,
    ) -> None:
        self.path = Path(path)
        self.population_size = max(2, population_size)
        self.rng = random.Random(seed)
        self.elite_fraction = elite_fraction
        self.mutation_rate = mutation_rate
        self.population: list[Genome] = []
        self._counter = 0
        self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _new_id(self) -> str:
        self._counter += 1
        return f"g{self._counter}"

    def seed_population(self) -> None:
        """Create the initial random population from the principle pool."""
        self.population = []
        for _ in range(self.population_size):
            size = self.rng.randint(3, min(5, len(PRINCIPLES)))
            strategy = self.rng.sample(PRINCIPLES, size)
            genome = Genome(
                id=self._new_id(),
                strategy=strategy,
                traits={k: self.rng.random() for k in DEFAULT_TRAITS},
            )
            self.population.append(genome)
        self.save()
        logger.info("Seeded population of %d genomes", len(self.population))

    def load(self) -> None:
        data = read_json(self.path, default=None)
        if isinstance(data, dict) and isinstance(data.get("population"), list):
            self.population = [
                Genome.from_dict(item) for item in data["population"]
            ]
            for genome in self.population:
                try:
                    self._counter = max(self._counter, int(genome.id[1:]))
                except (ValueError, IndexError):
                    continue
        if not self.population:
            self.seed_population()

    def save(self) -> None:
        write_json(
            self.path,
            {"population": [genome.to_dict() for genome in self.population]},
        )

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------
    def best(self) -> Genome:
        """Highest-fitness genome (ties broken by fewer tasks, then by id)."""
        return max(
            self.population,
            key=lambda g: (g.fitness, -g.tasks, g.id),
        )

    def get(self, genome_id: str) -> Genome | None:
        for genome in self.population:
            if genome.id == genome_id:
                return genome
        return None

    def record_task(self, genome_id: str, score: float) -> None:
        """Fold a task score into the genome's running-average fitness."""
        genome = self.get(genome_id) or self.best()
        genome.fitness = (genome.fitness * genome.tasks + score) / (genome.tasks + 1)
        genome.tasks += 1
        self.save()

    # ------------------------------------------------------------------
    # Genetic operators
    # ------------------------------------------------------------------
    def mutate(self, genome: Genome) -> Genome:
        """Return a mutated child: add/remove/replace/shuffle a principle."""
        strategy = list(genome.strategy)
        available = [p for p in PRINCIPLES if p not in strategy]
        operation = self.rng.choice(["add", "remove", "replace", "shuffle"])

        if operation == "add" or (not available and operation == "replace"):
            if available:
                position = self.rng.randint(0, len(strategy))
                strategy.insert(position, self.rng.choice(available))
        elif operation == "remove" and len(strategy) > 1:
            strategy.pop(self.rng.randrange(len(strategy)))
        elif operation == "replace":
            if available and strategy:
                index = self.rng.randrange(len(strategy))
                strategy[index] = self.rng.choice(available)
        elif operation == "shuffle" and len(strategy) > 1:
            self.rng.shuffle(strategy)

        traits = {
            key: min(1.0, max(0.0, value + self.rng.uniform(-0.05, 0.05)))
            for key, value in genome.traits.items()
        }
        child = Genome(
            id=self._new_id(),
            strategy=strategy,
            generation=genome.generation + 1,
            traits=traits,
        )
        return child

    def crossover(self, parent_a: Genome, parent_b: Genome) -> Genome:
        """Combine principles from two parents (unique, capped)."""
        combined = list(parent_a.strategy)
        for principle in parent_b.strategy:
            if principle not in combined:
                combined.append(principle)
        if len(combined) > MAX_PRINCIPLES:
            combined = combined[:MAX_PRINCIPLES]
        traits = {
            key: (parent_a.traits.get(key, 0.5) + parent_b.traits.get(key, 0.5)) / 2.0
            for key in DEFAULT_TRAITS
        }
        return Genome(
            id=self._new_id(),
            strategy=combined,
            generation=max(parent_a.generation, parent_b.generation) + 1,
            traits=traits,
        )

    # ------------------------------------------------------------------
    # Evolution step
    # ------------------------------------------------------------------
    def evolve(self) -> Genome:
        """One generation: elitism + crossover/mutation; returns the best."""
        ranked = sorted(self.population, key=lambda g: g.fitness, reverse=True)
        elite_count = max(1, int(round(self.population_size * self.elite_fraction)))
        elites = ranked[:elite_count]

        children: list[Genome] = [Genome.from_dict(elite.to_dict()) for elite in elites]
        parent_pool = ranked[: max(2, len(ranked) // 2)]

        while len(children) < self.population_size:
            if len(parent_pool) >= 2:
                parent_a, parent_b = self.rng.sample(parent_pool, 2)
                child = self.crossover(parent_a, parent_b)
            else:
                child = Genome.from_dict(self.best().to_dict())
            if self.rng.random() < self.mutation_rate:
                child = self.mutate(child)
            children.append(child)

        self.population = children
        self.save()
        best = self.best()
        logger.info(
            "Evolved generation %d: best=%s fitness=%.3f",
            best.generation, best.id, best.fitness,
        )
        return best

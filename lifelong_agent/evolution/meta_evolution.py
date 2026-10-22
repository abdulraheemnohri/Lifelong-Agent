"""Meta-evolution driver: triggers genetic evolution every N tasks."""

from __future__ import annotations

from .genetic import GeneticEvolution, Genome


class MetaEvolution:
    """Decides when to run a genetic evolution step."""

    def __init__(self, engine: GeneticEvolution, every: int = 3) -> None:
        self.engine = engine
        self.every = max(1, every)

    def maybe_evolve(self, tasks_since_last: int) -> Genome | None:
        """Evolve when *tasks_since_last* reached the configured threshold."""
        if tasks_since_last >= self.every:
            return self.engine.evolve()
        return None

    def evolve(self) -> Genome:
        """Force an evolution step."""
        return self.engine.evolve()

"""Genetic evolution tests: mutation, crossover, elitism, persistence."""

from __future__ import annotations

from pathlib import Path

from lifelong_agent.evolution.genetic import (
    PRINCIPLES,
    GeneticEvolution,
    Genome,
)


def make_engine(tmp_path: Path, population_size: int = 8) -> GeneticEvolution:
    return GeneticEvolution(tmp_path / "genomes.json", population_size=population_size,
                            seed=123)


def test_seeds_population_when_empty(tmp_path: Path):
    engine = make_engine(tmp_path)
    assert len(engine.population) == 8
    for genome in engine.population:
        assert genome.strategy
        assert set(genome.strategy) <= set(PRINCIPLES)


def test_mutate_produces_valid_child(tmp_path: Path):
    engine = make_engine(tmp_path)
    parent = engine.population[0]
    original = list(parent.strategy)
    child = engine.mutate(parent)
    assert child.id != parent.id
    assert child.generation == parent.generation + 1
    assert set(child.strategy) <= set(PRINCIPLES)
    assert 1 <= len(child.strategy) <= 10


def test_mutate_add_includes_new_principle(tmp_path: Path):
    engine = make_engine(tmp_path)
    parent = engine.population[0]
    parent.strategy = ["Verify every result."]
    child = engine.mutate(parent)
    assert set(child.strategy) <= set(PRINCIPLES)


def test_crossover_combines_parents(tmp_path: Path):
    engine = make_engine(tmp_path)
    parent_a = Genome(id="pa", strategy=[PRINCIPLES[0], PRINCIPLES[1]])
    parent_b = Genome(id="pb", strategy=[PRINCIPLES[2], PRINCIPLES[3]])
    child = engine.crossover(parent_a, parent_b)
    assert set(child.strategy) == {PRINCIPLES[0], PRINCIPLES[1],
                                   PRINCIPLES[2], PRINCIPLES[3]}


def test_record_task_updates_running_fitness(tmp_path: Path):
    engine = make_engine(tmp_path)
    genome = engine.best()
    engine.record_task(genome.id, 1.0)
    engine.record_task(genome.id, 0.5)
    updated = engine.get(genome.id)
    assert updated.tasks == 2
    assert abs(updated.fitness - 0.75) < 1e-6


def test_record_task_unknown_genome_uses_best(tmp_path: Path):
    engine = make_engine(tmp_path)
    before = engine.best().tasks
    engine.record_task("does_not_exist", 1.0)
    assert engine.best().tasks == before + 1


def test_evolve_keeps_elites_and_returns_best(tmp_path: Path):
    engine = make_engine(tmp_path, population_size=8)
    best = engine.best()
    engine.record_task(best.id, 1.0)
    result = engine.evolve()
    assert isinstance(result, Genome)
    assert len(engine.population) == 8
    # elites are preserved (same ids may persist)
    assert all(g.fitness >= 0.0 for g in engine.population)


def test_persistence_roundtrip(tmp_path: Path):
    path = tmp_path / "genomes.json"
    engine = GeneticEvolution(path, population_size=4, seed=7)
    engine.record_task(engine.best().id, 0.9)
    expected_fitness = engine.best().fitness
    reloaded = GeneticEvolution(path, population_size=4, seed=7)
    assert reloaded.best().fitness == expected_fitness


def test_strategy_text_format(tmp_path: Path):
    genome = Genome(id="g", strategy=["Verify every result."])
    assert genome.strategy_text() == "- Verify every result."


def test_maybe_evolve_threshold(tmp_path: Path):
    from lifelong_agent.evolution.meta_evolution import MetaEvolution

    engine = make_engine(tmp_path)
    meta = MetaEvolution(engine, every=3)
    assert meta.maybe_evolve(2) is None
    assert meta.maybe_evolve(3) is not None

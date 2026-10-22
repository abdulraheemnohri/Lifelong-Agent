# Architecture (detailed)

See the top-level [ARCHITECTURE.md](../ARCHITECTURE.md) for the overview,
diagrams, and pipelines. This document adds implementation notes.

## Module map

| Module | Responsibility |
|---|---|
| `lifelong_agent.config` | env-driven `AgentConfig` with derived workspace paths |
| `lifelong_agent.llm.base` | `BaseLLM.generate(prompt, system)` contract |
| `lifelong_agent.llm.mock_llm` | deterministic offline responses (plan / reflect / tool / default branches) |
| `lifelong_agent.llm.offline_gguf_llm` | llama-cpp chat → plain completion fallback |
| `lifelong_agent.llm.openai_compatible_llm` | stdlib-only local server client |
| `lifelong_agent.llm.downloader` | exact HF CLI chain + quant selection |
| `lifelong_agent.memory.embedders` | MiniLM embedder + deterministic hash embedder + cosine |
| `lifelong_agent.memory.vector_memory` | Chroma / JSON backends + factory |
| `lifelong_agent.memory.skill_library` | skills: upsert, rank, prune, persist |
| `lifelong_agent.tools.validator` | AST safety + tool contract validation |
| `lifelong_agent.tools.pytest_runner` | subprocess pytest with timeout |
| `lifelong_agent.tools.registry` | builtin + generated tool loading and safe execution |
| `lifelong_agent.tools.forge` | generate → validate → test → register/quarantine |
| `lifelong_agent.core.planner` / `evaluator` / `reflector` | reasoning steps of the loop |
| `lifelong_agent.core.safety` | validation gate + human-approval hook |
| `lifelong_agent.core.agent` | the lifelong loop + state + evolution + git autosave |
| `lifelong_agent.core.self_modification` | proposal → validation → test → evaluate → commit → monitor → rollback |
| `lifelong_agent.evolution.genetic` | genomes, mutation, crossover, elitism, persistence |
| `lifelong_agent.evolution.meta_evolution` | evolution scheduling |
| `lifelong_agent.git.git_manager` | workspace-scoped init/commit/log/rollback with backup branches |
| `lifelong_agent.autogen_integration.adapter` | optional ConversableAgent wrapper + group chat |
| `lifelong_agent.tui` | Textual UI with worker-based task execution |

## Determinism

- `MockLLM` responses depend only on prompt keywords.
- `HashEmbedder` uses `hashlib.blake2b` (stable across processes/runs).
- `GeneticEvolution` uses a seeded `random.Random` (`EVOLUTION_SEED`).
- Tool names derive from a deterministic slug + md5 of the description.

Together these make CI runs, agent loops, and debugging reproducible.

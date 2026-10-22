# Extension guide

## Add a builtin tool

Create `lifelong_agent/builtin_tools/<name>.py` following the contract:

```python
TOOL_NAME = "my_tool"
TOOL_DESCRIPTION = "What it does, in plain words; used for tool matching."

def run(payload: str = "", **kwargs):
    return {"ok": True, "output": "result"}
```

The registry discovers and registers it automatically on startup.

## Add an LLM backend

1. Subclass `lifelong_agent.llm.base.BaseLLM` and implement
   `generate(prompt, system="") -> str`.
2. Register it in `lifelong_agent/llm/__init__.py::get_llm`.

## Add an embedding or vector backend

- Embedders: subclass `BaseEmbedder` in
  `lifelong_agent/memory/embedders.py` and extend `get_embedder`.
- Vector stores: subclass `BaseVectorMemory` in
  `lifelong_agent/memory/vector_memory.py` and extend `get_vector_memory`
  (keep the `add(text, metadata) / search(query, k) / count()` interface).

## Add a reasoning step to the loop

The loop lives in `lifelong_agent/core/agent.py::run_task`. Insert new steps
(retrieval, verification, planning improvements) around the existing
`Planner → execute → Evaluator → Reflector` sequence and record the extra
data in the memory metadata.

## Add evolution traits

Extend `DEFAULT_TRAITS` in `lifelong_agent/evolution/genetic.py`; traits are
blended on crossover and jittered on mutation. To influence behavior, read
the active genome's traits inside the planner or tool selector.

## Add a self-modification proposal kind

Extend `SelfModificationEngine.apply` in
`lifelong_agent/core/self_modification.py` with a new `Proposal.kind`,
keeping the pipeline: validation → testing → evaluation → commit →
monitoring → rollback.

## Add a benchmark suite

Wrap `LifelongAgent.run_loop` with a set of tasks + assertions over the
resulting state (tasks_completed, mean score, tools written) and evolve the
genomes across runs; the deterministic seed makes runs comparable.

## Skill marketplace format

Skills serialize to plain JSON (`skill_library.json`) with stable fields
(`id`, `name`, `description`, `score`, `uses`, `created_at`, `updated_at`,
`keywords`) — publish and import skill files by copying them between
workspaces and calling `SkillLibrary.upsert`.

## CI

The included workflow (`.github/workflows/ci.yml`) runs pytest on
3.10/3.11/3.12. Add a lint job or matrix entries there.

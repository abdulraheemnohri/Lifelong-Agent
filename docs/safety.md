# Safety (detailed)

See the top-level [SAFETY.md](../SAFETY.md) for the threat model and the
production checklist. This document records the enforcement points in code.

## Enforcement points

1. **`lifelong_agent/tools/validator.py`** — AST walk rejecting forbidden
   imports (`os`, `sys`, `subprocess`, `pathlib`, `shutil`, `socket`,
   `requests`, `urllib`, `importlib`, `ctypes`, `multiprocessing`, relative
   imports), forbidden calls (`eval`, `exec`, `compile`, `__import__`,
   `open`, `input`, `breakpoint`, `globals`, `locals`, `getattr`, `setattr`,
   `delattr`), forbidden attributes (`system`, `popen`, `subprocess`,
   `rmtree`, `remove`, `unlink`, `write_text`, `read_text`, `environ`), and
   enforcing the tool contract (`TOOL_NAME`, `TOOL_DESCRIPTION`, `run`).

2. **`lifelong_agent/tools/forge.py`** — no tool is registered unless
   validation passes AND its generated pytest suite passes in a subprocess.
   Rejected artifacts are quarantined in `tools_generated/failed/` with a
   reason JSON and are never imported.

3. **`lifelong_agent/tools/registry.py`** — tool execution catches all
   exceptions and always returns dicts; a crashing tool can never take down
   the agent loop.

4. **`lifelong_agent/core/safety.py`** — central `SafetyGuard` with an
   injectable human-approval callback; `ENABLE_CORE_SELF_MODIFICATION`
   defaults to `false` so the agent cannot edit its own core source.

5. **`lifelong_agent/git/git_manager.py`** — workspace guard (never the
   filesystem anchor or home directory), automatic `.gitignore` protection
   for model/vector/cache artifacts, and a mandatory backup branch before
   every `git reset --hard`.

6. **`lifelong_agent/core/self_modification.py`** — the monitoring step
   compares recent task scores against the baseline and triggers Git
   rollback + full reload when performance degrades.

## Limits of static validation

AST checks stop *known* patterns; they cannot prove the absence of novel
escape vectors (e.g. exotic attribute chains, generator tricks, or resource
exhaustion). Treat the validator as a gate, not a boundary. The boundary is
the container (Docker/Podman/gVisor/Firecracker).

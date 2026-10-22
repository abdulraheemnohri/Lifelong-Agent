# Safety

## ⚠️ Core warning

**AST validation is NOT a full sandbox.** The static validator blocks known
dangerous patterns, but determined or novel code may bypass static analysis.
For any production or untrusted deployment, run the agent (and especially its
generated tools) inside a real sandbox: **Docker, Podman, gVisor, Firecracker,
or an equivalent isolation layer.** This repository ships a hardened
`Dockerfile` and `docker-compose.yml` for that purpose.

## Threat model

| Threat | Vector | Mitigation |
|---|---|---|
| Malicious generated tool | LLM proposes unsafe code | AST validator bans dangerous imports/calls/attributes; mandatory pytest gate; failures quarantined in `tools_generated/failed/` |
| Code execution escape | static checks bypassed | container sandbox required for production; non-root Docker user; `no-new-privileges` |
| Destructive filesystem access | tool tries `os`/`shutil`/paths | forbidden-import list includes `os`, `sys`, `pathlib`, `shutil`, `subprocess`, `ctypes`, ... |
| Network exfiltration | tool tries `requests`/`urllib`/`socket` | all banned in generated tools; container network policy recommended |
| Unsafe self-modification | agent edits its own core | core self-modification off by default (`ENABLE_CORE_SELF_MODIFICATION=false`); pipeline is validate → test → evaluate → commit → monitor → rollback |
| Irreversible rollback | `git reset --hard` destroys work | backup branch created before every reset; rollback limited to the agent workspace |
| Secret leakage | tokens in code/state | `HF_TOKEN` via env only; `.gitignore` excludes `.env`, keys, credentials; never commit model artifacts or vector DBs |

## Generated tool contract

```python
TOOL_NAME = "generated_tool_xyz"
TOOL_DESCRIPTION = "Description here."

def run(payload: str = "", **kwargs):
    return {"ok": True, "output": "result"}
```

Banned in generated tools:

- **Imports**: `os`, `sys`, `subprocess`, `pathlib`, `shutil`, `socket`,
  `requests`, `urllib`, `importlib`, `ctypes`, `multiprocessing`, relative imports
- **Calls**: `eval`, `exec`, `compile`, `__import__`, `open`, `input`,
  `breakpoint`, `globals`, `locals`, `getattr`, `setattr`, `delattr`
- **Attributes**: `system`, `popen`, `subprocess`, `rmtree`, `remove`,
  `unlink`, `write_text`, `read_text`, `environ`

Every accepted tool must pass its **generated pytest suite** (run in a
subprocess) before it is imported and registered. Rejected tools are moved to
`tools_generated/failed/` with a machine-readable reason file and are never
loaded.

## Git rollback risks

- Rollback uses `git reset --hard` **inside the agent workspace only** and
  only after creating an `agent-backup-<timestamp>` branch.
- Uncommitted work is destroyed by design — commit meaningful milestones
  frequently (the agent does this automatically after tasks and evolutions).
- After rollback the agent fully reloads state, skills, tools, genomes, and
  memory from the restored tree.

## Human approval mode

Set `HUMAN_APPROVAL_MODE=true` (or register an approval callback, e.g. from
the TUI) to require explicit approval for high-risk actions such as strategy
evolution or configuration changes. Tool generation remains automated but is
still gated by validation + tests.

## Production checklist

- [ ] Run inside the provided Docker sandbox (non-root, `no-new-privileges`).
- [ ] Restrict/Disable container network access unless explicitly needed.
- [ ] Mount the workspace as a volume; keep the repo read-only.
- [ ] Keep `ENABLE_CORE_SELF_MODIFICATION=false`.
- [ ] Set `HF_TOKEN` via environment/secret, never in files.
- [ ] Confirm `.gitignore` excludes models, vector DBs, caches, secrets.
- [ ] Run `pytest` in CI on every push (workflow included).
- [ ] Review the Git log and `tools_generated/failed/` regularly.
- [ ] Back up the workspace volume (state + git history).
- [ ] Monitor task scores after every self-modification (auto-rollback on degradation).

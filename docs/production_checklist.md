# Production safety checklist

Run through this list before exposing the agent to anything untrusted.

## Sandboxing

- [ ] Agent runs inside the provided Docker/Podman sandbox (never bare metal).
- [ ] Container runs as non-root with `no-new-privileges`.
- [ ] Container network access restricted or disabled unless required.
- [ ] Workspace mounted as a volume; repository mounted read-only.
- [ ] Consider gVisor/Firecracker for stronger isolation of generated tools.

## Configuration

- [ ] `ENABLE_CORE_SELF_MODIFICATION` stays `false`.
- [ ] `HUMAN_APPROVAL_MODE=true` for deployments needing sign-off.
- [ ] `ENABLE_SELF_WRITING_TOOLS=false` if tool generation is not required.
- [ ] Secrets (`HF_TOKEN`, API keys) come from the environment/secrets store.
- [ ] `.gitignore` verified: no models, vector DBs, caches, secrets committed.

## Validation

- [ ] `pytest` passes in CI on every push.
- [ ] No generated tool is loaded without validation + pytest (enforced in code).
- [ ] `tools_generated/failed/` reviewed periodically.
- [ ] Git log reviewed after self-modification sessions.

## Recovery

- [ ] Workspace volume backed up (contains git history + state).
- [ ] Rollback tested: `python scripts/rollback.py` restores previous state
      and the agent reloads cleanly.
- [ ] Performance monitoring after changes (auto-rollback on degradation is
      built into the self-modification engine).

## Incident readiness

- [ ] Someone can interpret `agent_state.json`, `genomes.json`, and the git
      history on short notice.
- [ ] Backup branches (`agent-backup-*`) are preserved, not pruned casually.

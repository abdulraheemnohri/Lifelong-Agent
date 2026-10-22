# Deployment guide

## Local (development)

```bash
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
pytest
python scripts/download_model.py
python -m lifelong_agent.tui
```

## Docker sandbox (recommended)

The provided `Dockerfile` and `docker-compose.yml` implement the production
sandbox: non-root user, `no-new-privileges`, isolated workspace volume, and
optional read-only model mount.

```bash
# build and run headless
docker compose up --build

# run the TUI inside the container (needs a TTY)
docker compose run --rm lifelong-agent python /app/scripts/run_tui.py
```

Mount a pre-downloaded model instead of downloading inside the container by
uncommenting the model volume in `docker-compose.yml`.

## Environment

Pass backend switches via environment variables (see USAGE.md), e.g.:

```yaml
environment:
  - LLM_BACKEND=gguf
  - GGUF_N_GPU_LAYERS=0
  - EMBEDDER_BACKEND=hash
  - VECTOR_BACKEND=json
  - ENABLE_GENETIC_EVOLUTION=true
  - HF_TOKEN=${HF_TOKEN}
```

## Persistence layout (inside the workspace)

| Path | Content |
|---|---|
| `agent_state.json` | lifelong agent state (strategy, counters, reflections) |
| `skill_library.json` | distilled skills |
| `genomes.json` | genetic population |
| `vector_db/` or `vector_memory_*.json` | vector memory |
| `tools_generated/` | accepted generated tools + tests |
| `tools_generated/failed/` | rejected tools + reason files |
| `.git/` | self-modification history |
| `Youtu-LLM-2B-GGUF/` | downloaded model artifacts (git-ignored) |

## CI

`.github/workflows/ci.yml` runs the full offline pytest suite on Python
3.10/3.11/3.12 for every push and pull request.

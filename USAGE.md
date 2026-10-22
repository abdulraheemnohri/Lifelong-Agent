# Usage

## CLI commands

| Command | Purpose |
|---|---|
| `pip install -r requirements.txt` | install core dependencies |
| `pip install -r requirements-optional.txt` | optional backends (GGUF, chroma, ST, autogen) |
| `python scripts/download_model.py` | download the offline GGUF model |
| `python scripts/download_model.py --list-only` | show found GGUF files |
| `python -m lifelong_agent.tui` | launch the TUI |
| `python scripts/run_headless.py --task "..."` | run one task headless |
| `python scripts/run_headless.py --loop --tasks "a,b,c"` | run several tasks |
| `python -m lifelong_agent.main --backend mock --task "..."` | force a backend |
| `python scripts/rollback.py` | roll back to `HEAD~1` and reload |
| `python scripts/rollback.py --commit <sha>` | roll back to a specific commit |
| `pytest` | run the offline test suite |
| `docker compose up --build` | run inside the sandbox container |

## TUI controls

- **Input**: type a task; comma-separated input = multiple tasks.
- **Run Task**: run the input as a single task (Enter also works).
- **Run Loop**: run all comma-separated tasks in sequence.
- **Meta Evolve**: force a genetic evolution step and activate the best genome.
- **Git Rollback**: backup branch + reset to `HEAD~1` + full agent reload.
- **Quit**: exit (`q` binding also works).
- **Tabs**:
  - *Log* — real-time agent log stream
  - *Skills* — current skill library (name, score, uses)
  - *Tools* — registered builtin + generated tools
  - *Memory* — stored experience count
  - *Git* — recent commit history

Tasks run in Textual workers, so the UI stays responsive while the agent
works.

## Configuration options (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `WORKSPACE` | current directory | agent workspace |
| `MODEL_REPO` | `tencent/Youtu-LLM-2B` | Hugging Face repo |
| `MODEL_REVISION` | `GGUF` | repo revision/branch |
| `MODEL_DIR` | `Youtu-LLM-2B-GGUF` | local model directory |
| `LLM_BACKEND` | `auto` | `auto` \| `gguf` \| `mock` \| `openai_compatible` |
| `GGUF_N_CTX` | `4096` | context window |
| `GGUF_N_GPU_LAYERS` | `0` | GPU layers (0 = CPU) |
| `GGUF_N_THREADS` | `4` | CPU threads |
| `GGUF_MAX_TOKENS` | `512` | generation limit |
| `GGUF_TEMPERATURE` | `0.7` | sampling temperature |
| `OPENAI_BASE_URL` | `http://127.0.0.1:8000/v1` | local server endpoint |
| `EMBEDDER_BACKEND` | `auto` | `auto` \| `sentence-transformer` \| `hash` |
| `VECTOR_BACKEND` | `auto` | `auto` \| `chroma` \| `json` |
| `ENABLE_SELF_WRITING_TOOLS` | `true` | tool generation on/off |
| `ENABLE_GENETIC_EVOLUTION` | `true` | evolution on/off |
| `ENABLE_GIT_SELF_MODIFICATION` | `true` | git tracking on/off |
| `ENABLE_TUI` | `true` | TUI availability |
| `ENABLE_CORE_SELF_MODIFICATION` | `false` | DANGEROUS: core code edits |
| `HUMAN_APPROVAL_MODE` | `false` | require approval for high-risk actions |
| `SUCCESS_THRESHOLD` | `0.75` | minimum task score |
| `MAX_TASK_ATTEMPTS` | `2` | retry count per task |
| `META_EVOLVE_EVERY` | `3` | tasks between evolutions |
| `GENETIC_EVOLVE_EVERY` | `3` | evolution threshold |
| `POPULATION_SIZE` | `8` | genome population size |
| `EVOLUTION_SEED` | `42` | deterministic evolution |
| `HF_TOKEN` | — | Hugging Face token for gated repos |

## Example tasks

```text
count the words in this sentence
reverse the payload string safely
summarize the feedback and record a lesson
Debug a failing Python test
uppercase the payload text
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `huggingface-cli` not found | the downloader automatically falls back to `python -c` and in-process `snapshot_download`; or `pip install huggingface-hub` |
| No GGUF model, agent still works | expected: `auto` backend falls back to `MockLLM`; download the model for real inference |
| `llama-cpp-python` install fails | requires a C++ toolchain; the agent works with MockLLM without it |
| `chromadb` / `sentence-transformers` missing | JSON vector memory and hash embeddings are used automatically |
| TUI won't start | `pip install textual`, or use the headless CLI |
| AutoGen features disabled | `pip install pyautogen` |
| Rollback did nothing | there was no previous commit, or git is unavailable |
| Generated tool disappeared | it failed validation or pytest; check `tools_generated/failed/` and its `*.reason.json` |
| Gated model download 401 | set `HF_TOKEN` in the environment |

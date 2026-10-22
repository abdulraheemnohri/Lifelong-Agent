"""Headless entry point for the lifelong agent.

Usage::

    python -m lifelong_agent.main --task "Debug a failing Python test"
    python -m lifelong_agent.main --loop --tasks "task one,task two"
    python -m lifelong_agent.main --tui
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lifelong_agent", description="Lifelong self-evolving offline agent"
    )
    parser.add_argument("--task", help="Run a single task")
    parser.add_argument("--loop", action="store_true", help="Run a list of tasks in a loop")
    parser.add_argument("--tasks", help="Comma-separated list of tasks")
    parser.add_argument("--tui", action="store_true", help="Launch the Textual TUI")
    parser.add_argument(
        "--backend", default=None,
        help="LLM backend override: auto | gguf | mock | openai_compatible",
    )
    parser.add_argument("--workspace", default=None, help="Workspace directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.tui or not (args.task or args.tasks):
        from .tui import main as tui_main

        tui_main()
        return 0

    from .config import AgentConfig
    from .core.agent import LifelongAgent

    config = AgentConfig.from_env(Path(args.workspace) if args.workspace else None)
    if args.backend:
        config.llm_backend = args.backend
    agent = LifelongAgent(config)

    if args.task:
        results = [agent.run_task(args.task)]
    else:
        tasks = [t.strip() for t in (args.tasks or "").split(",") if t.strip()]
        results = agent.run_loop(tasks)

    print(json.dumps({"results": results}, indent=2, default=str))
    return 0 if all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

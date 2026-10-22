#!/usr/bin/env python
"""Roll the agent workspace back to a previous Git commit and reload state.

Usage::

    python scripts/rollback.py                # roll back to HEAD~1
    python scripts/rollback.py --commit abc123
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lifelong_agent.config import AgentConfig  # noqa: E402
from lifelong_agent.core.agent import LifelongAgent  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Git rollback with agent reload")
    parser.add_argument("--commit", default=None, help="Target commit (default HEAD~1)")
    parser.add_argument("--workspace", default=None, help="Workspace directory")
    args = parser.parse_args()

    config = AgentConfig.from_env(Path(args.workspace) if args.workspace else None)
    agent = LifelongAgent(config)

    target = agent.git.rollback(commit=args.commit)
    if target is None:
        print("Rollback failed: no previous commit or git unavailable.", file=sys.stderr)
        return 1
    agent.reload()
    print(f"Rolled back to {target}")
    print(f"Strategy version: {agent.state['strategy_version']}")
    print(f"Tasks attempted:  {agent.state['tasks_attempted']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

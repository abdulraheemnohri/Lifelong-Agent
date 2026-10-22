#!/usr/bin/env python
"""Headless runner for the lifelong agent.

Usage::

    python scripts/run_headless.py --task "Debug a failing Python test"
    python scripts/run_headless.py --loop --tasks "task one,task two"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lifelong_agent.main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Launch the Textual TUI.

Usage::

    python scripts/run_tui.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lifelong_agent.tui import main  # noqa: E402

if __name__ == "__main__":
    main()

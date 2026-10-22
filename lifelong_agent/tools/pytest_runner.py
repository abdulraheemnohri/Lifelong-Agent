"""Run pytest on generated tool test files in an isolated subprocess."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestResult:
    """Outcome of a pytest run."""

    ok: bool
    returncode: int
    output: str


def run_pytest(test_path: Path, timeout: int = 120, cwd: Path | None = None) -> TestResult:
    """Run ``python -m pytest <test_path>`` and report the outcome."""
    test_path = Path(test_path)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        str(test_path),
    ]
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd or test_path.parent),
            capture_output=True,
            text=True,
            timeout=max(30, timeout),
            env=env,
        )
    except subprocess.TimeoutExpired:
        return TestResult(ok=False, returncode=-1, output=f"pytest timed out after {timeout}s")
    except OSError as exc:
        return TestResult(ok=False, returncode=-1, output=f"failed to run pytest: {exc}")

    output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    return TestResult(ok=completed.returncode == 0, returncode=completed.returncode,
                      output=output.strip())

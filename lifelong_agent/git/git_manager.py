"""Git-based self-modification history with backup-branch rollback.

Safety rules:
- Operates only inside the designated project workspace.
- Refuses to run against the filesystem anchor or the user's home directory.
- Creates a backup branch before every rollback.
- Uses ``git reset --hard`` only inside the workspace.
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..utils.logging import get_logger

logger = get_logger("git")

GITIGNORE_LINES: list[str] = [
    "Youtu-LLM-2B-GGUF/",
    "vector_db/",
    "__pycache__/",
    "*.pyc",
    ".pytest_cache/",
    "*.log",
    ".agent_cache/",
]


class GitManager:
    """Manage a Git repository scoped to a single workspace directory."""

    def __init__(self, workspace: Path, enabled: bool = True) -> None:
        self.workspace = Path(workspace).resolve()
        self._enabled = bool(enabled) and shutil.which("git") is not None
        if not self._enabled and enabled:
            logger.warning("Git unavailable or disabled; self-modification tracking is off")
        self._guard()

    # ------------------------------------------------------------------
    # Guards and plumbing
    # ------------------------------------------------------------------
    def _guard(self) -> None:
        anchor = Path(self.workspace.anchor)
        home = Path.home()
        if self.workspace == anchor or self.workspace == home:
            raise RuntimeError(
                f"Refusing to manage Git in unsafe workspace: {self.workspace}"
            )

    def _run(
        self, *args: str, check: bool = True
    ) -> subprocess.CompletedProcess:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(self.workspace),
            capture_output=True,
            text=True,
        )
        if check and completed.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args)} failed: {(completed.stderr or '').strip()}"
            )
        return completed

    @property
    def enabled(self) -> bool:
        return self._enabled

    def is_repo(self) -> bool:
        if not self._enabled:
            return False
        return self._run("rev-parse", "--is-inside-work-tree",
                         check=False).returncode == 0

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def init(self) -> "GitManager":
        """Initialize the workspace repo, user config, and .gitignore."""
        if not self._enabled:
            return self
        if not self.is_repo():
            self._run("init", check=False)
            logger.info("Initialized Git repository in %s", self.workspace)
        self._ensure_user_config()
        self._ensure_gitignore()
        return self

    def _ensure_user_config(self) -> None:
        if self._run("config", "user.name", check=False).returncode != 0:
            self._run("config", "user.name", "Lifelong Agent")
        if self._run("config", "user.email", check=False).returncode != 0:
            self._run("config", "user.email", "agent@lifelong.local")

    def _ensure_gitignore(self) -> None:
        gitignore = self.workspace / ".gitignore"
        existing = (
            gitignore.read_text(encoding="utf-8").splitlines()
            if gitignore.exists()
            else []
        )
        missing = [line for line in GITIGNORE_LINES if line not in existing]
        if missing:
            content = "\n".join(existing + missing) + "\n"
            gitignore.write_text(content, encoding="utf-8")
            logger.info("Updated .gitignore with %d entries", len(missing))

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------
    def commit(
        self,
        message: str,
        paths: Optional[list[Path]] = None,
        allow_empty: bool = False,
    ) -> Optional[str]:
        """Stage (selected paths or everything) and commit.

        Returns the new commit SHA, or ``None`` when there was nothing to
        commit and ``allow_empty`` is false.
        """
        if not self._enabled or not self.is_repo():
            return None

        if paths:
            for path in paths:
                resolved = Path(path).resolve()
                if self.workspace not in resolved.parents:
                    logger.warning("Skipping out-of-workspace path %s", path)
                    continue
                self._run("add", "--", str(resolved))
        else:
            self._run("add", "-A")

        staged = self._run("diff", "--cached", "--quiet", check=False).returncode != 0
        if not staged:
            if not allow_empty:
                return None
            self._run("commit", "--allow-empty", "-m", message)
        else:
            self._run("commit", "-m", message)

        sha = self._run("rev-parse", "HEAD").stdout.strip()
        logger.info("Committed %s: %s", sha[:8], message)
        return sha

    def log(self, n: int = 20) -> list[dict[str, str]]:
        """Recent commits as a list of {hash, short, date, message}."""
        if not self._enabled or not self.is_repo():
            return []
        result = self._run(
            "log", f"-{max(1, n)}", "--pretty=format:%H|%h|%ad|%s", "--date=iso",
            check=False,
        )
        entries: list[dict[str, str]] = []
        for line in result.stdout.splitlines():
            parts = line.split("|", 3)
            if len(parts) == 4:
                entries.append(
                    {
                        "hash": parts[0],
                        "short": parts[1],
                        "date": parts[2],
                        "message": parts[3],
                    }
                )
        return entries

    def previous_commit(self) -> Optional[str]:
        """SHA of HEAD~1, or ``None`` when there is no previous commit."""
        if not self._enabled or not self.is_repo():
            return None
        result = self._run("rev-parse", "--verify", "HEAD~1", check=False)
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    def branches(self, pattern: str = "*") -> list[str]:
        """List branch names matching a glob pattern."""
        if not self._enabled or not self.is_repo():
            return []
        result = self._run("branch", "--list", pattern, check=False)
        return [
            line.strip().lstrip("* ").strip()
            for line in result.stdout.splitlines()
            if line.strip()
        ]

    def rollback(self, commit: Optional[str] = None) -> Optional[str]:
        """Reset the workspace to a previous commit after backing up HEAD.

        Default target is ``HEAD~1``. Returns the target SHA, or ``None``
        when rollback was impossible.
        """
        if not self._enabled or not self.is_repo():
            logger.warning("Rollback requested but Git is unavailable")
            return None

        target = commit or "HEAD~1"
        verify = self._run("rev-parse", "--verify", f"{target}^{{commit}}", check=False)
        if verify.returncode != 0:
            logger.warning("Rollback target %r does not exist", target)
            return None
        target_sha = verify.stdout.strip()

        backup = f"agent-backup-{datetime.now():%Y%m%d-%H%M%S}"
        self._run("branch", backup, check=False)
        logger.info("Created backup branch %s before rollback", backup)

        self._run("reset", "--hard", target)
        logger.warning("Rolled back workspace to %s (%s)", target_sha[:8], target)
        return target_sha

"""GitManager tests: init, commit, log, previous commit, rollback.

Skipped automatically when the git binary is unavailable.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from lifelong_agent.git.git_manager import GITIGNORE_LINES, GitManager

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")


def make_repo(tmp_path: Path) -> GitManager:
    return GitManager(tmp_path, enabled=True).init()


def test_init_creates_gitignore(tmp_path: Path):
    make_repo(tmp_path)
    gitignore = tmp_path / ".gitignore"
    assert gitignore.exists()
    content = gitignore.read_text()
    for line in GITIGNORE_LINES:
        assert line in content


def test_commit_and_log(tmp_path: Path):
    manager = make_repo(tmp_path)
    (tmp_path / "a.txt").write_text("one")
    sha = manager.commit("add a", paths=[tmp_path / "a.txt"])
    assert sha
    (tmp_path / "b.txt").write_text("two")
    assert manager.commit("add b", paths=[tmp_path / "b.txt"])
    entries = manager.log(10)
    assert len(entries) == 2
    assert entries[0]["message"] == "add b"


def test_commit_without_changes_returns_none(tmp_path: Path):
    manager = make_repo(tmp_path)
    # init() writes a fresh .gitignore; commit it once to reach a clean tree.
    assert manager.commit("initial state") is not None
    assert manager.commit("empty") is None


def test_previous_commit(tmp_path: Path):
    manager = make_repo(tmp_path)
    (tmp_path / "a.txt").write_text("1")
    manager.commit("first", paths=[tmp_path / "a.txt"])
    assert manager.previous_commit() is None
    (tmp_path / "a.txt").write_text("2")
    manager.commit("second", paths=[tmp_path / "a.txt"])
    assert isinstance(manager.previous_commit(), str)


def test_rollback_restores_previous_state(tmp_path: Path):
    manager = make_repo(tmp_path)
    target_file = tmp_path / "a.txt"
    target_file.write_text("safe")
    manager.commit("safe state", paths=[target_file])
    target_file.write_text("broken")
    manager.commit("broken state", paths=[target_file])

    result = manager.rollback()
    assert result is not None
    assert target_file.read_text() == "safe"
    backups = manager.branches("agent-backup-*")
    assert backups, "a backup branch must be created before rollback"


def test_rollback_without_history_returns_none(tmp_path: Path):
    manager = make_repo(tmp_path)
    (tmp_path / "a.txt").write_text("only")
    manager.commit("only commit", paths=[tmp_path / "a.txt"])
    assert manager.rollback() is None


def test_refuses_unsafe_workspace(tmp_path, monkeypatch):
    from lifelong_agent.git.git_manager import GitManager as GM

    with pytest.raises(RuntimeError):
        GM(Path("/"), enabled=True)


def test_disabled_manager_is_noop(tmp_path: Path):
    manager = GitManager(tmp_path, enabled=False)
    assert manager.enabled is False
    assert manager.commit("noop") is None
    assert manager.log() == []
    assert manager.rollback() is None

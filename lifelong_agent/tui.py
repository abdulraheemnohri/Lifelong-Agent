"""Textual terminal user interface for the lifelong agent.

Run with::

    python -m lifelong_agent.tui

Tasks run inside Textual workers so the UI never blocks; logs stream into
the Log tab, and skills / tools / memory / git panels refresh after each
operation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import AgentConfig
from .utils.logging import get_logger

logger = get_logger("tui")

try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal
    from textual.widgets import (
        Button,
        DataTable,
        Header,
        Input,
        RichLog,
        TabbedContent,
        TabPane,
    )

    TEXTUAL_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional dependency
    TEXTUAL_AVAILABLE = False

CSS = """
#controls {
    height: auto;
    margin: 0 1;
}
#controls Button {
    margin: 0 1 0 0;
}
#task-input {
    margin: 0 1;
}
"""


def _build_agent(workspace: Path | None = None) -> Any:
    """Construct the lifelong agent for the TUI (isolated import)."""
    from .core.agent import LifelongAgent

    config = AgentConfig.from_env(workspace or Path.cwd())
    return LifelongAgent(config)


if TEXTUAL_AVAILABLE:

    class LifelongAgentTUI(App[None]):
        """Terminal UI for the lifelong self-evolving agent."""

        TITLE = "Lifelong Self-Evolving Agent"
        SUB_TITLE = "offline • self-writing tools • genetic evolution • git rollback"
        CSS = CSS
        BINDINGS = [("q", "quit", "Quit")]

        def __init__(self, workspace: Path | None = None) -> None:
            super().__init__()
            self._workspace = workspace
            self.agent: Any = None

        # --------------------------------------------------------------
        # Layout
        # --------------------------------------------------------------
        def compose(self) -> ComposeResult:
            yield Header()
            yield Input(
                placeholder="Enter a task (comma-separated for a loop)...",
                id="task-input",
            )
            with Horizontal(id="controls"):
                yield Button("Run Task", id="run-task", variant="primary")
                yield Button("Run Loop", id="run-loop")
                yield Button("Meta Evolve", id="meta-evolve")
                yield Button("Git Rollback", id="git-rollback", variant="error")
                yield Button("Quit", id="quit")
            with TabbedContent(initial="log"):
                with TabPane("Log", id="log"):
                    yield RichLog(id="log-log", markup=False, highlight=False)
                with TabPane("Skills", id="skills"):
                    yield DataTable(id="skills-table")
                with TabPane("Tools", id="tools"):
                    yield DataTable(id="tools-table")
                with TabPane("Memory", id="memory"):
                    yield RichLog(id="memory-log", markup=False)
                with TabPane("Git", id="git"):
                    yield RichLog(id="git-log", markup=False)

        # --------------------------------------------------------------
        # Lifecycle
        # --------------------------------------------------------------
        def on_mount(self) -> None:
            self.agent = _build_agent(self._workspace)
            self.agent.log_sink = self._thread_safe_log
            table = self.query_one("#skills-table", DataTable)
            table.add_columns("Name", "Score", "Uses")
            tools = self.query_one("#tools-table", DataTable)
            tools.add_columns("Name", "Source", "Description")
            self._thread_safe_log("Agent ready. Enter a task and press Run Task.")
            self.refresh_panels()

        def _thread_safe_log(self, message: str) -> None:
            try:
                self.call_from_thread(self._append_log, message)
            except Exception:  # noqa: BLE001 - not mounted yet or wrong thread
                self._append_log(message)

        def _append_log(self, message: str) -> None:
            log_widget = self.query_one("#log-log", RichLog)
            log_widget.write(str(message))

        # --------------------------------------------------------------
        # Panel refresh
        # --------------------------------------------------------------
        def refresh_panels(self) -> None:
            if self.agent is None:
                return
            # skills
            skills_table = self.query_one("#skills-table", DataTable)
            skills_table.clear()
            for skill in self.agent.skills.list():
                skills_table.add_row(skill.name, f"{skill.score:.2f}", str(skill.uses))
            # tools
            tools_table = self.query_one("#tools-table", DataTable)
            tools_table.clear()
            for spec in self.agent.registry.list_tools():
                tools_table.add_row(spec.name, spec.source, spec.description[:80])
            # memory
            memory_log = self.query_one("#memory-log", RichLog)
            memory_log.clear()
            memory_log.write(f"Memories stored: {self.agent.memory.count()}")
            # git
            git_log = self.query_one("#git-log", RichLog)
            git_log.clear()
            git_log.write(f"Git enabled: {self.agent.git.enabled}")
            for entry in self.agent.git.log(10):
                git_log.write(f"{entry['short']}  {entry['message'][:80]}")

        # --------------------------------------------------------------
        # Handlers
        # --------------------------------------------------------------
        def on_input_submitted(self, event: Input.Submitted) -> None:
            self._start_run(loop_mode=False)

        def on_button_pressed(self, event: Button.Pressed) -> None:
            button_id = event.button.id
            if button_id == "run-task":
                self._start_run(loop_mode=False)
            elif button_id == "run-loop":
                self._start_run(loop_mode=True)
            elif button_id == "meta-evolve":
                self.run_worker(self._worker_meta_evolve, thread=True, exclusive=True)
            elif button_id == "git-rollback":
                self.run_worker(self._worker_rollback, thread=True, exclusive=True)
            elif button_id == "quit":
                self.exit()

        def _start_run(self, loop_mode: bool) -> None:
            if self.agent is None:
                return
            text = self.query_one("#task-input", Input).value.strip()
            if not text:
                self._append_log("Enter a task first.")
                return
            tasks = [part.strip() for part in text.split(",") if part.strip()]
            self.run_worker(
                self._worker_run, tasks, loop_mode, thread=True, exclusive=True
            )

        # --------------------------------------------------------------
        # Workers (never block the UI)
        # --------------------------------------------------------------
        def _worker_run(self, tasks: list[str], loop_mode: bool) -> None:
            try:
                for task in tasks:
                    self._thread_safe_log(f">>> {'LOOP' if loop_mode else 'TASK'}: {task}")
                    result = self.agent.run_task(task)
                    self._thread_safe_log(
                        f"<<< done ok={result.get('ok')} score={result.get('score')}"
                    )
            except Exception as exc:  # noqa: BLE001
                self._thread_safe_log(f"ERROR: {exc}")
            self._thread_safe_refresh()

        def _worker_meta_evolve(self) -> None:
            try:
                summary = self.agent.meta_evolve()
                self._thread_safe_log(f"Evolved: {summary}")
            except Exception as exc:  # noqa: BLE001
                self._thread_safe_log(f"ERROR: {exc}")
            self._thread_safe_refresh()

        def _worker_rollback(self) -> None:
            try:
                self._thread_safe_log("Rolling back to previous commit...")
                target = self.agent.git.rollback()
                if target is None:
                    self._thread_safe_log("Rollback impossible (no previous commit?)")
                    return
                self.agent.reload()
                self._thread_safe_log(f"Rolled back to {target[:8]}; state reloaded")
            except Exception as exc:  # noqa: BLE001
                self._thread_safe_log(f"ERROR: {exc}")
            self._thread_safe_refresh()

        def _thread_safe_refresh(self) -> None:
            try:
                self.call_from_thread(self.refresh_panels)
            except Exception:  # noqa: BLE001
                self.refresh_panels()


def main() -> None:
    """Entry point: ``python -m lifelong_agent.tui``."""
    if not TEXTUAL_AVAILABLE:
        print(
            "The TUI requires the 'textual' package.\n"
            "Install it with: pip install textual\n"
            "Or run headless: python scripts/run_headless.py --task \"...\""
        )
        return
    LifelongAgentTUI().run()


if __name__ == "__main__":
    main()

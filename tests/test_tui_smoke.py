"""TUI smoke tests: importability, instantiation, and headless mount."""

from __future__ import annotations

import asyncio

import pytest

from lifelong_agent.tui import TEXTUAL_AVAILABLE

pytestmark = pytest.mark.skipif(
    not TEXTUAL_AVAILABLE, reason="textual is not installed"
)


def test_tui_module_reports_availability():
    assert isinstance(TEXTUAL_AVAILABLE, bool)


def test_tui_app_instantiable(tmp_path):
    from lifelong_agent.tui import LifelongAgentTUI

    app = LifelongAgentTUI(workspace=tmp_path)
    assert app.TITLE == "Lifelong Self-Evolving Agent"
    assert hasattr(app, "compose")
    assert hasattr(app, "on_button_pressed")


def test_tui_mounts_headless(tmp_path):
    from lifelong_agent.tui import LifelongAgentTUI

    async def run():
        app = LifelongAgentTUI(workspace=tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one("#task-input")
            assert app.agent is not None

    asyncio.run(run())

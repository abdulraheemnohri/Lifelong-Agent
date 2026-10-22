"""ToolRegistry tests: builtin loading, execution, generated tool loading."""

from __future__ import annotations

from pathlib import Path

from lifelong_agent.tools.registry import ToolRegistry

GENERATED = '''
TOOL_NAME = "generated_tool_demo"
TOOL_DESCRIPTION = "Demo generated tool."

def run(payload: str = "", **kwargs):
    return {"ok": True, "output": "demo:" + str(payload)}
'''


def test_builtin_tools_loaded():
    registry = ToolRegistry()
    loaded = registry.load_builtin_tools()
    assert loaded >= 3
    for name in ("echo", "word_count", "char_count"):
        assert registry.has(name)


def test_run_builtin_tool():
    registry = ToolRegistry()
    registry.load_builtin_tools()
    result = registry.run("echo", payload="hello")
    assert result["ok"] is True
    assert result["output"] == "hello"


def test_run_missing_tool_returns_error_dict():
    registry = ToolRegistry()
    result = registry.run("does_not_exist")
    assert result["ok"] is False
    assert "error" in result


def test_run_tool_raising_returns_error_dict(tmp_path: Path):
    registry = ToolRegistry()
    registry.load_builtin_tools()
    # None payload handled via kwargs contract; a raising tool is simulated
    # through a directly registered bad function.
    from lifelong_agent.tools.registry import ToolSpec

    def bad_run(payload: str = "", **kwargs):
        raise ValueError("boom")

    registry.register(
        ToolSpec(name="bad", description="", source="test", path=None, fn=bad_run)
    )
    result = registry.run("bad")
    assert result["ok"] is False
    assert "boom" in result["error"]


def test_load_generated_tool(tmp_path: Path):
    registry = ToolRegistry()
    tools_dir = tmp_path / "tools_generated"
    tools_dir.mkdir()
    (tools_dir / "generated_tool_demo.py").write_text(GENERATED)
    loaded = registry.load_generated_tools(tools_dir)
    assert loaded == 1
    assert registry.has("generated_tool_demo")
    assert registry.run("generated_tool_demo", payload="x")["ok"] is True


def test_load_generated_tool_missing_dir(tmp_path: Path):
    registry = ToolRegistry()
    assert registry.load_generated_tools(tmp_path / "nope") == 0


def test_load_generated_tool_ignores_tests(tmp_path: Path):
    registry = ToolRegistry()
    tools_dir = tmp_path / "tools_generated"
    tools_dir.mkdir()
    (tools_dir / "test_skip_me.py").write_text(GENERATED)
    (tools_dir / "_private.py").write_text(GENERATED)
    assert registry.load_generated_tools(tools_dir) == 0

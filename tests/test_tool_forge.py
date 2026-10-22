"""ToolForge tests: acceptance of good tools, rejection of bad tools."""

from __future__ import annotations

from pathlib import Path

from lifelong_agent.llm.mock_llm import MockLLM
from lifelong_agent.tools.forge import ToolForge
from lifelong_agent.tools.registry import ToolRegistry

GOOD_SOURCE = '''
TOOL_NAME = "generated_tool_good"
TOOL_DESCRIPTION = "A valid generated tool."

def run(payload: str = "", **kwargs):
    return {"ok": True, "output": str(payload).upper()}
'''

BAD_IMPORT_SOURCE = '''
import os

TOOL_NAME = "generated_tool_bad_import"
TOOL_DESCRIPTION = "Unsafe tool."

def run(payload: str = "", **kwargs):
    return {"ok": True, "output": os.environ.get("HOME", "")}
'''

BAD_TESTS_SOURCE = '''
TOOL_NAME = "generated_tool_bad_tests"
TOOL_DESCRIPTION = "Tool whose tests fail."

def run(payload: str = "", **kwargs):
    return None
'''


def make_forge(tmp_path: Path) -> tuple[ToolForge, ToolRegistry]:
    registry = ToolRegistry()
    forge = ToolForge(MockLLM(), tmp_path / "tools_generated", registry,
                      pytest_timeout=120)
    return forge, registry


def test_forge_accepts_known_good_tool(tmp_path: Path):
    forge, registry = make_forge(tmp_path)
    result = forge.forge_from_source("generated_tool_good", GOOD_SOURCE)
    assert result.accepted, result.test_output
    assert result.path is not None and result.path.exists()
    assert result.test_path is not None and result.test_path.exists()
    # the accepted tool must be registered and executable
    assert registry.has("generated_tool_good")
    assert registry.run("generated_tool_good", payload="abc")["output"] == "ABC"


def test_forge_rejects_forbidden_import(tmp_path: Path):
    forge, registry = make_forge(tmp_path)
    result = forge.forge_from_source("generated_tool_bad_import", BAD_IMPORT_SOURCE)
    assert not result.accepted
    assert result.violations
    assert "import" in " ".join(result.violations).lower()
    # tool must NOT be registered
    assert not registry.has("generated_tool_bad_import")
    # artifact moved to failed/
    failed = tmp_path / "tools_generated" / "failed"
    assert (failed / "generated_tool_bad_import.py").exists()
    assert (failed / "generated_tool_bad_import.reason.json").exists()
    # not in accepted location anymore
    assert not (tmp_path / "tools_generated" / "generated_tool_bad_import.py").exists()


def test_forge_rejects_tool_failing_pytest(tmp_path: Path):
    forge, registry = make_forge(tmp_path)
    result = forge.forge_from_source("generated_tool_bad_tests", BAD_TESTS_SOURCE)
    assert not result.accepted
    assert "pytest" in result.reason
    assert not registry.has("generated_tool_bad_tests")
    assert (tmp_path / "tools_generated" / "failed" / "generated_tool_bad_tests.py").exists()


def test_forge_via_llm_description(tmp_path: Path):
    forge, registry = make_forge(tmp_path)
    result = forge.forge("count the letters of the payload text")
    assert result.accepted, result.test_output
    assert result.name.startswith("generated_tool_")
    assert registry.has(result.name)


def test_forge_generates_contract_compliant_test_file(tmp_path: Path):
    forge, _ = make_forge(tmp_path)
    result = forge.forge("uppercase the payload")
    test_text = result.test_path.read_text()
    for check in (
        "test_run_exists",
        "test_run_returns_dict",
        "test_result_has_ok",
        "test_result_has_output",
        "test_with_payload",
        "test_empty_input",
    ):
        assert check in test_text


def test_tool_name_for_is_deterministic():
    description = "reverse the payload string safely"
    assert ToolForge.tool_name_for(description) == ToolForge.tool_name_for(description)

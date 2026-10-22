"""ToolValidator tests: AST safety checks and contract enforcement."""

from __future__ import annotations

from lifelong_agent.tools.validator import validate_source

GOOD = '''
TOOL_NAME = "generated_tool_ok"
TOOL_DESCRIPTION = "A safe tool."

def run(payload: str = "", **kwargs):
    return {"ok": True, "output": str(payload)}
'''


def test_good_tool_passes():
    result = validate_source(GOOD)
    assert result.ok, result.violations
    assert result.contract_ok


def test_syntax_error_fails():
    result = validate_source("def broken(:\n")
    assert not result.ok
    assert any("syntax" in v for v in result.violations)


def test_missing_contract_fails():
    result = validate_source("x = 1\n")
    assert not result.ok
    assert not result.contract_ok


def test_forbidden_imports_detected():
    for module in ["os", "sys", "subprocess", "socket", "urllib", "importlib"]:
        source = f"import {module}\n" + GOOD
        result = validate_source(source)
        assert not result.ok, module


def test_forbidden_from_import_detected():
    result = validate_source("from os import path\n" + GOOD)
    assert not result.ok


def test_relative_import_detected():
    result = validate_source("from . import sibling\n" + GOOD)
    assert not result.ok


def test_forbidden_calls_detected():
    for call in ["eval", "exec", "compile", "open", "input", "globals", "getattr"]:
        source = GOOD + f"\ntrigger = {call}('x')\n"
        result = validate_source(source)
        assert not result.ok, call


def test_forbidden_attributes_detected():
    source = GOOD + "\nleak = __import__('os').system\n"
    result = validate_source(source)
    assert not result.ok
    assert any("attribute" in v or "call" in v for v in result.violations)


def test_dunder_import_detected():
    result = validate_source(GOOD + "\nmod = __import__('os')\n")
    assert not result.ok

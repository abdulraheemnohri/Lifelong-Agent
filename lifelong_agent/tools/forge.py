"""ToolForge: generates, validates, tests, and registers Python tools.

Pipeline per tool:
1. Compose source (LLM proposal with deterministic template fallback).
2. Static AST safety validation.
3. Generate the pytest file and run it in a subprocess.
4. On success: register in the ToolRegistry.
5. On failure: move tool + test to ``tools_generated/failed/`` with a reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..llm.base import BaseLLM
from ..utils.fs import short_hash, slugify
from ..utils.logging import get_logger
from .pytest_runner import TestResult, run_pytest
from .registry import ToolRegistry
from .validator import ValidationResult, validate_source

logger = get_logger("forge")

TOOL_PROMPT_TEMPLATE = """Write a single self-contained Python tool module.
Contract (mandatory):
- TOOL_NAME = a snake_case string constant
- TOOL_DESCRIPTION = a one-line string constant
- def run(payload: str = "", **kwargs): returns a dict with "ok" and "output"

Hard constraints:
- Only pure Python. No imports of any kind.
- No file, network, or subprocess access. No eval/exec.
Task: {description}
Reply with the complete Python module only."""

_TEMPLATE = '''TOOL_NAME = "{name}"
TOOL_DESCRIPTION = "{description}"

def run(payload: str = "", **kwargs):
    """Generated tool: {description}"""
    text = str(payload)
    return {{
        "ok": True,
        "output": "{name} processed: " + text,
        "tool": TOOL_NAME,
    }}
'''


def _test_source(name: str) -> str:
    """The pytest file generated for every tool (spec-required checks)."""
    return f'''"""Auto-generated pytest suite for the {name} tool."""
import importlib.util
from pathlib import Path

TOOL_PATH = Path(__file__).resolve().parent / "{name}.py"


def _load():
    spec = importlib.util.spec_from_file_location("{name}_under_test", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_exists():
    module = _load()
    assert callable(getattr(module, "run", None))


def test_run_returns_dict():
    module = _load()
    result = module.run(payload="hello")
    assert isinstance(result, dict)


def test_result_has_ok():
    module = _load()
    result = module.run(payload="hello")
    assert "ok" in result


def test_result_has_output():
    module = _load()
    result = module.run(payload="hello")
    assert "output" in result


def test_with_payload():
    module = _load()
    result = module.run(payload="hello")
    assert result.get("ok") is True


def test_empty_input():
    module = _load()
    result = module.run()
    assert isinstance(result, dict)
    assert "ok" in result
'''


@dataclass
class ForgeResult:
    """Outcome of a tool forge cycle."""

    accepted: bool
    name: str
    path: Path | None = None
    test_path: Path | None = None
    violations: list[str] = field(default_factory=list)
    test_output: str = ""
    reason: str = ""


class ToolForge:
    """Generates and safely onboards new Python tools."""

    def __init__(
        self,
        llm: BaseLLM,
        tools_dir: Path,
        registry: ToolRegistry,
        pytest_timeout: int = 120,
    ) -> None:
        self.llm = llm
        self.tools_dir = Path(tools_dir)
        self.registry = registry
        self.pytest_timeout = pytest_timeout

    # ------------------------------------------------------------------
    # Source composition
    # ------------------------------------------------------------------
    def generate_source(self, name: str, description: str) -> str:
        """Ask the LLM for tool code; fall back to a safe deterministic template."""
        try:
            proposal = self.llm.generate(TOOL_PROMPT_TEMPLATE.format(description=description))
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM tool generation failed (%s); using template", exc)
            proposal = ""

        if proposal:
            proposal = self._rename_tool(proposal, name)
            candidate = validate_source(proposal)
            if candidate.ok:
                return proposal
            logger.info("LLM proposal rejected: %s", candidate.summary())

        return self._template(name, description)

    def _template(self, name: str, description: str) -> str:
        safe_description = description.replace('"', "'")[:200]
        return _TEMPLATE.format(name=name, description=safe_description)

    @staticmethod
    def _rename_tool(source: str, name: str) -> str:
        """Force the TOOL_NAME constant to the forge-chosen name."""
        return re.sub(r"^TOOL_NAME\s*=\s*.*$", f'TOOL_NAME = "{name}"', source,
                      count=1, flags=re.MULTILINE)

    @staticmethod
    def tool_name_for(description: str) -> str:
        """Deterministic unique-ish tool name from a description."""
        return f"generated_tool_{slugify(description, 3)}_{short_hash(description)}"

    # ------------------------------------------------------------------
    # Forge pipeline
    # ------------------------------------------------------------------
    def forge(self, description: str) -> ForgeResult:
        """Full pipeline: compose -> validate -> test -> register/reject."""
        name = self.tool_name_for(description)
        source = self.generate_source(name, description)
        return self.forge_from_source(name, source, description=description)

    def forge_from_source(
        self,
        name: str,
        source: str,
        description: str = "",
        register: bool = True,
    ) -> ForgeResult:
        """Validate, test, and onboard an explicit tool source."""
        tool_path = self.tools_dir / f"{name}.py"
        test_path = self.tools_dir / f"test_{name}.py"

        # 1) static validation
        validation: ValidationResult = validate_source(source)
        if not validation.ok:
            return self._reject(
                name, source, test_path,
                violations=validation.violations,
                reason="static validation failed",
            )

        # 2) write tool + generated tests
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        tool_path.write_text(source, encoding="utf-8")
        test_path.write_text(_test_source(name), encoding="utf-8")

        # 3) pytest
        result: TestResult = run_pytest(test_path, timeout=self.pytest_timeout)
        if not result.ok:
            return self._reject(
                name, source, test_path,
                test_output=result.output,
                reason=f"pytest failed (rc={result.returncode})",
            )

        # 4) register
        if register:
            spec = self.registry.load_module_tool(tool_path, source="generated")
            if spec is None:
                return self._reject(
                    name, source, test_path,
                    reason="tool could not be loaded after tests",
                )

        logger.info("Tool %s accepted and registered", name)
        return ForgeResult(
            accepted=True, name=name, path=tool_path, test_path=test_path,
            test_output=result.output,
        )

    def _reject(
        self,
        name: str,
        source: str,
        test_path: Path,
        violations: Optional[list[str]] = None,
        test_output: str = "",
        reason: str = "",
    ) -> ForgeResult:
        """Move a rejected tool to failed/ with a reason file."""
        failed_dir = self.tools_dir / "failed"
        failed_dir.mkdir(parents=True, exist_ok=True)
        (failed_dir / f"{name}.py").write_text(source, encoding="utf-8")
        if test_path.exists():
            (failed_dir / f"test_{name}.py").write_text(
                test_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
        from ..utils.json_utils import write_json

        write_json(
            failed_dir / f"{name}.reason.json",
            {"name": name, "reason": reason, "violations": violations or []},
        )
        # clean the accepted locations
        for path in (self.tools_dir / f"{name}.py", test_path):
            if path.exists():
                path.unlink()
        logger.warning("Tool %s rejected: %s", name, reason)
        return ForgeResult(
            accepted=False, name=name, path=failed_dir / f"{name}.py",
            violations=violations or [], test_output=test_output, reason=reason,
        )

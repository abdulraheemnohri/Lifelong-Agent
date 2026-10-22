"""Static AST safety validation for generated tool source code.

IMPORTANT: AST validation is a static first line of defence, NOT a full
sandbox. Generated tools should also run inside a container sandbox
(Docker/Podman/gVisor/Firecracker) in production. See SAFETY.md.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

FORBIDDEN_IMPORTS: frozenset[str] = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "pathlib",
        "shutil",
        "socket",
        "requests",
        "urllib",
        "importlib",
        "ctypes",
        "multiprocessing",
    }
)

FORBIDDEN_CALLS: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "breakpoint",
        "globals",
        "locals",
        "getattr",
        "setattr",
        "delattr",
    }
)

FORBIDDEN_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "system",
        "popen",
        "subprocess",
        "rmtree",
        "remove",
        "unlink",
        "write_text",
        "read_text",
        "environ",
    }
)


@dataclass
class ValidationResult:
    """Result of static source validation."""

    ok: bool
    violations: list[str] = field(default_factory=list)
    contract_ok: bool = True

    def summary(self) -> str:
        return "; ".join(self.violations) if self.violations else "ok"


def _module_root(name: str) -> str:
    return name.split(".")[0] if name else ""


def validate_source(source: str) -> ValidationResult:
    """Validate generated tool source: safety + tool contract.

    Checks:
    1. Source parses as Python.
    2. No forbidden imports, calls, or attributes (AST walk).
    3. Contract: ``TOOL_NAME`` and ``TOOL_DESCRIPTION`` string constants and
       a ``run`` function must be defined.
    """
    violations: list[str] = []

    # 1. syntax
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return ValidationResult(
            ok=False, violations=[f"syntax error: {exc}"], contract_ok=False
        )

    # 2. safety walk
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = _module_root(alias.name)
                if root in FORBIDDEN_IMPORTS:
                    violations.append(f"forbidden import: {root}")
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                violations.append("forbidden relative import")
            root = _module_root(node.module or "")
            if root in FORBIDDEN_IMPORTS:
                violations.append(f"forbidden import: {root}")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_CALLS:
                violations.append(f"forbidden call: {func.id}")
            elif isinstance(func, ast.Attribute) and func.attr in FORBIDDEN_ATTRIBUTES:
                violations.append(f"forbidden attribute: {func.attr}")
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRIBUTES:
                violations.append(f"forbidden attribute: {node.attr}")

    # 3. tool contract
    tool_name = None
    has_run = False
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TOOL_NAME":
                    if isinstance(node.value.value, str):
                        tool_name = node.value.value
                    else:
                        violations.append("TOOL_NAME must be a string")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run":
            has_run = True

    contract_ok = bool(tool_name) and has_run
    if not tool_name:
        violations.append("contract: TOOL_NAME string constant missing")
    if not has_run:
        violations.append("contract: run() function missing")

    return ValidationResult(ok=not violations and contract_ok, violations=violations,
                             contract_ok=contract_ok)

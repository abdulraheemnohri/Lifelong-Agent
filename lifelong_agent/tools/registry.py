"""Tool registry: loads builtin and generated tools, executes them safely."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from ..utils.logging import get_logger

logger = get_logger("registry")


@dataclass
class ToolSpec:
    """A registered tool."""

    name: str
    description: str
    source: str  # "builtin" | "generated"
    path: Path | None
    fn: Callable[..., Any]

    def run(self, payload: str = "", **kwargs: Any) -> dict[str, Any]:
        try:
            result = self.fn(payload=payload, **kwargs)
        except Exception as exc:  # noqa: BLE001 - tools must never crash the agent
            return {"ok": False, "error": f"tool {self.name!r} raised: {exc}"}
        return result


class ToolRegistry:
    """Registry of builtin and generated tools with the standard contract."""

    def __init__(self) -> None:
        self.tools: dict[str, ToolSpec] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(self, spec: ToolSpec) -> None:
        if spec.name in self.tools:
            logger.debug("Replacing already-registered tool %r", spec.name)
        self.tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self.tools.get(name)

    def has(self, name: str) -> bool:
        return name in self.tools

    def list_tools(self) -> list[ToolSpec]:
        return sorted(self.tools.values(), key=lambda spec: spec.name)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_module(self, path: Path) -> ModuleType | None:
        path = Path(path)
        module_name = f"lifelong_tool_{path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as exc:  # noqa: BLE001 - a bad tool must not break loading
            logger.warning("Failed to import tool module %s: %s", path, exc)
            return None

    def load_module_tool(self, path: Path, source: str) -> ToolSpec | None:
        """Load one tool module and register it if it honours the contract."""
        module = self._load_module(path)
        if module is None:
            return None
        name = getattr(module, "TOOL_NAME", None)
        description = getattr(module, "TOOL_DESCRIPTION", "")
        run_fn = getattr(module, "run", None)
        if not isinstance(name, str) or not name or not callable(run_fn):
            logger.warning("Module %s does not honour the tool contract", path)
            return None
        spec = ToolSpec(
            name=name,
            description=str(description),
            source=source,
            path=Path(path),
            fn=run_fn,
        )
        self.register(spec)
        return spec

    def load_builtin_tools(self) -> int:
        """Load all builtin tools shipped with the package."""
        import lifelong_agent.builtin_tools as builtin_pkg

        package_dir = Path(builtin_pkg.__file__).parent
        loaded = 0
        for path in sorted(package_dir.glob("*.py")):
            if path.name.startswith("_"):
                continue
            if self.load_module_tool(path, source="builtin") is not None:
                loaded += 1
        logger.info("Loaded %d builtin tools", loaded)
        return loaded

    def load_generated_tools(self, tools_dir: Path) -> int:
        """Load every accepted generated tool (failed/ and test_ excluded)."""
        tools_dir = Path(tools_dir)
        if not tools_dir.exists():
            return 0
        loaded = 0
        for path in sorted(tools_dir.glob("*.py")):
            if path.name.startswith("test_") or path.name.startswith("_"):
                continue
            if self.load_module_tool(path, source="generated") is not None:
                loaded += 1
        if loaded:
            logger.info("Loaded %d generated tools", loaded)
        return loaded

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run(self, name: str, payload: str = "", **kwargs: Any) -> dict[str, Any]:
        """Run a registered tool by name; errors come back as dicts."""
        spec = self.tools.get(name)
        if spec is None:
            return {"ok": False, "error": f"unknown tool: {name!r}"}
        try:
            result = spec.fn(payload=payload, **kwargs)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"tool {name!r} raised: {exc}"}
        if not isinstance(result, dict):
            return {"ok": False, "error": f"tool {name!r} did not return a dict"}
        return result

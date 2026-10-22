"""Builtin tool: echo."""

TOOL_NAME = "echo"
TOOL_DESCRIPTION = "Echo the payload back unchanged. Useful for connectivity checks."


def run(payload: str = "", **kwargs):
    return {"ok": True, "output": str(payload), "tool": TOOL_NAME}

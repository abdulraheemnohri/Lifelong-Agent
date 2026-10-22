"""Builtin tool: char_count."""

TOOL_NAME = "char_count"
TOOL_DESCRIPTION = "Count characters in the payload text, with and without spaces."


def run(payload: str = "", **kwargs):
    text = str(payload)
    without_spaces = text.replace(" ", "").replace("\t", "").replace("\n", "")
    return {
        "ok": True,
        "output": f"chars={len(text)} chars_no_spaces={len(without_spaces)}",
        "chars": len(text),
        "chars_no_spaces": len(without_spaces),
        "tool": TOOL_NAME,
    }

"""Builtin tool: word_count."""

TOOL_NAME = "word_count"
TOOL_DESCRIPTION = "Count words, characters, and lines in the payload text."


def run(payload: str = "", **kwargs):
    text = str(payload)
    words = text.split()
    return {
        "ok": True,
        "output": (
            f"words={len(words)} chars={len(text)} lines={len(text.splitlines())}"
        ),
        "words": len(words),
        "chars": len(text),
        "lines": len(text.splitlines()),
        "tool": TOOL_NAME,
    }

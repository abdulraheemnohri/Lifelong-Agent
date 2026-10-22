"""Optional AutoGen integration.

The adapter wraps the lifelong agent as a ConversableAgent-compatible
object. It works when ``pyautogen`` is installed and degrades gracefully
(with a plain ``generate_reply`` wrapper) when it is not.
"""

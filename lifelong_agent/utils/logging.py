"""Logging helpers built on the standard library."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once."""
    global _configured
    if _configured:
        logging.getLogger().setLevel(getattr(logging, level.upper(), logging.INFO))
        return
    logging.basicConfig(format=_FORMAT, level=getattr(logging, level.upper(), logging.INFO))
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Namespaced logger for agent subsystems."""
    return logging.getLogger(f"lifelong_agent.{name}")

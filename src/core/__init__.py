"""Lazy public exports keep leaf modules independent of agent initialization."""

from importlib import import_module

_EXPORTS = {
    "Agent": "src.core.agent",
    "LLM": "src.core.llm",
    "Message": "src.core.message",
    "PaperContext": "src.domain.paper_context",
    "AgentContextView": "src.domain.paper_context",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value

__all__ = ["Agent", "LLM", "Message", "PaperContext", "AgentContextView"]

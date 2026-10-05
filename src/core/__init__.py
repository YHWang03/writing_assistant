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
    '''按需导入公开对象并缓存到模块命名空间。

    paras:
        name: 需要访问的模块公开对象名称。
    return: 对应的公开对象；未知名称抛出 AttributeError。
    '''
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value

__all__ = ["Agent", "LLM", "Message", "PaperContext", "AgentContextView"]

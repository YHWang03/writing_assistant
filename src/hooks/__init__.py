"""hooks 包 — hook 基类与注册表。"""

from .base import Hook
from .registry import HookRegistry

__all__ = ["Hook", "HookRegistry"]

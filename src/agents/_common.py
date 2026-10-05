"""Shared declarative specifications for business agents."""

from ..hooks.builtin import (
    CollectWrittenPathsHook,
    DeadlineNudgeHook,
    MemoryExtractHook,
    MemoryRecallHook,
    OutputGateHook,
)


STANDARD_HOOKS = (
    MemoryRecallHook,
    DeadlineNudgeHook,
    CollectWrittenPathsHook,
    OutputGateHook,
    MemoryExtractHook,
)

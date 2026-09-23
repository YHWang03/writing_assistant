"""Shared declarative specifications for business agents."""

from ..hooks.builtin import (
    CollectWrittenPathsHook,
    DeadlineNudgeHook,
    FinishNudgeHook,
    MemoryExtractHook,
    MemoryRecallHook,
    OutputGateHook,
)


STANDARD_HOOKS = (
    MemoryRecallHook,
    DeadlineNudgeHook,
    CollectWrittenPathsHook,
    FinishNudgeHook,
    OutputGateHook,
    MemoryExtractHook,
)

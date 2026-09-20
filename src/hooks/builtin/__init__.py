"""内置 hooks — 论文写作助手所有核心 hook"""

from .memory_recall import MemoryRecallHook
from .collect_paths import CollectWrittenPathsHook
from .memory_extract import MemoryExtractHook
from .deadline import DeadlineNudgeHook
from .finish_nudge import FinishNudgeHook
from .output_gate import OutputGateHook

__all__ = [
    "MemoryRecallHook", "CollectWrittenPathsHook", "MemoryExtractHook",
    "DeadlineNudgeHook", "FinishNudgeHook", "OutputGateHook",
]

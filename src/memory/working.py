"""
WorkingMemory — 短期记忆（滑动窗口）

保留最近 N 轮对话，超出窗口自动丢弃旧记录。
用于 Agent 在当前任务中保持上下文连贯。
"""

from .base import BaseMemory, MemoryItem


class WorkingMemory(BaseMemory):
    """短期工作记忆，滑动窗口"""

    def __init__(self, max_size: int = 20):
        super().__init__(max_size=max_size)

    def add(self, item: MemoryItem):
        self._items.append(item)
        # 超出窗口丢弃最旧
        while len(self._items) > self.max_size:
            self._items.pop(0)

    def get_context(self, limit: int = 10) -> str:
        """返回最近 N 条记忆的文本摘要"""
        recent = self._items[-limit:] if len(self._items) > limit else self._items
        lines = []
        for item in recent:
            content_preview = item.content[:200].replace("\n", " ")
            lines.append(f"[{item.role}] {content_preview}")
        return "\n".join(lines)

    def get_recent(self, n: int = 5) -> list[MemoryItem]:
        """获取最近 N 条记忆"""
        return self._items[-n:] if len(self._items) > n else self._items.copy()
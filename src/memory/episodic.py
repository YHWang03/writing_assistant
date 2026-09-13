"""
EpisodicMemory — 情景记忆

按 session 组织，存储完整的交互事件。
高重要性条目可通过 consolidate 提升为长期记忆。
"""

from datetime import datetime, timedelta
from .base import BaseMemory, MemoryItem


class EpisodicMemory(BaseMemory):
    """情景记忆，按 session 组织交互事件"""

    def __init__(self, max_size: int = 200):
        super().__init__(max_size=max_size)
        # session_id -> [item_index, ...]
        self._sessions: dict[str, list[int]] = {}

    def add(self, item: MemoryItem,
            session_id: str = "default") -> str:
        """添加情景记忆，关联到指定 session"""
        self._items.append(item)
        idx = len(self._items) - 1
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(idx)

        # 超出容量时淘汰重要性最低的
        if len(self._items) > self.max_size:
            # 收集需要淘汰的索引
            sorted_idx = sorted(
                range(len(self._items)),
                key=lambda i: self._items[i].importance,
            )
            excess = len(self._items) - self.max_size
            to_remove = set(sorted_idx[:excess])

            # 重建 items 和 sessions
            new_items = []
            old_to_new: dict[int, int] = {}
            for i, item in enumerate(self._items):
                if i not in to_remove:
                    old_to_new[i] = len(new_items)
                    new_items.append(item)
            self._items = new_items

            # 重建 sessions 索引
            for sid, indices in self._sessions.items():
                self._sessions[sid] = [
                    old_to_new[i] for i in indices if i in old_to_new
                ]
                if not self._sessions[sid]:
                    del self._sessions[sid]

        return item.content[:50]

    def get_context(self, limit: int = 10) -> str:
        """返回最近 N 条情景记忆"""
        recent = self._items[-limit:] if len(self._items) > limit else self._items
        lines = []
        for item in recent:
            lines.append(
                f"[{item.timestamp.strftime('%m-%d %H:%M')}] "
                f"({item.role}) {item.content[:200]}"
            )
        return "\n".join(lines)

    def search(self, keyword: str, limit: int = 5) -> list[MemoryItem]:
        """关键词搜索（情景记忆量小，关键词匹配即可）"""
        results = []
        for item in reversed(self._items):
            if keyword.lower() in item.content.lower():
                results.append(item)
                if len(results) >= limit:
                    break
        return results

    def get_session(self, session_id: str) -> list[MemoryItem]:
        """获取指定 session 的所有记忆"""
        indices = self._sessions.get(session_id, [])
        return [self._items[i] for i in indices if i < len(self._items)]

    def get_all(self) -> list[MemoryItem]:
        """获取所有情景记忆"""
        return self._items.copy()

    def forget(self, strategy: str = "importance_based",
               threshold: float = 0.1, max_age_days: int = 30) -> int:
        """按策略遗忘情景记忆"""
        now = datetime.now()
        to_remove: set[int] = set()

        for i, item in enumerate(self._items):
            if strategy == "importance_based":
                if item.importance < threshold:
                    to_remove.add(i)
            elif strategy == "time_based":
                if now - item.timestamp > timedelta(days=max_age_days):
                    to_remove.add(i)
            elif strategy == "capacity_based":
                if len(self._items) > self.max_size:
                    sorted_idx = sorted(
                        range(len(self._items)),
                        key=lambda j: self._items[j].importance,
                    )
                    excess = len(self._items) - self.max_size
                    to_remove = set(sorted_idx[:excess])
                    break

        if not to_remove:
            return 0

        # 重建 items 和 sessions
        new_items = []
        old_to_new: dict[int, int] = {}
        for i, item in enumerate(self._items):
            if i not in to_remove:
                old_to_new[i] = len(new_items)
                new_items.append(item)
        self._items = new_items

        for sid, indices in list(self._sessions.items()):
            self._sessions[sid] = [
                old_to_new[i] for i in indices if i in old_to_new
            ]
            if not self._sessions[sid]:
                del self._sessions[sid]

        return len(to_remove)
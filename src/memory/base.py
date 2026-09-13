"""
BaseMemory — 记忆基类

每个 Agent 通过 MemoryManager 统一管理情景记忆（EpisodicMemory）和长期记忆（LongTermMemory）。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MemoryItem:
    """单条记忆项"""
    content: str
    role: str           # "user" | "assistant" | "tool" | "system"
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 0.5   # 0.0 ~ 1.0，默认中等
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "role": self.role,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "metadata": self.metadata,
        }


class BaseMemory(ABC):
    """记忆抽象基类"""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._items: list[MemoryItem] = []

    @abstractmethod
    def add(self, item: MemoryItem):
        """添加记忆项"""
        ...

    @abstractmethod
    def get_context(self, limit: int = 10) -> str:
        """获取记忆上下文（转为 LLM 可读的文本）"""
        ...

    def clear(self):
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(items={len(self)})"

    @staticmethod
    def _calculate_importance(content: str, base: float = 0.5) -> float:
        """根据内容计算重要性分数"""
        score = base
        if len(content) > 100:
            score += 0.1
        important_keywords = ["重要", "关键", "必须", "注意", "警告", "错误",
                              "important", "critical", "key", "注意", "core"]
        if any(kw in content.lower() for kw in important_keywords):
            score += 0.2
        return max(0.0, min(1.0, score))
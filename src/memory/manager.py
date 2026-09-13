"""
MemoryManager — 统一记忆管理器

管理两种记忆类型：
- EpisodicMemory: 情景记忆，按 session 组织的完整交互，可搜索
- LongTermMemory: 长期记忆，持久化，TF-IDF 检索

注意：短期记忆（WorkingMemory）已被 self._history 替代，
_history 存储完整消息，比 WorkingMemory 的截断文本更完整。
"""

import uuid
from datetime import datetime
from .base import BaseMemory, MemoryItem
from .episodic import EpisodicMemory
from .long_term import LongTermMemory


class MemoryManager:
    """统一记忆管理器"""

    def __init__(self,
                 long_term_max_size: int = 500,
                 episodic_max_size: int = 200,
                 long_term_path: str | None = None):
        self.episodic = EpisodicMemory(max_size=episodic_max_size)
        self.long_term = LongTermMemory(
            max_size=long_term_max_size,
            file_path=long_term_path,
        )
        self._current_session_id: str | None = None
        if long_term_path:
            self.long_term.load()

    # ---- 框架自动记录 ----

    def record_interaction(self, agent_name: str,
                           user_input: str, result: str):
        """每轮对话结束后自动记录到情景记忆和长期记忆"""
        # 确保有 session_id
        if self._current_session_id is None:
            self._current_session_id = self._generate_session_id(agent_name)

        # 情景记忆 — 完整交互（带 session_id）
        self.episodic.add(MemoryItem(
            content=f"User: {user_input[:300]}", role="user",
            importance=BaseMemory._calculate_importance(user_input),
        ), session_id=self._current_session_id)
        self.episodic.add(MemoryItem(
            content=f"Assistant: {result[:500]}", role="assistant",
            importance=BaseMemory._calculate_importance(result, base=0.6),
        ), session_id=self._current_session_id)

        # 长期记忆 — 摘要形式
        summary = f"[{agent_name}] Task: {user_input[:200]} | Result: {result[:200]}"
        importance = BaseMemory._calculate_importance(summary, base=0.4)
        self.long_term.add(MemoryItem(
            content=summary, role="system", importance=importance,
        ))

        # 自动 consolidate：高重要性情景记忆提升为长期记忆
        self._auto_consolidate()

    # ---- 委托给 LongTermMemory ----

    def search(self, query: str, limit: int = 5) -> list[MemoryItem]:
        """搜索长期记忆"""
        return self.long_term.search(query, limit)

    def add_memory(self, content: str, importance: float | None = None):
        """Agent 主动添加长期记忆"""
        if importance is None:
            importance = BaseMemory._calculate_importance(content)
        self.long_term.add(MemoryItem(
            content=content, role="system", importance=importance,
        ))

    # ---- 委托给 EpisodicMemory ----

    def search_episodic(self, keyword: str, limit: int = 5) -> list[MemoryItem]:
        """搜索情景记忆（最近交互记录）"""
        return self.episodic.search(keyword, limit)

    def get_episodic_context(self, limit: int = 10) -> str:
        """获取最近的情景记忆上下文"""
        return self.episodic.get_context(limit)

    # ---- 记忆整合 ----

    def consolidate(self, importance_threshold: float = 0.6) -> int:
        """将高重要性情景记忆提升为长期记忆"""
        all_items = self.episodic.get_all()
        promoted = self.long_term.consolidate(all_items, importance_threshold)
        return promoted

    def _auto_consolidate(self):
        """自动整合：每次记录交互后，检查是否有高重要性情景记忆"""
        # 仅处理最近 10 条，避免每次全量扫描
        recent = self.episodic.get_all()[-10:]
        self.long_term.consolidate(
            [item for item in recent if item.importance >= 0.7],
            importance_threshold=0.7,
        )

    # ---- 持久化 ----

    def save(self):
        self.long_term.save()

    def load(self):
        self.long_term.load()

    # ---- 遗忘 ----

    def forget(self, strategy: str = "importance_based",
               threshold: float = 0.1, max_age_days: int = 30) -> int:
        count = self.long_term.forget(strategy, threshold, max_age_days)
        count += self.episodic.forget(strategy, threshold, max_age_days)
        return count

    # ---- 内部 ----

    @staticmethod
    def _generate_session_id(agent_name: str) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{agent_name}_{ts}_{uuid.uuid4().hex[:6]}"
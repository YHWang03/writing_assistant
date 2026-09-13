"""
LongTermMemory — 长期记忆（持久化）

存储关键决策、经验教训、用户偏好，支持文件持久化。
- TF-IDF 向量相似度检索（关键词 fallback）
- 重要性评分 + 按重要性淘汰
- forget 策略（importance_based / time_based / capacity_based）
- consolidate 机制（从 EpisodicMemory 提升重要记忆）
"""

import json
import math
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from .base import BaseMemory, MemoryItem


def _tokenize(text: str) -> list[str]:
    """简单分词：提取长度 > 1 的字母/数字/中文序列"""
    # 英文单词 + 中文字符
    tokens = re.findall(r'[a-zA-Z]+|[\u4e00-\u9fff]+|\d+', text.lower())
    return [t for t in tokens if len(t) > 1]


class LongTermMemory(BaseMemory):
    """长期记忆，支持 TF-IDF 检索 + 重要性遗忘 + 文件持久化"""

    def __init__(self, max_size: int = 500, file_path: str | None = None):
        super().__init__(max_size=max_size)
        self.file_path = file_path
        # TF-IDF 缓存
        self._idf: dict[str, float] = {}
        self._dirty: bool = True  # 标记是否需要重建 IDF

    # ---- 增删 ----

    def add(self, item: MemoryItem):
        """添加记忆项；超出容量时淘汰重要性最低的（保持时间顺序）"""
        self._items.append(item)
        self._dirty = True
        if len(self._items) > self.max_size:
            # 找到重要性最低的 excess 条，移除它们（不改变原顺序）
            excess = len(self._items) - self.max_size
            sorted_indices = sorted(
                range(len(self._items)),
                key=lambda i: self._items[i].importance,
            )
            to_remove = set(sorted_indices[:excess])
            self._items = [
                item for i, item in enumerate(self._items) if i not in to_remove
            ]

    # ---- 检索 ----

    def search(self, keyword: str, limit: int = 5) -> list[MemoryItem]:
        """TF-IDF 向量相似度检索，关键词匹配作为 fallback"""
        if not self._items:
            return []

        self._ensure_idf()

        query_vec = self._tfidf_vector(keyword, self._idf)
        scored: list[tuple[float, MemoryItem]] = []

        for item in self._items:
            doc_vec = self._tfidf_vector(item.content, self._idf)
            tfidf_score = self._cosine(query_vec, doc_vec)

            # 回退关键词匹配
            kw_score = 1.0 if keyword.lower() in item.content.lower() else 0.0

            # 综合得分：向量 0.7 + 关键词 0.3
            base = tfidf_score * 0.7 + kw_score * 0.3

            # 重要性加权 [0.8, 1.2]
            importance_weight = 0.8 + item.importance * 0.4
            combined = base * importance_weight

            if combined > 0:
                scored.append((combined, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def get_context(self, limit: int = 10) -> str:
        """返回最近 N 条长期记忆"""
        recent = self._items[-limit:] if len(self._items) > limit else self._items
        lines = []
        for item in recent:
            lines.append(
                f"[{item.timestamp.strftime('%m-%d %H:%M')}] "
                f"(重要性:{item.importance:.2f}) {item.content[:200]}"
            )
        return "\n".join(lines)

    # ---- 遗忘策略 ----

    def forget(self, strategy: str = "importance_based",
               threshold: float = 0.1, max_age_days: int = 30) -> int:
        """按策略遗忘记忆，返回被遗忘数量"""
        now = datetime.now()
        to_remove: list[int] = []

        for i, item in enumerate(self._items):
            if strategy == "importance_based":
                if item.importance < threshold:
                    to_remove.append(i)
            elif strategy == "time_based":
                if now - item.timestamp > timedelta(days=max_age_days):
                    to_remove.append(i)
            elif strategy == "capacity_based":
                if len(self._items) > self.max_size:
                    sorted_idx = sorted(
                        range(len(self._items)),
                        key=lambda j: self._items[j].importance,
                    )
                    excess = len(self._items) - self.max_size
                    to_remove = sorted_idx[:excess]
                    break  # capacity_based 一次性算完

        # 按索引降序排列，确保 pop 时不会偏移
        to_remove = sorted(to_remove, reverse=True)
        for i in to_remove:
            self._items.pop(i)

        if to_remove:
            self._dirty = True
        return len(to_remove)

    # ---- 记忆整合 ----

    def consolidate(self, episodic_items: list[MemoryItem],
                    importance_threshold: float = 0.6) -> int:
        """将情景记忆中重要性 >= threshold 的条目提升为长期记忆"""
        count = 0
        for item in episodic_items:
            if item.importance >= importance_threshold:
                # 创建副本，避免修改原始 MemoryItem 导致重复整合时重要性持续累加
                new_item = MemoryItem(
                    content=item.content,
                    role=item.role,
                    importance=min(item.importance * 1.1, 1.0),
                    metadata=item.metadata.copy(),
                )
                self.add(new_item)
                count += 1
        return count

    # ---- 持久化 ----

    def save(self, file_path: str | None = None):
        """保存到 JSON 文件"""
        path = Path(file_path or self.file_path)
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [item.to_dict() for item in self._items]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, file_path: str | None = None):
        """从 JSON 文件加载"""
        path = Path(file_path or self.file_path)
        if not path or not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self._items = []
        for d in data:
            self._items.append(MemoryItem(
                content=d.get("content", ""),
                role=d.get("role", "system"),
                importance=d.get("importance", 0.5),
                metadata=d.get("metadata", {}),
            ))
        self._dirty = True

    # ---- TF-IDF 内部实现 ----

    def _ensure_idf(self):
        """必要时重建 IDF 词典"""
        if not self._dirty:
            return
        self._idf = self._build_idf([item.content for item in self._items])
        self._dirty = False

    @staticmethod
    def _build_idf(documents: list[str]) -> dict[str, float]:
        """构建 IDF 词典"""
        N = len(documents)
        if N == 0:
            return {}
        df: dict[str, int] = {}
        for doc in documents:
            for word in set(_tokenize(doc)):
                df[word] = df.get(word, 0) + 1
        return {w: math.log((N + 1) / (c + 1)) + 1 for w, c in df.items()}

    @staticmethod
    def _tfidf_vector(text: str, idf: dict[str, float]) -> dict[str, float]:
        """计算文本的 TF-IDF 向量"""
        tokens = _tokenize(text)
        if not tokens:
            return {}
        tf = Counter(tokens)
        total = len(tokens)
        return {w: (tf[w] / total) * idf.get(w, 0) for w in tf if w in idf}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        """余弦相似度"""
        if not a or not b:
            return 0.0
        dot = sum(a.get(k, 0) * b.get(k, 0) for k in set(a) | set(b))
        norm_a = math.sqrt(sum(v ** 2 for v in a.values()))
        norm_b = math.sqrt(sum(v ** 2 for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
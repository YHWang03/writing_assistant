"""记忆工具 — 让 Agent 主动查询和存储记忆
- MemoryTool: 支持 search/add/episodic/consolidate 操作
"""

from ..base import Tool


class MemoryTool(Tool):
    """记忆工具，Agent 可在 ReAct 循环中主动调用，
    查询长期记忆、情景记忆或整合记忆。"""

    def __init__(self):
        super().__init__(
            name="memory",
            description="查询或存储记忆。支持三种操作："
                        "search=搜索长期记忆, episodic=搜索最近交互记录, "
                        "add=存储新记忆, consolidate=将高重要性情景记忆提升为长期记忆"
        )
        self._manager = None  # MemoryManager 实例，由外部注入

    def set_memory_manager(self, manager):
        """注入 MemoryManager 实例"""
        self._manager = manager

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["search", "add", "episodic", "consolidate"],
                    "description": "search=搜索长期记忆, episodic=搜索最近交互记录, "
                                   "add=存储新记忆, consolidate=整合高重要性情景记忆",
                },
                "query": {
                    "type": "string",
                    "description": "搜索关键词（action=search/episodic 时必填）",
                },
                "content": {
                    "type": "string",
                    "description": "要存储的记忆内容（action=add 时必填）",
                },
                "importance": {
                    "type": "number",
                    "description": "重要性评分 0.0~1.0（action=add 时可选，默认自动计算）",
                },
            },
            "required": ["action"],
        }

    def execute(self, action: str, query: str = "",
                content: str = "", importance: float = 0.0) -> str:
        if self._manager is None:
            return "Error: 记忆系统未注入，无法使用记忆工具"

        if action == "search":
            return self._do_search(query)

        elif action == "episodic":
            return self._do_episodic(query)

        elif action == "add":
            return self._do_add(content, importance)

        elif action == "consolidate":
            return self._do_consolidate()

        return f"Error: 不支持的操作 '{action}'（支持: search, episodic, add, consolidate）"

    def _do_search(self, query: str) -> str:
        if not query.strip():
            return "请提供搜索关键词"
        results = self._manager.search(query, limit=5)
        if not results:
            return f"未找到与 '{query}' 相关的长期记忆"
        return self._format_results("长期记忆", results)

    def _do_episodic(self, query: str) -> str:
        if not query.strip():
            return "请提供搜索关键词"
        results = self._manager.search_episodic(query, limit=5)
        if not results:
            return f"未找到与 '{query}' 相关的交互记录"
        return self._format_results("情景记忆（最近交互）", results)

    def _do_add(self, content: str, importance: float) -> str:
        if not content.strip():
            return "请提供要存储的记忆内容"
        if importance <= 0:
            self._manager.add_memory(content)
        else:
            self._manager.add_memory(content, importance=importance)
        return "记忆已存储"

    def _do_consolidate(self) -> str:
        count = self._manager.consolidate()
        return f"已整合 {count} 条高重要性情景记忆到长期记忆"

    @staticmethod
    def _format_results(label: str, items) -> str:
        lines = [f"=== {label} ==="]
        for item in items:
            lines.append(
                f"- [{item.role}] (重要性:{item.importance:.2f}) "
                f"{item.content[:200]}"
            )
        return "\n".join(lines)
"""上下文工具 — Agent 主动读取 PaperContext 数据。"""

import json as json_mod

from ..base import Tool


class ReadContextTool(Tool):
    """读取当前 Agent 有权访问的 PaperContext 字段"""

    def __init__(self):
        super().__init__(
            name="read_context",
            description="读取当前 Agent 可访问的上下文信息（论文路径、创新点、"
                        "实验描述、文献库、当前状态等）。可指定 field 参数读取单个字段的完整内容。"
        )
        self._context_view = None

    def set_context(self, context_view):
        """注入 AgentContextView。

        paras:
            context_view: AgentContextView 实例
        """
        self._context_view = context_view

    def get_parameters(self) -> dict:
        """返回工具参数的 JSON Schema 定义。

        return: input_schema 字典
        """
        return {
            "type": "object",
            "properties": {
                "field": {
                    "type": "string",
                    "description": (
                        "可选，指定要读取的字段名。不指定则返回所有可读字段的摘要。"
                        "合法字段取决于当前 Agent 的权限，常见的有："
                        "reference_library, innovation_points, experiment_description,"
                        "formula_manuscript, sections, main_tex_path, output_dir 等。"
                    ),
                },
            },
            "required": [],
        }

    def execute(self, field: str = "") -> str:
        """读取上下文字段。

        paras:
            field: 要读取的字段名；为空返回所有可读字段的摘要
        return: 字段内容文本；未注入/无权限/异常返回错误字符串
        """
        if self._context_view is None:
            return "Error: 上下文未注入"
        try:
            if not field:
                return self._context_view.get_readable_summary()

            readable = getattr(self._context_view, "_readable", set())
            if field not in readable:
                valid = sorted(readable)
                lines = [
                    f"字段 '{field}' 不可读（当前 Agent 无权读取）。",
                    f"可读字段: {valid}",
                ]
                return "\n".join(lines)

            value = getattr(self._context_view, field)
            return self._format_field_value(field, value)
        except Exception as e:
            return f"Error: 读取上下文失败 — {e}"

    def _format_field_value(self, field: str, value) -> str:
        """格式化单个字段的完整内容。

        paras:
            field: 字段名
            value: 字段值（任意类型）
        return: 可读的文本表示
        """
        if value is None:
            return f"{field}: (无数据)"

        if isinstance(value, str):
            if not value:
                return f"{field}: (空字符串)"
            return f"{field}:\n{value}"

        if isinstance(value, bool):
            return f"{field}: {value}"

        if isinstance(value, dict):
            if not value:
                return f"{field}: (空字典)"
            return f"{field}:\n{json_mod.dumps(value, ensure_ascii=False, indent=2)}"

        if isinstance(value, list):
            if not value:
                return f"{field}: (空列表)"

            if field == "reference_library":
                return self._format_reference_library(value)

            lines = [f"{field}: [{len(value)} 个元素]"]
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    lines.append(f"  [{i}] {json_mod.dumps(item, ensure_ascii=False)}")
                else:
                    lines.append(f"  [{i}] {item}")
            return "\n".join(lines)

        return f"{field}: {value}"

    def _format_reference_library(self, refs: list) -> str:
        """格式化 reference_library，完整展示每篇文献的所有字段。

        paras:
            refs: 文献对象列表（含 to_dict 方法）
        return: 文本表示
        """
        lines = [f"reference_library: [{len(refs)} 篇文献]"]
        for i, ref in enumerate(refs):
            lines.append(f"--- 文献 {i+1} ---")
            lines.append(json_mod.dumps(ref.to_dict(), ensure_ascii=False, indent=2))
        return "\n".join(lines)

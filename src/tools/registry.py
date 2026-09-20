"""工具注册表 — 每个 Agent 独立实例，实现不同 Agent 的工具权限隔离。"""

import logging
from .base import Tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """工具注册表（非全局单例，每个 Agent 独立实例）"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._functions: dict[str, dict] = {}

    def register(self, tool: Tool):
        """注册一个 Tool 实例。

        paras:
            tool: Tool 实例，重名时覆盖并告警
        """
        if tool.name in self._tools:
            logger.warning(f"Tool '{tool.name}' already registered, overwriting.")
        self._tools[tool.name] = tool

    def register_function(self, name: str, description: str, func,
                          properties: dict, required: list[str] | None = None):
        """注册一个普通函数作为工具（轻量方式，不需要写 Tool 子类）。

        paras:
            name: 工具名
            description: 工具描述
            func: 可调用对象
            properties: input_schema 的 properties 部分
            required: 必填参数列表
        """
        self._functions[name] = {
            "description": description,
            "func": func,
            "properties": properties,
            "required": required or [],
        }

    def execute(self, name: str, params: dict) -> str:
        """执行一个工具（先查 Tool 实例，再查注册函数）。

        paras:
            name: 工具名
            params: 工具参数字典
        return: 工具返回的字符串；未知工具或执行异常返回 "Error: ..." 字符串
        """
        if name in self._tools:
            try:
                return self._tools[name].execute(**params)
            except Exception as e:
                return f"Error: tool '{name}' failed — {e}"

        if name in self._functions:
            try:
                return self._functions[name]["func"](**params)
            except Exception as e:
                return f"Error: tool '{name}' failed — {e}"

        return f"Error: unknown tool '{name}'."

    def to_anthropic_format(self) -> list[dict]:
        """转为 Anthropic API 的 tools 参数格式。

        return: 所有 Tool 实例与注册函数的 tool 格式列表
        """
        result = []
        for tool in self._tools.values():
            result.append(tool.to_anthropic_format())
        for name, info in self._functions.items():
            result.append({
                "name": name,
                "description": info["description"],
                "input_schema": {
                    "type": "object",
                    "properties": info["properties"],
                    "required": info["required"],
                },
            })
        return result

    def list_names(self) -> list[str]:
        """返回所有已注册工具名。

        return: 工具名列表
        """
        return list(self._tools.keys()) + list(self._functions.keys())

    def get_tool(self, name: str) -> Tool | None:
        """获取指定名称的 Tool 实例。

        paras:
            name: 工具名
        return: Tool 实例；不存在返回 None
        """
        return self._tools.get(name)

    def subset(self, names: list[str]) -> "ToolRegistry":
        """创建子注册表，只包含指定名称的工具。

        paras:
            names: 工具名列表
        return: 新的 ToolRegistry 实例
        """
        sub = ToolRegistry()
        for name in names:
            if name in self._tools:
                sub.register(self._tools[name])
            if name in self._functions:
                info = self._functions[name]
                sub.register_function(
                    name, info["description"], info["func"],
                    info["properties"], info["required"]
                )
        return sub

    def __len__(self) -> int:
        return len(self._tools) + len(self._functions)

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={self.list_names()})"

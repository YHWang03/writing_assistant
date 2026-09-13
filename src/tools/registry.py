"""
工具注册表 — 管理每个 Agent 专用的工具集合

不同于 hello_agents 的全局单例，这里每个 Agent 创建自己的 ToolRegistry 实例，
实现不同 Agent 拥有不同工具（权限隔离）。
"""

import logging
from .base import Tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """工具注册表（非全局单例，每个 Agent 独立实例）"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._functions: dict[str, dict] = {}

    # ---- 注册 ----

    def register(self, tool: Tool):
        """注册一个 Tool 实例"""
        if tool.name in self._tools:
            logger.warning(f"Tool '{tool.name}' already registered, overwriting.")
        self._tools[tool.name] = tool

    def register_function(self, name: str, description: str, func,
                          properties: dict, required: list[str] | None = None):
        """
        注册一个普通函数作为工具（轻量方式，不需要写 Tool 子类）。

        参数：
          name:        工具名
          description: 工具描述
          func:        可调用对象
          properties:  input_schema 的 properties 部分
          required:    必填参数列表
        """
        self._functions[name] = {
            "description": description,
            "func": func,
            "properties": properties,
            "required": required or [],
        }

    # ---- 执行 ----

    def execute(self, name: str, params: dict) -> str:
        """执行一个工具并返回字符串结果"""
        # 优先查 Tool 实例
        if name in self._tools:
            try:
                return self._tools[name].execute(**params)
            except Exception as e:
                return f"Error: tool '{name}' failed — {e}"

        # 再查注册的函数
        if name in self._functions:
            try:
                return self._functions[name]["func"](**params)
            except Exception as e:
                return f"Error: tool '{name}' failed — {e}"

        return f"Error: unknown tool '{name}'."

    # ---- 输出 ----

    def to_anthropic_format(self) -> list[dict]:
        """转为 Anthropic API 的 tools 参数格式"""
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
        return list(self._tools.keys()) + list(self._functions.keys())

    def get_tool(self, name: str) -> Tool | None:
        """return Tool 实例（如果存在）"""
        return self._tools.get(name)

    def subset(self, names: list[str]) -> "ToolRegistry":
        """创建子注册表，只包含指定名称的工具"""
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
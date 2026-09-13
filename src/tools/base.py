"""
工具基类 — 所有工具的抽象父类

每个工具需要提供：
  - name: 工具名（模型通过这个名字调用）
  - description: 工具描述（模型据此判断何时调用）
  - parameters: 输入参数的 JSON Schema
  - execute(): 执行逻辑
"""

from abc import ABC, abstractmethod


class Tool(ABC):
    """工具抽象基类"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """执行工具，返回字符串结果"""
        ...

    @abstractmethod
    def get_parameters(self) -> dict:
        """
        返回 Anthropic input_schema 格式的参数定义：
          {
            "type": "object",
            "properties": {...},
            "required": [...]
          }
        """
        ...

    def to_anthropic_format(self) -> dict:
        """
        转为 Anthropic Messages API 的 tool 格式
        返回：
          dict: Anthropic tool 格式的字典
            {
              "name": str,
              "description": str,
              "input_schema": dict
            }
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.get_parameters(),
        }

    def __str__(self) -> str:
        return f"Tool(name={self.name})"

    def __repr__(self) -> str:
        return self.__str__()
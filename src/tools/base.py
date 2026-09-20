"""工具基类 — 所有工具的抽象父类。"""

from abc import ABC, abstractmethod


class Tool(ABC):
    """工具抽象基类"""

    def __init__(self, name: str, description: str):
        """初始化工具。

        paras:
            name: 工具名（模型通过这个名字调用）
            description: 工具描述（模型据此判断何时调用）
        """
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """执行工具。

        return: 字符串结果
        """
        ...

    @abstractmethod
    def get_parameters(self) -> dict:
        """返回工具的参数定义。

        return: Anthropic input_schema 格式的字典（type/properties/required）
        """
        ...

    def to_anthropic_format(self) -> dict:
        """转为 Anthropic Messages API 的 tool 格式。

        return: {"name", "description", "input_schema"} 字典
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

"""消息系统 — 对话中的单条消息"""

from typing import Literal

MessageRole = Literal["user", "assistant", "system", "tool"]


class Message:
    """一条对话消息（Agent 内部 _history 使用）"""

    def __init__(self, content: str, role: MessageRole):
        """构造消息。

        paras:
            content: 消息文本
            role: 消息角色
        return: 无
        """
        self.content = content
        self.role = role

    def to_dict(self) -> dict:
        """转为 API 格式。

        paras: 无
        return: {"role": ..., "content": ...} dict
        """
        return {"role": self.role, "content": self.content}

    def __str__(self) -> str:
        preview = self.content[:80].replace("\n", " ")
        return f"[{self.role}] {preview}"

    def __repr__(self) -> str:
        return self.__str__()

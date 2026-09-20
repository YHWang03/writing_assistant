"""结束工具 — Agent 调用后主动退出 run_loop。"""

from ..base import Tool


class FinishTool(Tool):
    """Agent 调用 finish 表示任务完成，loop 检测到此调用后直接退出"""

    def __init__(self):
        super().__init__(
            name="finish",
            description="任务完成时调用此工具，传入总结报告。调用后对话结束。"
                        "**重要：确认所有任务已完成后再调用此工具。**"
        )

    def get_parameters(self) -> dict:
        """返回工具参数的 JSON Schema 定义。

        return: input_schema 字典
        """
        return {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "任务完成总结，包括已完成的工作、产出文件、未解决的问题等",
                },
            },
            "required": ["summary"],
        }

    def execute(self, summary: str = "") -> str:
        """返回任务总结。

        paras:
            summary: 任务完成总结
        return: summary 原文（loop 检测到 finish 即退出，一般不会真正执行到）
        """
        return summary

"""结束工具 — Agent 调用此工具表示任务完成，主动退出循环"""

from ..base import Tool


class FinishTool(Tool):
    """
    Agent 调用 finish 表示任务已完成，loop 检测到此工具后直接退出
    输入参数 summary, 返回 summary
    此工具什么都不做，但是run_loop过程中识别到该工具调用后结束循环
    """

    def __init__(self):
        super().__init__(
            name="finish",
            description="任务完成时调用此工具，传入总结报告。调用后对话结束。"
                        "**重要：确认所有任务已完成后再调用此工具。**"
        )

    def get_parameters(self) -> dict:
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
        # 实际不会被 _run_loop 调用，loop 在检测到 finish 时直接返回
        return summary
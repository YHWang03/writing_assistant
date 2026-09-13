"""分派工具 — MasterAgent 分派任务给子 Agent
- DispatchTaskTool: 分派任务给指定子 Agent
"""

from ..base import Tool


class DispatchTaskTool(Tool):
    """
    分派任务给指定的子 Agent 执行
    实现完全依赖于dispatch_func
    需要通过set_dispatch方法设置dispatch_func，才能正常工作
    """

    def __init__(self, dispatch_func=None):
        super().__init__(
            name="dispatch_task",
            description=(
                "分派任务给指定的子 Agent 执行。"
                "可用的子 Agent：LiteratureAgent（文献处理）、WritingAgent（论文写作）、"
                "CitationAgent（引用检查）、ReviewAgent（论文评审）、BuildAgent（模板与编译）。"
                "每次调用只分派一个任务，等待子 Agent 返回结果后再决定下一步。"
            ),
        )
        self._dispatch = dispatch_func

    def set_dispatch(self, dispatch_func):
        self._dispatch = dispatch_func

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "子 Agent 名称：LiteratureAgent / WritingAgent / CitationAgent / ReviewAgent / BuildAgent",
                },
                "task": {
                    "type": "string",
                    "description": "要分派的任务描述，应包含具体指令和所需上下文信息",
                },
            },
            "required": ["agent_name", "task"],
        }

    def execute(self, agent_name: str, task: str) -> str:
        if self._dispatch is None:
            return "Error: dispatch_task 工具未正确配置（缺少 dispatch_func）"
        return self._dispatch(agent_name, task)
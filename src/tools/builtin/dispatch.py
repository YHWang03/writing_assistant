"""分派工具 — MasterAgent 分派任务给子 Agent。"""

from ..base import Tool


class DispatchTaskTool(Tool):
    """分派任务给指定的子 Agent 执行（依赖注入 dispatch_func）"""

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
        """注入分派回调。

        paras:
            dispatch_func: callable(agent_name, task) -> str
        """
        self._dispatch = dispatch_func

    def get_parameters(self) -> dict:
        """返回工具参数的 JSON Schema 定义。

        return: input_schema 字典
        """
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
        """分派任务给指定子 Agent。

        paras:
            agent_name: 子 Agent 名称
            task: 任务描述
        return: 子 Agent 的执行结果字符串；未注入回调返回错误字符串
        """
        if self._dispatch is None:
            return "Error: dispatch_task 工具未正确配置（缺少 dispatch_func）"
        return self._dispatch(agent_name, task)

"""ReviewAgent — 评审 Agent：阅读论文草稿，从多维度评审并生成结构化评审报告。"""

from ..core.agent import Agent
from ..core.llm import LLM
from ..prompts import load_prompt
from ..tools.builtin import ReadFileTool, WriteFileTool, ListFilesTool, ReadContextTool, FinishTool
from ._common import STANDARD_HOOKS


class ReviewAgent(Agent):
    """评审 Agent"""

    tool_types = (ReadFileTool, WriteFileTool, ListFilesTool, ReadContextTool, FinishTool)
    hook_types = STANDARD_HOOKS

    def __init__(self, llm: LLM, max_steps: int = 20, max_tokens: int = 12288,
                 run_mode: str = "react"):
        """初始化 ReviewAgent，加载系统提示词并注册工具集。

        paras:
        llm: LLM 实例
        max_steps: 最大执行步数
        max_tokens: 单次生成最大 token 数
        run_mode: 运行模式（react/plan_execute）
        return: 无
        """
        super().__init__(
            name="ReviewAgent", llm=llm,
            system_prompt=load_prompt("review_agent.md"),
            max_steps=max_steps, max_tokens=max_tokens,
            run_mode=run_mode,
        )
        self._setup_declared_components()

    def _sync_context_to_tools(self):
        """将 context 注入到 ReadContextTool。"""
        if self.context is not None:
            self.require_tool("read_context").set_context(self.context)

    def run(self, input_text: str) -> str:
        """运行 ReviewAgent 执行循环。

        paras:
        input_text: 任务输入文本
        return: 执行结果文本
        """
        return self._run_loop(input_text, verbose=True)

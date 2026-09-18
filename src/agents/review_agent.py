"""
ReviewAgent — 评审 Agent

负责：
- 阅读论文草稿
- 从多个维度评审（结构、逻辑、语言、引用、格式）
- 生成结构化的评审报告（review_report.txt）

注意：ReviewAgent 只负责评审和生成报告，不负责修改 TeX 文件。
修改 TeX 的工作由 MasterAgent 分派给 WritingAgent 完成。
"""

from ..core.agent import Agent
from ..core.llm import LLM
from ..prompts import load_prompt
from ..tools.registry import ToolRegistry
from ..tools.builtin import ReadFileTool, WriteFileTool, ListFilesTool, ReadContextTool, FinishTool


class ReviewAgent(Agent):
    """评审 Agent"""

    def __init__(self, llm: LLM, max_steps: int = 20, max_tokens: int = 12288,
                 run_mode: str = "react"):
        super().__init__(
            name="ReviewAgent", llm=llm,
            system_prompt=load_prompt("review_agent.md"),
            max_steps=max_steps, max_tokens=max_tokens,
            run_mode=run_mode,
        )
        self._setup_tools()

    def _setup_tools(self):
        self.tool_registry = ToolRegistry()
        self.tool_registry.register(ReadFileTool())
        write_file = WriteFileTool()
        write_file.set_agent_name("ReviewAgent")
        self.tool_registry.register(write_file)
        self.tool_registry.register(ListFilesTool())
        self._read_context = ReadContextTool()
        self.tool_registry.register(self._read_context)
        self.tool_registry.register(FinishTool())

    def _sync_context_to_tools(self):
        """将 context 注入到 ReadContextTool"""
        if self.context is not None:
            try:
                self._read_context.set_context(self.context)
            except Exception:
                pass

    def run(self, input_text: str) -> str:
        return self._run_loop(input_text, verbose=True)
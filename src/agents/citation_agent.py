"""CitationAgent — 引用检查 Agent：扫描 \\cite 引用、比对文献库并生成引用检查报告。"""

from ..core.agent import Agent
from ..core.llm import LLM
from ..prompts import load_prompt
from ..tools.builtin import (
    ReadFileTool, WriteFileTool, ListFilesTool, ScanCitationsTool,
    LookupPaperInfoTool, CompareCitationTool, ValidateAllCitationsTool,
    ReadContextTool, ListCiteKeysTool, FinishTool,
)
from ._common import STANDARD_HOOKS


class CitationAgent(Agent):
    """引用检查 Agent"""

    tool_types = (
        ReadFileTool, WriteFileTool, ListFilesTool, ValidateAllCitationsTool,
        ScanCitationsTool, LookupPaperInfoTool, CompareCitationTool,
        ReadContextTool, ListCiteKeysTool, FinishTool,
    )
    hook_types = STANDARD_HOOKS

    def __init__(self, llm: LLM, max_steps: int = 12, max_tokens: int = 8192,
                 run_mode: str = "react"):
        """初始化 CitationAgent，加载系统提示词并注册工具集。

        paras:
        llm: LLM 实例
        max_steps: 最大执行步数
        max_tokens: 单次生成最大 token 数
        run_mode: 运行模式（react/plan_execute）
        return: 无
        """
        super().__init__(
            name="CitationAgent", llm=llm,
            system_prompt=load_prompt("citation_agent.md"),
            max_steps=max_steps, max_tokens=max_tokens,
            run_mode=run_mode,
        )
        self._setup_declared_components()

    def _sync_context_to_tools(self):
        """将 PaperContext 中的 reference_library 与 context 注入到各工具。"""
        if self.context is None:
            return
        references = self.context.get_references()
        self.require_tool("lookup_paper_info").set_reference_library(references)
        self.require_tool("validate_all_citations").set_reference_library(references)
        self.require_tool('compare_citation').reference_provider = self.context.get_references
        self.require_tool("list_cite_keys").set_reference_library(references)
        self.require_tool("read_context").set_context(self.context)

    def run(self, input_text: str) -> str:
        """运行 CitationAgent 执行循环。

        paras:
        input_text: 任务输入文本
        return: 执行结果文本
        """
        return self._run_loop(input_text, verbose=True)

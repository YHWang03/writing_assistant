"""
CitationAgent — 引用检查 Agent

负责：
- 扫描 .tex 文件中的所有 \\cite{} 引用
- 从文献库查找被引论文信息
- 比对引用描述与被引文献原文是否一致
- 生成引用检查报告
"""

from ..core.agent import Agent
from ..core.llm import LLM
from ..prompts import load_prompt
from ..tools.registry import ToolRegistry
from ..tools.builtin import (
    ReadFileTool, WriteFileTool, ListFilesTool, ScanCitationsTool,
    LookupPaperInfoTool, CompareCitationTool, ValidateAllCitationsTool,
    ReadContextTool, ListCiteKeysTool, FinishTool,
)


class CitationAgent(Agent):
    """引用检查 Agent"""

    def __init__(self, llm: LLM, max_steps: int = 12, max_tokens: int = 8192,
                 run_mode: str = "react"):
        super().__init__(
            name="CitationAgent", llm=llm,
            system_prompt=load_prompt("citation_agent.md"),
            max_steps=max_steps, max_tokens=max_tokens,
            run_mode=run_mode,
        )
        self._setup_tools()

    def _setup_tools(self):
        self.tool_registry = ToolRegistry()
        self.tool_registry.register(ReadFileTool())
        write_file = WriteFileTool()
        write_file.set_agent_name("CitationAgent")
        self.tool_registry.register(write_file)
        self.tool_registry.register(ListFilesTool())
        self._validate_all = ValidateAllCitationsTool()
        self.tool_registry.register(self._validate_all)
        self.tool_registry.register(ScanCitationsTool())
        self.tool_registry.register(LookupPaperInfoTool())
        self.tool_registry.register(CompareCitationTool())
        self._read_context = ReadContextTool()
        self.tool_registry.register(self._read_context)
        self._list_cite_keys = ListCiteKeysTool()
        self.tool_registry.register(self._list_cite_keys)
        self.tool_registry.register(FinishTool())

    def _sync_context_to_tools(self):
        """将 PaperContext 中的 reference_library 注入到各工具"""
        if self.context is None:
            return
        try:
            lookup = self.tool_registry.get_tool("lookup_paper_info")
            if lookup is not None:
                lookup.set_reference_library(self.context.reference_library)
        except Exception:
            pass
        try:
            self._validate_all.set_reference_library(self.context.reference_library)
        except Exception:
            pass
        try:
            self._list_cite_keys.set_reference_library(self.context.reference_library)
        except Exception:
            pass
        try:
            self._read_context.set_context(self.context)
        except Exception:
            pass

    def run(self, input_text: str) -> str:
        return self._run_loop(input_text, verbose=True)
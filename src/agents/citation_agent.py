"""CitationAgent — 引用检查 Agent：扫描 \\cite 引用、比对文献库并生成引用检查报告。"""

from ..core.agent import Agent
from ..core.llm import LLM
from ..prompts import load_prompt
from ..tools.registry import ToolRegistry
from ..tools.builtin import (
    ReadFileTool, WriteFileTool, ListFilesTool, ScanCitationsTool,
    LookupPaperInfoTool, CompareCitationTool, ValidateAllCitationsTool,
    ReadContextTool, ListCiteKeysTool, FinishTool,
)
from ..hooks.builtin import (
    MemoryRecallHook, DeadlineNudgeHook, CollectWrittenPathsHook,
    FinishNudgeHook, OutputGateHook, MemoryExtractHook,
)


class CitationAgent(Agent):
    """引用检查 Agent"""

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
        self._setup_tools()
        self._setup_hooks()

    def _setup_tools(self):
        """注册本 Agent 的工具集。

        paras: 无
        return: 无
        """
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

    def _setup_hooks(self):
        """注册本 Agent 的 hooks。

        paras: 无
        return: 无
        """
        self.hooks.register(MemoryRecallHook())
        self.hooks.register(DeadlineNudgeHook())
        self.hooks.register(CollectWrittenPathsHook())
        self.hooks.register(FinishNudgeHook())
        self.hooks.register(OutputGateHook())
        self.hooks.register(MemoryExtractHook())

    def _sync_context_to_tools(self):
        """将 PaperContext 中的 reference_library 与 context 注入到各工具。

        paras: 无
        return: 无
        """
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
        """运行 CitationAgent 执行循环。

        paras:
        input_text: 任务输入文本
        return: 执行结果文本
        """
        return self._run_loop(input_text, verbose=True)

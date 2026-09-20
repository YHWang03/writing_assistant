"""WritingAgent — 论文写作 Agent：基于模板撰写论文各章节并生成完整的 paper_draft.tex。"""

from ..core.agent import Agent
from ..core.llm import LLM
from ..prompts import load_prompt
from ..tools.registry import ToolRegistry
from ..tools.builtin import ReadFileTool, WriteFileTool, ListFilesTool, ScanCitationsTool, LookupPaperInfoTool, ReadContextTool, ListCiteKeysTool, FinishTool
from ..hooks.builtin import (
    MemoryRecallHook, DeadlineNudgeHook, CollectWrittenPathsHook,
    FinishNudgeHook, OutputGateHook, MemoryExtractHook,
)


class WritingAgent(Agent):
    """论文写作 Agent"""

    def __init__(self, llm: LLM, max_steps: int = 10, max_tokens: int = 8192,
                 run_mode: str = "react"):
        """初始化 WritingAgent，加载系统提示词、注册工具集并设置产出闸门。

        paras:
        llm: LLM 实例
        max_steps: 最大执行步数
        max_tokens: 单次生成最大 token 数
        run_mode: 运行模式（react/plan_execute）
        return: 无
        """
        super().__init__(
            name="WritingAgent", llm=llm,
            system_prompt=load_prompt("writing_agent.md"),
            max_steps=max_steps, max_tokens=max_tokens,
            run_mode=run_mode,
        )
        self._setup_tools()
        self._setup_hooks()
        self.required_output_exts = [".tex"]

    def _setup_tools(self):
        """注册本 Agent 的工具集。

        paras: 无
        return: 无
        """
        self.tool_registry = ToolRegistry()
        read_file = ReadFileTool()
        read_file.set_agent_name("WritingAgent")
        self.tool_registry.register(read_file)
        write_file = WriteFileTool()
        write_file.set_agent_name("WritingAgent")
        self.tool_registry.register(write_file)
        self.tool_registry.register(ListFilesTool())
        self.tool_registry.register(ScanCitationsTool())
        self.tool_registry.register(LookupPaperInfoTool())
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
            self._list_cite_keys.set_reference_library(self.context.reference_library)
        except Exception:
            pass
        try:
            self._read_context.set_context(self.context)
        except Exception:
            pass

    def run(self, input_text: str) -> str:
        """运行 WritingAgent 执行循环。

        paras:
        input_text: 任务输入文本
        return: 执行结果文本
        """
        return self._run_loop(input_text, verbose=True)

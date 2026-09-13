"""
WritingAgent — 论文写作 Agent

负责：
- 读取 LaTeX 模板
- 撰写论文各章节（abstract, introduction, methods, results, conclusion, related work）
- 根据用户提供的创新点、实验描述、公式等撰写内容
- 生成完整的 paper_draft.tex
"""

from ..core.agent import Agent
from ..core.llm import LLM
from ..prompts import load_prompt
from ..tools.registry import ToolRegistry
from ..tools.builtin import ReadFileTool, WriteFileTool, ListFilesTool, ScanCitationsTool, LookupPaperInfoTool, ReadContextTool, MemoryTool, ListCiteKeysTool, FinishTool


class WritingAgent(Agent):
    """论文写作 Agent"""

    def __init__(self, llm: LLM, max_steps: int = 10, max_tokens: int = 8192,
                 run_mode: str = "react"):
        super().__init__(
            name="WritingAgent", llm=llm,
            system_prompt=load_prompt("writing_agent.md"),
            max_steps=max_steps, max_tokens=max_tokens,
            run_mode=run_mode,
        )
        self._setup_tools()
        # 确定性产出闸门：完成前必须写出 .tex 文件（存在且非空）才允许 finish
        self.required_output_exts = [".tex"]

    def _setup_tools(self):
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
        self.tool_registry.register(MemoryTool())
        self.tool_registry.register(FinishTool())

    def _sync_context_to_tools(self):
        """将 PaperContext 中的 reference_library 注入到 LookupPaperInfoTool，
        并将 memory_manager 注入到 MemoryTool"""
        # memory 注入不依赖 context，独立处理
        try:
            mem_tool = self.tool_registry.get_tool("memory")
            if mem_tool is not None and self.memory_manager is not None:
                mem_tool.set_memory_manager(self.memory_manager)
        except Exception:
            pass

        if self.context is None:
            return
        try:
            lookup = self.tool_registry.get_tool("lookup_paper_info")
            if lookup is not None:
                lookup.set_reference_library(self.context.reference_library) # 将agent上下文的reference_library注入到lookup_paper_info工具中
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
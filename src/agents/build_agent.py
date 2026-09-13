"""
BuildAgent — 构建 Agent

负责：
- 搜索/验证/下载 LaTeX 模板
- 编译 .tex 文件生成 PDF
- 解析编译日志，定位错误
"""

from ..core.agent import Agent
from ..core.llm import LLM
from ..prompts import load_prompt
from ..tools.registry import ToolRegistry
from ..tools.builtin import (
    DeleteFileTool, ReadFileTool, WriteFileTool, ListFilesTool,
    SearchTemplateTool, ValidateTemplateTool, DownloadTemplateTool,
    CompileLatexTool, ParseLatexLogTool, ReadContextTool, MemoryTool,
    FinishTool,
)


class BuildAgent(Agent):
    """构建 Agent：模板管理 + LaTeX 编译"""

    def __init__(self, llm: LLM, max_steps: int = 8, max_tokens: int = 8192,
                 run_mode: str = "react"):
        super().__init__(
            name="BuildAgent", llm=llm,
            system_prompt=load_prompt("build_agent.md"),
            max_steps=max_steps, max_tokens=max_tokens,
            run_mode=run_mode,
        )
        self._setup_tools()

    def _setup_tools(self):
        self.tool_registry = ToolRegistry()
        self.tool_registry.register(ReadFileTool())
        write_file = WriteFileTool()
        write_file.set_agent_name("BuildAgent")
        self.tool_registry.register(write_file)
        self.tool_registry.register(ListFilesTool())
        self.tool_registry.register(SearchTemplateTool())
        self.tool_registry.register(ValidateTemplateTool())
        self.tool_registry.register(DownloadTemplateTool())
        self.tool_registry.register(CompileLatexTool())
        self.tool_registry.register(ParseLatexLogTool())
        self.tool_registry.register(DeleteFileTool())
        self._read_context = ReadContextTool()
        self.tool_registry.register(self._read_context)
        self.tool_registry.register(MemoryTool())
        self.tool_registry.register(FinishTool())

    def _sync_context_to_tools(self):
        """将 context 注入到 ReadContextTool，memory_manager 注入到 MemoryTool"""
        try:
            mem_tool = self.tool_registry.get_tool("memory")
            if mem_tool is not None and self.memory_manager is not None:
                mem_tool.set_memory_manager(self.memory_manager)
        except Exception:
            pass

        if self.context is not None:
            try:
                self._read_context.set_context(self.context)
            except Exception:
                pass

    def run(self, input_text: str) -> str:
        return self._run_loop(input_text, verbose=True)
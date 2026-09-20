"""BuildAgent — 构建 Agent：验证 LaTeX 模板、编译 .tex 生成 PDF、解析编译日志定位错误。"""

from ..core.agent import Agent
from ..core.llm import LLM
from ..prompts import load_prompt
from ..tools.registry import ToolRegistry
from ..tools.builtin import (
    DeleteFileTool, ReadFileTool, WriteFileTool, ListFilesTool,
    ValidateTemplateTool,
    CompileLatexTool, ParseLatexLogTool, ReadContextTool,
    FinishTool,
)
from ..hooks.builtin import (
    MemoryRecallHook, DeadlineNudgeHook, CollectWrittenPathsHook,
    FinishNudgeHook, OutputGateHook, MemoryExtractHook,
)


class BuildAgent(Agent):
    """构建 Agent：模板管理 + LaTeX 编译"""

    def __init__(self, llm: LLM, max_steps: int = 8, max_tokens: int = 8192,
                 run_mode: str = "react"):
        """初始化 BuildAgent，加载系统提示词并注册工具集。

        paras:
        llm: LLM 实例
        max_steps: 最大执行步数
        max_tokens: 单次生成最大 token 数
        run_mode: 运行模式（react/plan_execute）
        return: 无
        """
        super().__init__(
            name="BuildAgent", llm=llm,
            system_prompt=load_prompt("build_agent.md"),
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
        write_file.set_agent_name("BuildAgent")
        self.tool_registry.register(write_file)
        self.tool_registry.register(ListFilesTool())
        self.tool_registry.register(ValidateTemplateTool())
        self.tool_registry.register(CompileLatexTool())
        self.tool_registry.register(ParseLatexLogTool())
        self.tool_registry.register(DeleteFileTool())
        self._read_context = ReadContextTool()
        self.tool_registry.register(self._read_context)
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
        """将 context 注入到 ReadContextTool。

        paras: 无
        return: 无
        """
        if self.context is not None:
            try:
                self._read_context.set_context(self.context)
            except Exception:
                pass

    def run(self, input_text: str) -> str:
        """运行 BuildAgent 执行循环。

        paras:
        input_text: 任务输入文本
        return: 执行结果文本
        """
        return self._run_loop(input_text, verbose=True)

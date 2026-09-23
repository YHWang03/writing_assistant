"""BuildAgent — 构建 Agent：验证 LaTeX 模板、编译 .tex 生成 PDF、解析编译日志定位错误。"""

from ..core.agent import Agent
from ..core.llm import LLM
from ..prompts import load_prompt
from ..tools.builtin import (
    DeleteFileTool, ReadFileTool, WriteFileTool, ListFilesTool,
    ValidateTemplateTool,
    CompileLatexTool, ParseLatexLogTool, ReadContextTool,
    FinishTool,
)
from ._common import STANDARD_HOOKS


class BuildAgent(Agent):
    """构建 Agent：模板管理 + LaTeX 编译"""

    tool_types = (
        ReadFileTool, WriteFileTool, ListFilesTool, ValidateTemplateTool,
        CompileLatexTool, ParseLatexLogTool, DeleteFileTool, ReadContextTool, FinishTool,
    )
    hook_types = STANDARD_HOOKS

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
        self._setup_declared_components()

    def _sync_context_to_tools(self):
        """将 context 注入到 ReadContextTool。

        paras: 无
        return: 无
        """
        if self.context is not None:
            self.require_tool("read_context").set_context(self.context)

    def run(self, input_text: str) -> str:
        """运行 BuildAgent 执行循环。

        paras:
        input_text: 任务输入文本
        return: 执行结果文本
        """
        return self._run_loop(input_text, verbose=True)

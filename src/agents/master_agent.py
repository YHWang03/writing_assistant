"""MasterAgent — 主 Agent：规划流程并通过 dispatch_task 将任务分派给子 Agent。"""

import logging
from ..core.agent import Agent
from ..core.llm import LLM
from ..prompts import load_prompt
from ..tools.registry import ToolRegistry
from ..tools.builtin import ReadFileTool, ListFilesTool, DispatchTaskTool, ReadContextTool, FinishTool
from ..hooks.builtin import MemoryRecallHook, MemoryExtractHook

logger = logging.getLogger(__name__)


class MasterAgent(Agent):
    """主 Agent：路由协调，不直接处理论文内容"""

    def __init__(self, llm: LLM,
                 max_steps: int = 12, max_tokens: int = 8192,
                 run_mode: str = "react"):
        """初始化 MasterAgent，创建子 Agent 注册表并注册工具集。

        paras:
        llm: LLM 实例
        max_steps: 最大执行步数
        max_tokens: 单次生成最大 token 数
        run_mode: 运行模式（react/plan_execute）
        return: 无
        """
        super().__init__(
            name="MasterAgent", llm=llm,
            system_prompt=load_prompt("system_prompt.md"),
            max_steps=max_steps, max_tokens=max_tokens,
            run_mode=run_mode,
        )
        self.sub_agents: dict[str, Agent] = {}
        self._setup_tools()
        self._setup_hooks()

    def _setup_tools(self):
        """注册本 Agent 的工具集。

        paras: 无
        return: 无
        """
        self.tool_registry = ToolRegistry()
        self.tool_registry.register(ReadFileTool())
        self.tool_registry.register(ListFilesTool())

        self._dispatch_tool = DispatchTaskTool(dispatch_func=self.dispatch)
        self.tool_registry.register(self._dispatch_tool)

        self._read_context = ReadContextTool()
        self.tool_registry.register(self._read_context)

        self.tool_registry.register(FinishTool())

    def _setup_hooks(self):
        """注册本 Agent 的 hooks（仅记忆召回与提取，本 Agent 不写产出文件）。

        paras: 无
        return: 无
        """
        self.hooks.register(MemoryRecallHook())
        self.hooks.register(MemoryExtractHook())

    def register_sub_agent(self, name: str, agent: Agent):
        """注册子 Agent。

        paras:
        name: 子 Agent 名称
        agent: 子 Agent 实例
        return: 无
        """
        self.sub_agents[name] = agent

    def dispatch(self, agent_name: str, task: str) -> str:
        """分派任务给指定子 Agent 并返回其执行结果。

        paras:
        agent_name: 子 Agent 名称
        task: 分派的任务描述
        return: 子 Agent 执行结果或错误信息
        """
        agent = self.sub_agents.get(agent_name)
        if agent is None:
            available = list(self.sub_agents.keys())
            return f"Error: 子 Agent '{agent_name}' 未注册。可用: {available}"
        logger.info(f"Dispatching to {agent_name}...")
        try:
            result = agent.run(task)
            logger.info(f"{agent_name} completed.")
            return result
        except Exception as e:
            return f"Error: {agent_name} 执行失败 — {e}"

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
        """主 Agent 运行入口，理解需求、分派任务并汇总结果。

        paras:
        input_text: 用户任务输入
        return: 汇总结果文本
        """
        return self._run_loop(input_text, verbose=True)

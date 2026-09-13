"""
MasterAgent — 主 Agent，负责任务路由与协调

- 能看到全部 PaperContext
- 工具：read_file（读取提示词/配置文件）、dispatch_task（分派子 Agent）
- 不直接执行论文处理，而是规划流程并分派给子 Agent
- LLM 通过 dispatch_task 工具自主决定何时调用哪个子 Agent
"""

import logging
from ..core.agent import Agent
from ..core.llm import LLM
from ..prompts import load_prompt
from ..tools.registry import ToolRegistry
from ..tools.builtin import ReadFileTool, ListFilesTool, DispatchTaskTool, MemoryTool, ReadContextTool, FinishTool

logger = logging.getLogger(__name__)


class MasterAgent(Agent):
    """主 Agent：路由协调，不直接处理论文内容"""

    def __init__(self, llm: LLM,
                 max_steps: int = 12, max_tokens: int = 8192,
                 run_mode: str = "react"):
        super().__init__(
            name="MasterAgent", llm=llm,
            system_prompt=load_prompt("system_prompt.md"),
            max_steps=max_steps, max_tokens=max_tokens,
            run_mode=run_mode,
        )
        self.sub_agents: dict[str, Agent] = {}
        self._setup_tools()

    def _setup_tools(self):
        self.tool_registry = ToolRegistry()
        self.tool_registry.register(ReadFileTool())
        self.tool_registry.register(ListFilesTool())

        # 注册 dispatch_task 工具 — MasterAgent 的核心能力
        self._dispatch_tool = DispatchTaskTool(dispatch_func=self.dispatch)
        self.tool_registry.register(self._dispatch_tool)

        self._read_context = ReadContextTool()
        self.tool_registry.register(self._read_context)

        self.tool_registry.register(MemoryTool())
        self.tool_registry.register(FinishTool())

    def register_sub_agent(self, name: str, agent: Agent):
        """注册子 Agent"""
        self.sub_agents[name] = agent

    def dispatch(self, agent_name: str, task: str) -> str:
        """分派任务给子 Agent"""
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
        """
        主 Agent 运行入口：
        1. 理解用户需求（通过 LLM + read_file 工具）
        2. 规划流程，通过 dispatch_task 工具分派给子 Agent
        3. 汇总结果
        """
        return self._run_loop(input_text, verbose=True)
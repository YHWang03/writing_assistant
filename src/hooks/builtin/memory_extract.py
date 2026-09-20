"""记忆提取 hook — 任务出口从对话提取持久化知识入库。"""

import logging

from ..base import Hook
from ...core.llm import get_tool_llm

logger = logging.getLogger(__name__)


class MemoryExtractHook(Hook):
    """stop：任务出口提取持久化知识入库。"""

    def on_stop(self, agent, user_input, result):
        """从最近对话提取可持久化知识存入 agent.memory，达到阈值时触发整理。

        paras:
            agent: 宿主 Agent
            user_input: 本次任务的用户输入
            result: 本次任务的最终回答
        return: None
        """
        if agent.memory is None:
            return None
        try:
            lines = [f"{m.role}: {str(m.content)[:500]}" for m in agent._history[-12:]]
            lines.append(f"user: {str(user_input)[:500]}")
            lines.append(f"assistant: {str(result)[:500]}")
            dialogue = "\n".join(lines)[:8000]
            stored = agent.memory.extract(dialogue, llm=get_tool_llm())
            if stored:
                agent.memory.consolidate(llm=get_tool_llm())
        except Exception as e:
            logger.warning(f"[{agent.name}] 记忆提取失败: {e}")
        return None

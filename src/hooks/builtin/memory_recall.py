"""记忆召回 hook — 任务入口按任务文本召回相关记忆。"""

import logging

from ..base import Hook
from ...core.llm import get_tool_llm

logger = logging.getLogger(__name__)


class MemoryRecallHook(Hook):
    """user_prompt_submit：任务入口召回相关记忆。"""

    def on_user_prompt_submit(self, agent, user_input):
        """按任务文本召回记忆，写入 agent._recalled_memory 供注入 system prompt。

        paras:
            agent: 宿主 Agent（需有 memory 属性）
            user_input: 用户任务文本
        return: None
        """
        agent._recalled_memory = ""
        if agent.memory is None:
            return None
        try:
            recalled = agent.memory.recall(user_input, llm=get_tool_llm())
            if recalled:
                agent._recalled_memory = recalled
                logger.info(f"[{agent.name}] 召回相关记忆 {len(recalled)} 字符",
                            extra={"event": "memory_recall", "agent": agent.name,
                                   "chars": len(recalled)})
        except Exception as e:
            logger.warning(f"[{agent.name}] 记忆召回失败: {e}")
        return None

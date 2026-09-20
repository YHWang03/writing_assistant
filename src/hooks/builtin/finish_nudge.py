"""finish 调用催办 hook — 模型未调用 finish 即返回文本时注入提示。"""

import logging

from ..base import Hook

logger = logging.getLogger(__name__)

# 连续无工具调用时的 finish 提示上限
MAX_TEXT_ONLY = 2

_NUDGE = (
    "你还没有调用 finish，任务尚未完成。不要就此停止。"
    "请继续执行剩余工作（尤其是写出任务要求的产出文件），"
    "确认全部完成后调用 finish。"
)


class FinishNudgeHook(Hook):
    """text_only：模型未调用 finish 即返回文本时注入提示（连续上限后放行给后续 hook）。"""

    def __init__(self, max_streak: int = MAX_TEXT_ONLY):
        """配置连续提示上限。

        paras:
            max_streak: 最多注入次数
        return: 无
        """
        self._max = max_streak
        self._streak = 0

    def reset(self):
        """清空连续计数。

        paras: 无
        return: 无
        """
        self._streak = 0

    def on_text_only(self, agent, result_text, require_finish):
        """未 finish 的纯文本回答注入提示，连续达上限返回 None。

        paras:
            agent: 宿主 Agent
            result_text: 文本回答（不使用）
            require_finish: False 时不干预
        return: 提示文本；不触发返回 None
        """
        if not require_finish or not agent._has_tool("finish"):
            return None
        if self._streak >= self._max:
            return None
        self._streak += 1
        logger.info(
            f"[{agent.name}] 未调用 finish 即返回文本，注入提示继续"
            f"（第 {self._streak}/{self._max} 次）",
            extra={"event": "finish_nudge", "agent": agent.name,
                   "streak": self._streak},
        )
        return _NUDGE

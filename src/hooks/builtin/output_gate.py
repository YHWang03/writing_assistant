"""产出闸门 hook — 产出闸门未达标时阻止结束（finish 与纯文本出口共享计数）。"""

import logging

from ..base import Hook

logger = logging.getLogger(__name__)

# 闸门提示总次数上限（finish 拦截 + 文本退出拦截共用计数）
MAX_OUTPUT_NUDGES = 3

_FINISH_TEMPLATE = (
    "任务尚未完成：你还没有写出要求的产出文件（{exts}）。不要就此 finish，"
    "请立即用已有材料写出产出文件，确认已写入且非空后再调用 finish。"
)

_TEXT_TEMPLATE = (
    "你还没有写出要求的产出文件（{exts}），任务尚未完成。不要就此结束，"
    "请立即写出产出文件，确认已写入且非空后再调用 finish。"
)


class OutputGateHook(Hook):
    """pre_finish + text_only：产出闸门未达标时阻止结束（两类出口共享计数）。"""

    def __init__(self, max_nudges: int = MAX_OUTPUT_NUDGES):
        """配置提示总次数上限。

        paras:
            max_nudges: finish 拦截与文本退出拦截合计上限
        return: 无
        """
        self._max = max_nudges
        self._count = 0

    def reset(self):
        """清空闸门计数。

        paras: 无
        return: 无
        """
        self._count = 0

    def _gate_nudge(self, agent, template: str):
        """闸门未达标且次数未超时返回提示，否则 None。

        paras:
            agent: 宿主 Agent
            template: 提示文本模板（含 {exts}）
        return: 提示文本；不触发返回 None
        """
        if (not agent.required_output_exts or self._count >= self._max
                or agent._output_satisfied(agent._written_paths)):
            return None
        self._count += 1
        exts = ", ".join(agent.required_output_exts)
        logger.info(
            f"[{agent.name}] 产出闸门未达标（需 {exts}），阻止结束"
            f"（第 {self._count}/{self._max} 次）",
            extra={"event": "output_gate", "agent": agent.name,
                   "count": self._count},
        )
        return template.format(exts=exts)

    def on_pre_finish(self, agent, require_finish):
        """finish 请求闸门校验。

        paras:
            agent: 宿主 Agent
            require_finish: False 时不干预
        return: 闸门提示文本；达标返回 None
        """
        if not require_finish:
            return None
        return self._gate_nudge(agent, _FINISH_TEMPLATE)

    def on_text_only(self, agent, result_text, require_finish):
        """纯文本退出闸门校验。

        paras:
            agent: 宿主 Agent
            result_text: 文本回答（不使用）
            require_finish: False 时不干预
        return: 闸门提示文本；达标返回 None
        """
        if not require_finish:
            return None
        return self._gate_nudge(agent, _TEXT_TEMPLATE)

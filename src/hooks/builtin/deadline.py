"""收尾催办 hook — 临近步数上限仍未尝试产出时注入收尾提示。"""

import logging

from ..base import Hook

logger = logging.getLogger(__name__)

# 判定「已产出文件」的工具集合（宽语义：调用尝试即算，不论成败）
OUTPUT_TOOLS = {
    "write_file", "write_bib_file",
    "generate_bib_from_ref_library", "generate_bibtex",
}

_NUDGE = (
    "你已经执行了多步，但还没有写出任何产出文件。"
    "不要继续收集信息或反复调用工具了，立即用现有材料写出任务要求的产出文件，"
    "然后调用 finish。"
)


class DeadlineNudgeHook(Hook):
    """pre_step + post_tool_use：临近步数上限仍未尝试产出时注入收尾提示（每轮一次）。"""

    def __init__(self, lead: int = 3):
        """配置提前催办的步数。

        paras:
            lead: 距上限还剩几步时开始催办
        return: 无
        """
        self._lead = lead
        self._wrote_output = False
        self._nudged = False

    def reset(self):
        """清空本 ReAct 轮次的产出尝试与催办标记。

        paras: 无
        return: 无
        """
        self._wrote_output = False
        self._nudged = False

    def on_post_tool_use(self, agent, block, output):
        """标记产出工具调用尝试（宽语义，不论成败）。

        paras:
            agent: 宿主 Agent
            block: tool_use block
            output: 工具输出（不使用）
        return: None
        """
        if block.name in OUTPUT_TOOLS:
            self._wrote_output = True
        return None

    def on_pre_step(self, agent, step, step_limit, require_finish):
        """临近上限且未尝试产出时注入收尾提示，仅一次。

        paras:
            agent: 宿主 Agent
            step: 当前步数（0 起）
            step_limit: 步数上限
            require_finish: False（plan 子步骤）时不催办
        return: 收尾提示文本；不触发返回 None
        """
        if (require_finish and not self._wrote_output and not self._nudged
                and step >= step_limit - self._lead):
            self._nudged = True
            logger.info(
                f"[{agent.name}] step {step+1} 临近步数上限且未产出文件，注入收尾提示",
                extra={"event": "deadline_nudge", "agent": agent.name, "step": step+1},
            )
            return _NUDGE
        return None

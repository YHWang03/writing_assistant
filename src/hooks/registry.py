"""hook 注册表 — 每 Agent 独立实例，按注册顺序派发事件。"""

from .base import Hook


class HookRegistry:
    """hook 注册表：register 去重保留首次顺序，trigger 返回第一个非 None 结果。"""

    EVENTS = (
        "user_prompt_submit", "pre_step", "pre_tool_use", "post_tool_use",
        "pre_finish", "text_only", "stop",
    )

    # 各事件除 agent 外的参数（hook 返回字符串即作为注入提示/拦截结果）
    EVENT_ARGS = {
        "user_prompt_submit": "user_input",
        "pre_step": "step, step_limit, require_finish",
        "pre_tool_use": "block",
        "post_tool_use": "block, output",
        "pre_finish": "require_finish",
        "text_only": "result_text, require_finish",
        "stop": "user_input, result",
    }

    def __init__(self):
        """构造空注册表。

        paras: 无
        return: 无
        """
        self._hooks: list[Hook] = []

    def register(self, hook: Hook):
        """注册一个 hook 实例（自动接收其重写的全部事件，去重保留首次顺序）。

        paras:
            hook: Hook 子类实例
        return: 无
        """
        if not isinstance(hook, Hook):
            raise TypeError(f"hook 必须是 Hook 子类实例: {hook!r}")
        if hook not in self._hooks:
            self._hooks.append(hook)

    def trigger(self, event: str, agent, *args):
        """派发事件，返回第一个非 None 结果。

        paras:
            event: 事件名（见 EVENTS）
            agent: 宿主 Agent（自动传入）
            args: 事件参数
        return: 第一个非 None 的 hook 返回值；全部 None 返回 None
        """
        if event not in self.EVENTS:
            raise ValueError(f"未知 hook 事件: {event}（可用: {self.EVENTS}）")
        for hook in self._hooks:
            method = getattr(hook, "on_" + event, None)
            if method is None:
                continue
            result = method(agent, *args)
            if result is not None:
                return result
        return None

    def reset_all(self):
        """调用全部 hook 的 reset（ReAct 轮次开始时）。

        paras: 无
        return: 无
        """
        for hook in self._hooks:
            hook.reset()

    def list_names(self) -> list[str]:
        """列出已注册 hook 类名。

        paras: 无
        return: 类名列表（注册顺序）
        """
        return [hook.__class__.__name__ for hook in self._hooks]

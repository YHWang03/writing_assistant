"""hook 基类 — 空标记基类，子类按需重写事件方法。"""


class Hook:
    """hook 基类：子类按需重写 on_<event>（事件与参数见 HookRegistry.EVENTS），
    未重写即不干预；有状态的重写 reset。"""

    def reset(self):
        """清空按 ReAct 轮次持有的状态。

        paras: 无
        return: 无
        """
        pass

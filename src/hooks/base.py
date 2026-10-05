"""hook 基类 — 空标记基类，子类按需重写事件方法。"""


class HookFinish(str):
    '''工具后置 Hook 返回的交付摘要，要求立即结束本次派发。'''


class Hook:
    """hook 基类：子类按需重写 on_<event>（事件与参数见 HookRegistry.EVENTS），
    未重写即不干预；有状态的重写 reset。"""

    def reset(self):
        """清空按 ReAct 轮次持有的状态。"""
        pass

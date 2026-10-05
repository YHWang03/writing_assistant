"""Call attribution and task usage accounting, including worker threads."""

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from functools import wraps
import logging
from threading import Lock
from time import perf_counter
from uuid import uuid4

logger = logging.getLogger(__name__)
_scope = ContextVar("llm_scope", default={})
_ledger = ContextVar("usage_ledger", default=None)
TOKEN_FIELDS = (
    "input_tokens", "output_tokens", "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


@contextmanager
def call_scope(**fields):
    '''临时绑定调用归属字段，退出作用域时恢复原上下文。

    paras:
        **fields: 临时覆盖或补充的调用归属字段。
    return: 供 with 语句使用的上下文管理器，不提供绑定值。
    '''
    token = _scope.set({**_scope.get(), **fields})
    try:
        yield
    finally:
        _scope.reset(token)


def attributed(purpose=None, agent=False):
    '''为函数及其嵌套调用设置日志归属，调用结束后恢复外层作用域。

    paras:
        purpose: 调用用途标签；为空时保留已有用途。
        agent: True 时从首个参数的 name 获取 Agent 名称，清除父调用的工具和步数归属，
            并记录 Agent 执行跨度；此时用途固定为 agent，覆盖 purpose。
    return: 保留原函数元信息及返回行为的装饰器。
    '''
    def decorate(function):
        '''为目标函数创建带调用归属信息的包装器。

        paras:
            function: 待包装或提交执行的函数。
        return: 保留原函数元信息的包装函数。
        '''
        @wraps(function)
        def wrapped(*args, **kwargs):
            '''在指定调用用途或 Agent 归属作用域内执行原函数。

            paras:
                *args: 传递给目标调用的位置参数。
                **kwargs: 传递给原函数的关键字参数。
            return: 原函数的返回值。
            '''
            fields = {}
            if purpose:
                fields["purpose"] = purpose
            if agent:
                fields.update(agent=args[0].name, tool=None, purpose="agent",
                              step=None, tool_use_id=None)
            with call_scope(**fields):
                if agent:
                    from .tracing import span
                    with span('agent'):
                        return function(*args, **kwargs)
                return function(*args, **kwargs)
        return wrapped
    return decorate


def submit_with_context(executor, function, *args):
    # Each worker needs its own Context; ledgers remain shared and locked.
    '''将当前上下文变量复制到线程池提交的任务中。

    paras:
        executor: 用于提交任务的线程池。
        function: 待包装或提交执行的函数。
        *args: 传递给目标调用的位置参数。
    return: 线程池任务对应的 Future。
    '''
    return executor.submit(copy_context().run, function, *args)


class UsageLedger:
    def __init__(self):
        '''初始化线程安全的模型调用账本。
        '''
        self.lock = Lock()
        self.calls = []

    def record(self, event):
        '''持有线程锁，将一次模型调用事件追加到账本。

        paras:
            event: 一次模型调用的用量及归属事件字典。
        '''
        with self.lock:
            self.calls.append(event)

    @staticmethod
    def aggregate(calls):
        '''汇总调用状态及已知 token 用量，并分别统计未知用量次数。

        paras:
            calls: 模型调用事件列表。
        return: 调用数、失败数、截断数和各 token 字段的汇总字典。
        '''
        result = {"calls": len(calls),
                  "failed_calls": sum(c["status"] == "failed" for c in calls),
                  "incomplete_calls": sum(c["status"] == "incomplete" for c in calls)}
        for key in TOKEN_FIELDS:
            values = [c[key] for c in calls if c[key] is not None]
            result[key] = sum(values) if values else None
            result[key + "_unknown_calls"] = len(calls) - len(values)
        return result

    def summary(self):
        '''对调用账本生成总计及按模型、Agent、用途和工具分组的统计。

        return: 总计和各维度分组统计字典。
        '''
        with self.lock:
            calls = list(self.calls)
        result = self.aggregate(calls)
        for field in ("model", "agent", "purpose", "tool"):
            groups = defaultdict(list)
            for call in calls:
                groups[call.get(field) or "unknown"].append(call)
            result["by_" + field] = {
                name: self.aggregate(items) for name, items in groups.items()}
        return result


def record_call(event):
    '''合并调用归属信息，更新当前任务账本并记录结构化模型调用日志。

    paras:
        event: 包含状态、模型、token 用量和耗时的调用事件。
    '''
    event = {**_scope.get(), **event}
    ledger = _ledger.get()
    if ledger is not None:
        ledger.record(event)
    logger.info(
        "LLM %s model=%s input=%s output=%s duration=%.3fs",
        event["status"], event["model"], event["input_tokens"],
        event["output_tokens"], event["duration"],
        extra={"event": "llm_call", **event},
    )


@contextmanager
def task_usage():
    '''创建任务用量作用域，记录开始及最终消耗，异常时标记失败并恢复上下文。

    return: 供 with 语句使用的上下文管理器，绑定当前 UsageLedger。
    '''
    ledger = UsageLedger()
    token = _ledger.set(ledger)
    start = perf_counter()
    status = "success"
    with call_scope(task_id=uuid4().hex, span_id=uuid4().hex, parent_span_id=None):
        logger.info("Task started", extra={"event": "task_start", **_scope.get()})
        try:
            yield ledger
        except BaseException:
            status = "failed"
            raise
        finally:
            summary = ledger.summary()
            logger.info(
                "Task %s: calls=%s input=%s output=%s (known usage only)",
                status, summary["calls"], summary["input_tokens"], summary["output_tokens"],
                extra={"event": "task_usage", **_scope.get(), "status": status,
                       "duration": perf_counter() - start, "summary": summary},
            )
            _ledger.reset(token)

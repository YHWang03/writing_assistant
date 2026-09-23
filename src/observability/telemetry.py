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
    token = _scope.set({**_scope.get(), **fields})
    try:
        yield
    finally:
        _scope.reset(token)


def attributed(purpose=None, agent=False):
    """Attribute nested calls without changing public method signatures."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            fields = {}
            if purpose:
                fields["purpose"] = purpose
            if agent:
                fields.update(agent=args[0].name, tool=None, purpose="agent")
            with call_scope(**fields):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def submit_with_context(executor, function, *args):
    # Each worker needs its own Context; ledgers remain shared and locked.
    return executor.submit(copy_context().run, function, *args)


class UsageLedger:
    def __init__(self):
        self.lock = Lock()
        self.calls = []

    def record(self, event):
        with self.lock:
            self.calls.append(event)

    @staticmethod
    def aggregate(calls):
        result = {"calls": len(calls),
                  "failed_calls": sum(c["status"] == "failed" for c in calls),
                  "incomplete_calls": sum(c["status"] == "incomplete" for c in calls)}
        for key in TOKEN_FIELDS:
            values = [c[key] for c in calls if c[key] is not None]
            result[key] = sum(values) if values else None
            result[key + "_unknown_calls"] = len(calls) - len(values)
        return result

    def summary(self):
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
    ledger = UsageLedger()
    token = _ledger.set(ledger)
    start = perf_counter()
    status = "success"
    with call_scope(task_id=uuid4().hex):
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

"""Usage comes from API responses, including nested and parallel calls."""

from concurrent.futures import ThreadPoolExecutor
import json
import logging
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

from src.core.llm import LLM
from src.observability.logging_setup import StructuredFormatter
from src.observability.telemetry import call_scope, task_usage, submit_with_context


def client(usage=None):
    llm = object.__new__(LLM)
    llm.model = "test-model"
    response = NS(content=[NS(type="text", text="hello")], usage=usage,
                  id="response-1", _request_id="request-1")
    llm._client = NS(messages=NS(create=Mock(return_value=response)))
    return llm


class TestTelemetry(unittest.TestCase):
    def test_both_entrypoints_record_usage_and_keep_return_types(self):
        usage = {"input_tokens": 11, "output_tokens": 3,
                 "cache_read_input_tokens": 8, "provider_extra": {"x": 4}}
        llm = client(usage)
        with self.assertLogs("src.observability.telemetry", logging.INFO) as logs:
            with task_usage() as ledger:
                self.assertEqual(llm.chat([]), "hello")
                self.assertIs(llm.chat_with_tools([], []), llm._client.messages.create.return_value)
        summary = ledger.summary()
        self.assertEqual(summary["input_tokens"], 22)
        self.assertEqual(summary["output_tokens"], 6)
        self.assertEqual(summary["cache_read_input_tokens"], 16)
        self.assertIsNone(summary["cache_creation_input_tokens"])
        events = [json.loads(StructuredFormatter().format(r)) for r in logs.records]
        calls = [e for e in events if e["event"] == "llm_call"]
        self.assertEqual(calls[0]["usage"], usage)
        self.assertNotEqual(calls[0]["call_id"], calls[1]["call_id"])
        self.assertEqual(calls[0]["task_id"], calls[1]["task_id"])
        self.assertEqual(events[-1]["event"], "task_usage")

    def test_unknown_usage_and_failed_task_are_reported(self):
        llm = client()
        with self.assertLogs("src.observability.telemetry", logging.INFO) as logs:
            with self.assertRaises(RuntimeError):
                with task_usage() as ledger:
                    llm.chat([])
                    llm._client.messages.create.side_effect = RuntimeError("private prompt")
                    llm.chat([])
        summary = ledger.summary()
        self.assertEqual(summary["calls"], 2)
        self.assertEqual(summary["failed_calls"], 1)
        self.assertIsNone(summary["input_tokens"])
        self.assertEqual(summary["input_tokens_unknown_calls"], 2)
        event = json.loads(StructuredFormatter().format(logs.records[-1]))
        self.assertEqual(event["status"], "failed")
        self.assertNotIn("private prompt", str(logs.output))

    def test_parallel_calls_keep_scope_and_restore_parent(self):
        llm = client({"input_tokens": 2, "output_tokens": 1})
        with task_usage() as ledger:
            with call_scope(agent="master", purpose="agent"):
                with call_scope(agent="literature", tool="parse_pdf", purpose="tool"):
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        futures = [submit_with_context(executor, llm.chat, []) for _ in range(12)]
                        for future in futures:
                            future.result()
                llm.chat([])
        summary = ledger.summary()
        self.assertEqual(summary["calls"], 13)
        self.assertEqual(summary["by_agent"]["literature"]["calls"], 12)
        self.assertEqual(summary["by_agent"]["master"]["calls"], 1)
        self.assertEqual(summary["by_tool"]["parse_pdf"]["input_tokens"], 24)
        with task_usage() as next_ledger:
            llm.chat([])
        self.assertEqual(next_ledger.summary()["calls"], 1)
        self.assertNotIn("master", next_ledger.summary()["by_agent"])

    def test_partial_usage_preserves_zero_and_marks_missing_fields(self):
        llm = client({"input_tokens": 0, "output_tokens": 0})
        with task_usage() as ledger:
            llm.chat([])
        summary = ledger.summary()
        self.assertEqual(summary["input_tokens"], 0)
        self.assertEqual(summary["input_tokens_unknown_calls"], 0)
        self.assertEqual(summary["cache_read_input_tokens_unknown_calls"], 1)


if __name__ == "__main__":
    unittest.main()

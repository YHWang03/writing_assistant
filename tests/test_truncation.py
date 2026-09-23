"""Offline regressions for output truncation and citation result completeness."""

import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from src.core.llm import LLM, IncompleteResponseError
from src.observability.telemetry import task_usage
from src.tools.builtin.citations.validation import CompareCitationTool, ValidateAllCitationsTool


def response(stop, text="ok"):
    return NS(stop_reason=stop, content=[NS(type="text", text=text)],
              usage={"input_tokens": 10, "output_tokens": 20})


def client(*responses):
    llm = LLM.__new__(LLM)
    llm.model = "test"
    llm._client = Mock()
    llm._client.messages.create.side_effect = responses
    return llm


class TruncationTests(unittest.TestCase):
    def test_recovery_records_both_calls_and_keeps_original_messages(self):
        llm = client(response("max_tokens", "partial"), response("end_turn", "complete"))
        messages = [{"role": "user", "content": "task"}]
        with task_usage() as ledger:
            self.assertEqual(llm.chat(messages, max_tokens=512), "complete")
        self.assertEqual(ledger.summary()["incomplete_calls"], 1)
        self.assertEqual(ledger.summary()["failed_calls"], 0)
        self.assertEqual(ledger.summary()["output_tokens"], 40)
        calls = llm._client.messages.create.call_args_list
        self.assertEqual([c.kwargs["max_tokens"] for c in calls], [512, 1024])
        self.assertEqual(messages, [{"role": "user", "content": "task"}])
        self.assertEqual(ledger.calls[0]["recovery_id"], ledger.calls[1]["recovery_id"])
        self.assertEqual(ledger.calls[1]["attempt"], 2)

    def test_truncated_tool_responses_never_reach_caller(self):
        bad = response("max_tokens")
        bad.content = [NS(type="tool_use", name="write_file", input={})]
        llm = client(bad, bad)
        execute = Mock()
        with self.assertRaises(IncompleteResponseError):
            result = llm.chat_with_tools([], [], max_tokens=8192)
            execute(result)
        execute.assert_not_called()
        self.assertEqual(llm._client.messages.create.call_count, 2)

    def test_batch_can_disable_truncation_retry(self):
        llm = client(response("max_tokens"))
        with self.assertRaises(IncompleteResponseError):
            llm.chat([], truncation_retries=0)
        self.assertEqual(llm._client.messages.create.call_count, 1)

    def test_transport_error_is_not_retried_by_recovery(self):
        llm = client(RuntimeError("network"))
        with self.assertRaises(RuntimeError):
            llm.chat([])
        self.assertEqual(llm._client.messages.create.call_count, 1)

    def test_comparison_does_not_stack_truncation_retries(self):
        llm = client(response("max_tokens"), response("max_tokens"))
        with patch('src.tools.builtin.citations.validation.get_tool_llm', return_value=llm):
            result = CompareCitationTool()._check_one("a", "context", "title", "abstract", as_json=True)
        self.assertIsNone(result)
        self.assertEqual(llm._client.messages.create.call_count, 2)

    def test_invalid_verdict_cannot_pass(self):
        for result in ['{}', '{"error":"bad"}', '{"verdict":"✅"}',
                       '{"cite_key":"other","verdict":"✅","reason":"ok"}']:
            self.assertIsNone(CompareCitationTool._parse_verdict_json(result, "a"))

    def test_batches_are_bounded(self):
        tool = ValidateAllCitationsTool()
        with patch.object(tool, '_compare_chunk', return_value=([], [])) as compare:
            tool._batch_compare([{"cite_key": str(i)} for i in range(10)])
        self.assertEqual([len(c.args[0]) for c in compare.call_args_list], [4, 4, 2])

    def test_partial_or_duplicate_batch_falls_back(self):
        citations = [dict(cite_key=k, paper_title="t", paper_abstract="a", citation_context="c")
                     for k in ("a", "b")]
        item = dict(cite_key="a", verdict="✅", reason="ok")
        for items in ([item], [item, item]):
            tool = ValidateAllCitationsTool()
            with patch('src.tools.builtin.citations.validation.get_tool_llm') as get_llm, \
                    patch.object(tool, '_compare_fallback', return_value=([], citations)) as fallback:
                get_llm.return_value.chat.return_value = json.dumps(items)
                good, bad = tool._batch_compare(citations)
            self.assertEqual(good, [])
            self.assertEqual(len(bad), 2)
            fallback.assert_called_once()


if __name__ == '__main__':
    unittest.main()

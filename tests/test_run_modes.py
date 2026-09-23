"""Fast execution-loop tests with an in-memory fake LLM."""

from types import SimpleNamespace
import unittest

from src.core.agent import Agent
from src.tools.builtin import FinishTool


class FakeLLM:
    def __init__(self, responses=None, chats=None):
        self.responses = list(responses or [])
        self.chats = list(chats or [])

    def chat_with_tools(self, **kwargs):
        return self.responses.pop(0)

    def chat(self, **kwargs):
        return self.chats.pop(0)


def finish_response(summary: str):
    block = SimpleNamespace(
        type="tool_use", name="finish", input={"summary": summary}, id="finish-1")
    usage = SimpleNamespace(input_tokens=10, output_tokens=2)
    return SimpleNamespace(content=[block], usage=usage)


class MinimalAgent(Agent):
    tool_types = (FinishTool,)

    def __init__(self, llm, run_mode="react"):
        super().__init__("MinimalAgent", llm, run_mode=run_mode)
        self._setup_declared_components()

    def run(self, input_text: str) -> str:
        return self._run_loop(input_text, verbose=False)


class TestRunModes(unittest.TestCase):
    def test_react_finish_records_history(self):
        agent = MinimalAgent(FakeLLM(responses=[finish_response("done")]))
        self.assertEqual(agent.run("task"), "done")
        self.assertEqual([message.role for message in agent.get_history()], ["user", "assistant"])

    def test_plan_execute_runs_steps_and_summary(self):
        llm = FakeLLM(
            responses=[finish_response("one"), finish_response("two"), finish_response("three")],
            chats=['["a", "b", "c"]', "all done"],
        )
        agent = MinimalAgent(llm, run_mode="plan_execute")
        self.assertEqual(agent.run("task"), "all done")


if __name__ == "__main__":
    unittest.main()

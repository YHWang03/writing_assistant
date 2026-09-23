"""Regression coverage for state propagation, API messages and file boundaries."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from src.agents import LiteratureAgent
from src.core.agent import Agent
from src.core.message import Message
from src.domain.paper import Paper
from src.domain.paper_context import PaperContext
from src.hooks.base import Hook
from src.hooks.builtin.collect_paths import CollectWrittenPathsHook
from src.tools.builtin import _safe_path


def tool(name, identifier, **params):
    return NS(type="tool_use", name=name, id=identifier, input=params)


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def chat_with_tools(self, **kwargs):
        messages = list(kwargs["messages"])
        self.requests.append(messages)
        pending = set()
        for message in messages:
            if message["role"] not in {"user", "assistant"}:
                raise AssertionError("Invalid API message role")
            content = message["content"]
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        pending.remove(block["tool_use_id"])
                    elif getattr(block, "type", None) == "tool_use":
                        pending.add(block.id)
        if pending:
            raise AssertionError(f"Missing tool results: {pending}")
        return NS(content=next(self.responses), usage=None)


class TestAgent(Agent):
    def run(self, prompt):
        return self._run_loop(prompt, verbose=False)

    def _execute_tool(self, name, params):
        return "ok"


class RejectOnce(Hook):
    def __init__(self):
        self.calls = 0

    def on_pre_finish(self, agent, required):
        self.calls += 1
        return "Continue working" if self.calls == 1 else None


class ReviewRegressions(unittest.TestCase):
    def test_same_run_reference_updates_reach_bib_and_disk(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            context = PaperContext(library_dir=str(root / "library"))
            agent = LiteratureAgent(NS())
            agent.context = context.view(
                {"reference_library", "seed_pdf_paths", "ref_pdf_paths", "library_dir"},
                {"reference_library"},
            )
            agent._sync_context_to_tools()
            agent.context.add_reference(Paper("fresh", "Fresh title", "Author", 2026))
            with patch.object(_safe_path, "_PROJECT_ROOT", root):
                result = agent.require_tool("generate_bib_from_ref_library").execute(
                    str(root / "references.bib"))
            self.assertEqual(json.loads(result)["added_keys"], ["fresh"])
            agent.require_tool("write_library").execute()
            saved = json.loads((root / "library/reference_library.json").read_text())
            self.assertEqual(saved[0]["cite_key"], "fresh")

    def test_second_task_after_tools_has_valid_api_roles(self):
        llm = ScriptedLLM([
            [tool("ls", "1")], [tool("finish", "2", summary="first")],
            [tool("finish", "3", summary="second")],
        ])
        agent = TestAgent("test", llm)
        self.assertEqual(agent.run("first task"), "first")
        self.assertEqual(agent.run("second task"), "second")
        self.assertTrue(all(m.role != "system" for m in agent.get_history()))

    def test_legacy_system_history_is_normalized(self):
        llm = ScriptedLLM([[tool("finish", "1", summary="done")]])
        agent = TestAgent("test", llm)
        agent.add_message(Message("Old tool summary", "system"))
        self.assertEqual(agent.run("task"), "done")

    def test_rejected_finish_preserves_same_batch_tool_results(self):
        llm = ScriptedLLM([
            [tool("ls", "1"), tool("finish", "2", summary="early")],
            [tool("finish", "3", summary="done")],
        ])
        agent = TestAgent("test", llm)
        agent.hooks.register(RejectOnce())
        self.assertEqual(agent.run("task"), "done")
        results = [block for m in llm.requests[1] if isinstance(m["content"], list)
                   for block in m["content"] if isinstance(block, dict)]
        self.assertEqual([b["tool_use_id"] for b in results], ["1"])

    def test_path_boundary_rejects_prefix_sibling_and_traversal(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            with patch.object(_safe_path, "_PROJECT_ROOT", root):
                self.assertEqual(_safe_path.safe_resolve(str(root / "a.txt")), root / "a.txt")
                for target in [root.parent / "project-backup/a.txt", root / "../a.txt"]:
                    with self.assertRaises(ValueError):
                        _safe_path.safe_resolve(str(target))

    def test_chapter_write_does_not_replace_main_tex_path(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            context = PaperContext(output_dir=str(root))
            agent = NS(name="WritingAgent", _written_paths=set(),
                       context=context.view({"output_dir", "main_tex_path"}, {"main_tex_path"}))
            hook = CollectWrittenPathsHook()
            with patch.object(_safe_path, "_PROJECT_ROOT", root):
                for relative in ["main.tex", "sections/intro.tex", "other/main.tex"]:
                    hook.on_post_tool_use(agent, tool("write_file", relative,
                        file_path=str(root / relative)), "written")
            self.assertEqual(context.main_tex_path, str(root / "main.tex"))

    def test_collection_assignment_is_rejected(self):
        context = PaperContext()
        view = context.view({"reference_library", "sections"}, {"reference_library", "sections"})
        with self.assertRaises(AttributeError):
            view.reference_library = []
        with self.assertRaises(AttributeError):
            view.sections = {}
        view.set_section("intro", "text")
        self.assertEqual(context.sections, {"intro": "text"})

    def test_add_reference_does_not_retain_mutable_input(self):
        context = PaperContext()
        view = context.view({"reference_library"}, {"reference_library"})
        original = Paper("key", "Original", "Author", 2026, keywords=["one"])
        view.add_reference(original)
        original.title = "Modified"
        original.keywords.append("two")
        stored = view.get_references()[0]
        self.assertEqual(stored.title, "Original")
        self.assertEqual(stored.keywords, ["one"])


if __name__ == "__main__":
    unittest.main()

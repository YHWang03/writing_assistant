"""Application assembly and lifecycle tests."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from src.bootstrap import build_agents, build_context
from src.config import load_config
from src.config import AGENT_NAMES
from src.domain.paper_context import PaperContext
from src.lifecycle import Application


class TestBootstrap(unittest.TestCase):
    def test_each_agent_memory_can_be_enabled_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            for selected in (*AGENT_NAMES, None):
                with self.subTest(selected=selected):
                    switches = "\n".join(
                        f"    {name}: {'true' if name == selected else 'false'}"
                        for name in AGENT_NAMES)
                    path.write_text("memory:\n  enabled:\n" + switches, encoding="utf-8")
                    with patch("src.bootstrap._memory") as factory:
                        agents = build_agents(load_config(path), SimpleNamespace())
                        for name, agent in agents.items():
                            self.assertEqual(agent.memory is not None, name == selected)
                        self.assertEqual(factory.call_count, int(selected is not None))
                        if selected:
                            self.assertEqual(factory.call_args.args[1], selected)

    def test_disabled_memory_hooks_do_not_create_llm(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("{}", encoding="utf-8")
            agents = build_agents(load_config(path), SimpleNamespace())
            with patch("src.hooks.builtin.memory_recall.get_tool_llm") as recall_llm, \
                 patch("src.hooks.builtin.memory_extract.get_tool_llm") as extract_llm:
                for name, agent in agents.items():
                    if name == "master":
                        continue
                    agent._trigger_hooks("user_prompt_submit", "task")
                    agent._trigger_hooks("stop", "task", "done")
                recall_llm.assert_not_called()
                extract_llm.assert_not_called()

    def test_context_loads_persistent_library(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "library").mkdir()
            (root / "library" / "reference_library.json").write_text(
                json.dumps([{
                    "cite_key": "key", "title": "Title", "authors": "A",
                    "year": 2026, "source": "user",
                }]),
                encoding="utf-8",
            )
            config_path = root / "config.yaml"
            config_path.write_text("paths:\n  library_dir: library\n", encoding="utf-8")
            context = build_context(load_config(config_path))
            self.assertEqual(context.reference_library[0].cite_key, "key")

    def test_agent_specs_are_assembled_declaratively(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("{}", encoding="utf-8")
            agents = build_agents(load_config(path), SimpleNamespace())
            self.assertIn("write_file", agents["writing"].tool_registry.list_names())
            self.assertIn("OutputGateHook", agents["writing"].hooks.list_names())
            self.assertNotIn("OutputGateHook", agents["master"].hooks.list_names())

    def test_context_manager_persists_on_exception(self):
        saved = []
        memory = SimpleNamespace(save=lambda: saved.append(True))
        agent = SimpleNamespace(memory=memory)
        with self.assertRaises(RuntimeError):
            with Application(PaperContext(), {"master": agent}):
                raise RuntimeError("boom")
        self.assertEqual(saved, [True])


if __name__ == "__main__":
    unittest.main()

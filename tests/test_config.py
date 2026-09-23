"""Configuration validation tests."""

from pathlib import Path
import tempfile
import unittest

from src.config import ConfigError, load_config


class TestConfig(unittest.TestCase):
    def _load(self, text: str):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(text, encoding="utf-8")
            return load_config(path)

    def test_defaults_are_typed(self):
        config = self._load("{}")
        self.assertEqual(config.agents.run_mode_for("WritingAgent"), "react")
        self.assertGreater(config.agents.writing.max_steps, 0)

    def test_invalid_run_mode_is_rejected(self):
        with self.assertRaises(ConfigError):
            self._load("agents:\n  run_mode: reflection\n")

    def test_memory_path_requires_agent_placeholder(self):
        with self.assertRaises(ConfigError):
            self._load("memory:\n  path: memory/shared.json\n")

    def test_unknown_path_key_is_rejected(self):
        with self.assertRaises(ConfigError):
            self._load("paths:\n  typo_dir: somewhere\n")

    def test_memory_switch_defaults_and_overrides(self):
        defaults = self._load("{}").memory.enabled
        self.assertEqual([name for name, value in defaults.items() if value], ["master"])
        custom = self._load("memory:\n  enabled:\n    master: false\n    build: true\n")
        self.assertEqual([name for name, value in custom.memory.enabled.items() if value], ["build"])

    def test_memory_switch_rejects_typos_and_non_boolean_values(self):
        for override in ['typo: true', 'master: "false"', 'master: 1']:
            with self.subTest(override=override), self.assertRaises(ConfigError):
                self._load("memory:\n  enabled:\n    " + override + "\n")


if __name__ == "__main__":
    unittest.main()

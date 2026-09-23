"""Command-line interface for the writing assistant."""

from __future__ import annotations

import argparse
import logging
from typing import Sequence

from .bootstrap import create_application
from .core.llm import IncompleteResponseError
from .config import ConfigError, PROJECT_DIR, load_config
from .observability.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="多 Agent 学术论文写作助手")
    parser.add_argument("--config", default="config.yaml", help="YAML 配置文件路径")
    return parser


def interactive(config_path: str) -> int:
    config = load_config(config_path)
    with create_application(config) as app:
        logger.info("=" * 60)
        logger.info("  writing_assistant — 论文写作助手")
        logger.info("  输入 'quit' 退出，输入需求开始写作")
        logger.info("=" * 60)
        while True:
            try:
                prompt = input("\n[You] > ").strip()
            except (EOFError, KeyboardInterrupt):
                logger.info("Goodbye.")
                break
            if not prompt:
                continue
            if prompt.lower() in {"quit", "exit", "q"}:
                logger.info("Goodbye.")
                break
            try:
                logger.info("[MasterAgent] %s", app.run_task(prompt))
            except KeyboardInterrupt:
                logger.info("任务已中断。")
            except IncompleteResponseError as exc:
                logger.error("任务未完成：%s 已保留现有产出，可缩小任务后继续。", exc)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    setup_logging(PROJECT_DIR / "logs")
    try:
        return interactive(args.config)
    except ConfigError as exc:
        logger.error("配置错误: %s", exc)
        return 2

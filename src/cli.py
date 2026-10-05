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
    '''构造命令行参数解析器。

    return: 支持配置文件路径参数的 ArgumentParser。
    '''
    parser = argparse.ArgumentParser(description="多 Agent 学术论文写作助手")
    parser.add_argument("--config", default="config.yaml", help="YAML 配置文件路径")
    return parser


def interactive(config_path: str) -> int:
    '''加载配置并循环接收用户任务，退出时关闭应用。

    paras:
        config_path: YAML 配置文件路径。
    return: 正常退出时返回状态码 0。
    '''
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
    '''初始化日志、解析命令行参数并启动交互入口。

    paras:
        argv: 命令行参数序列；None 表示读取进程参数。
    return: 退出状态码：正常为 0，配置错误为 2。
    '''
    args = _parser().parse_args(argv)
    setup_logging(PROJECT_DIR / "logs")
    try:
        return interactive(args.config)
    except ConfigError as exc:
        logger.error("配置错误: %s", exc)
        return 2

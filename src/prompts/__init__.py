"""提示词集中管理：各 Agent 的 system prompt 以 .md 文件存放在本目录，通过 load_prompt 加载。"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent


def load_prompt(filename: str, fallback: str = "") -> str:
    """按文件名从 prompts 目录加载提示词，文件缺失时返回 fallback 并告警。

    paras:
    filename: 提示词文件名
    fallback: 文件缺失时的兜底内容
    return: 提示词文本
    """
    path = _PROMPTS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    logger.warning(f"提示词文件缺失，使用 fallback: {path}")
    return fallback

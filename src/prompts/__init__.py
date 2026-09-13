"""提示词集中管理。

所有 Agent 的 system prompt 统一放在本目录下的 .md 文件中，
由各 Agent 通过 `load_prompt(<文件名>)` 加载。这样提示词与代码分离，
便于单独迭代提示词而无需改动 Agent 代码。
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent


def load_prompt(filename: str, fallback: str = "") -> str:
    """按文件名从 prompts 目录加载提示词；文件缺失时返回 fallback 并告警。"""
    path = _PROMPTS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    logger.warning(f"提示词文件缺失，使用 fallback: {path}")
    return fallback

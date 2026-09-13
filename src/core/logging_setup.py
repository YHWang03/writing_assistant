"""结构化日志模块 — 输出 JSONL 格式日志，便于查询和分析。

与现有文本日志并行输出，不破坏原有行为。

用法：
    logger.info("tool call", extra={
        "event": "tool_call", "agent": "MasterAgent", "step": 1,
        "tool": "read_file", "params": {"file_path": "..."},
    })
"""

import json
import logging
from pathlib import Path


class StructuredFormatter(logging.Formatter):
    """将日志记录格式化为单行 JSON，自动提取 extra 中的结构化字段。"""

    # 需要从 extra 中提取的字段列表
    _STRUCTURED_KEYS = (
        "event", "agent", "step", "tool", "params",
        "result_size", "is_error", "duration", "tokens",
        "mode", "old_count", "new_chars", "threshold",
        "from_agent", "to_agent", "total_steps",
    )

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": record.created,
            "level": record.levelname,
            "event": getattr(record, "event", "unknown"),
            "msg": record.getMessage(),
        }

        for key in self._STRUCTURED_KEYS:
            if key == "event":
                continue  # 已在上面处理
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val

        return json.dumps(entry, ensure_ascii=False)


def setup_structured_logging(log_path: Path) -> Path:
    """为根 logger 添加结构化 JSONL 输出 Handler。

    返回结构化日志文件路径。
    """
    logs_dir = log_path.parent
    structured_dir = logs_dir / "structured"
    structured_dir.mkdir(exist_ok=True)

    structured_path = structured_dir / f"{log_path.stem}.jsonl"

    handler = logging.FileHandler(str(structured_path), encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(StructuredFormatter())

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    return structured_path
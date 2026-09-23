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
from datetime import date
import re
from pathlib import Path
from .log_report import refresh_reports


def setup_logging(logs_dir: Path) -> Path:
    """Configure console, text and JSONL output for one application run."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().strftime("%Y_%m_%d")
    existing = [int(match.group(1)) for path in logs_dir.iterdir()
                if (match := re.fullmatch(rf"{today}_(\d+)\.log", path.name))]
    log_path = logs_dir / f"{today}_{max(existing, default=0) + 1}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
    )
    structured_path = setup_structured_logging(log_path)
    logging.getLogger(__name__).info(
        "日志文件: %s | 结构化日志: %s", log_path, structured_path,
        extra={"event": "startup", "agent": "root"})
    return log_path


class ReportFileHandler(logging.FileHandler):
    """Refresh offline reports only after the task summary has been flushed."""

    def emit(self, record):
        super().emit(record)
        if getattr(record, "event", None) == "task_usage":
            try:
                refresh_reports(Path(self.baseFilename).parent.parent)
            except Exception:
                # Do not recursively log through this handler or fail the task.
                self.handleError(record)


class StructuredFormatter(logging.Formatter):
    """将日志记录格式化为单行 JSON，自动提取 extra 中的结构化字段。"""

    # 需要从 extra 中提取的字段列表
    _STRUCTURED_KEYS = (
        "event", "agent", "step", "tool", "params",
        "result_size", "is_error", "duration", "tokens",
        "mode", "old_count", "new_chars", "threshold",
        "from_agent", "to_agent", "total_steps",
        "task_id", "call_id", "request_id", "response_id", "model", "response_model",
        "purpose", "operation", "status", "usage", "usage_source", "source",
        "input_tokens", "output_tokens", "cache_read_input_tokens",
        "cache_creation_input_tokens", "summary", "error_type", "http_status", "stop_reason",
        "recovery_id", "attempt", "requested_max_tokens",
        "file", "fingerprint", "attempts", "cache_hit", "error_kind", "retry_exhausted",
    )

    def format(self, record: logging.LogRecord) -> str:
        """把日志记录格式化为单行 JSON。

        paras:
            record: 日志记录
        return: JSON 文本
        """
        entry = {
            "ts": record.created,
            "level": record.levelname,
            "event": getattr(record, "event", "unknown"),
            "msg": record.getMessage(),
        }

        for key in self._STRUCTURED_KEYS:
            if key == "event":
                continue
            val = getattr(record, key, None)
            if hasattr(record, key):
                entry[key] = val

        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, ensure_ascii=False)


def setup_structured_logging(log_path: Path) -> Path:
    """为根 logger 添加 JSONL 结构化输出 Handler。

    paras:
        log_path: 文本日志路径（结构化日志写到同目录 structured/ 下同名 .jsonl）
    return: 结构化日志文件路径
    """
    logs_dir = log_path.parent
    structured_dir = logs_dir / "structured"
    structured_dir.mkdir(exist_ok=True)

    structured_path = structured_dir / f"{log_path.stem}.jsonl"

    handler = ReportFileHandler(str(structured_path), encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(StructuredFormatter())

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    try:
        refresh_reports(logs_dir)
    except Exception:
        logging.getLogger(__name__).warning("HTML 日志报告初始化失败", exc_info=True)

    return structured_path

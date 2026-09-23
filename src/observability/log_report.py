"""Offline HTML views of JSONL usage logs; no server or external assets."""

import html
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import quote


FIELDS = (
    "ts", "event", "task_id", "call_id", "request_id", "model", "agent", "tool",
    "purpose", "status", "duration", "input_tokens", "output_tokens",
    "cache_read_input_tokens", "cache_creation_input_tokens", "usage", "error_type",
    "http_status", "usage_source", "stop_reason",
    "recovery_id", "attempt", "requested_max_tokens",
)


def read_usage(path):
    events, skipped = [], 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except (ValueError, UnicodeError):
                skipped += 1
                continue
            if not isinstance(event, dict):
                skipped += 1
                continue
            if event.get("event") in {"llm_call", "task_start", "task_usage"}:
                events.append({key: event[key] for key in FIELDS if key in event})
    return {"events": events, "skipped": skipped, "run": path.stem}


def atomic_write(path, content):
    """Readers see either the old report or the complete replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as stream:
            name = stream.name
            stream.write(content)
        os.replace(name, path)
    finally:
        if name and Path(name).exists():
            Path(name).unlink()


def render_report(data):
    template = Path(__file__).with_name("log_report.html").read_text(encoding="utf-8")
    # JSON is embedded in a script element: escape markup delimiters, including
    # provider-supplied strings, so logs cannot terminate the element.
    payload = json.dumps(data, ensure_ascii=False).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    return template.replace("__REPORT_DATA__", payload)


def refresh_reports(logs_dir):
    """Update changed runs and the index, including older logs on first use."""
    logs_dir = Path(logs_dir)
    reports = logs_dir / "reports"
    rows = []
    sources = sorted((logs_dir / "structured").glob("*.jsonl"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    for source in sources:
        destination = reports / (source.stem + ".html")
        data = read_usage(source)
        if not destination.exists() or source.stat().st_mtime_ns > destination.stat().st_mtime_ns:
            atomic_write(destination, render_report(data))
        calls = [e for e in data["events"] if e["event"] == "llm_call"]
        tasks = {e.get("task_id") for e in data["events"] if e.get("task_id")}
        rows.append('<tr><td><a href="reports/' + quote(destination.name) + '">' +
                    html.escape(source.stem) + '</a></td><td>' + str(len(tasks)) +
                    '</td><td>' + str(len(calls)) + '</td><td>' +
                    ("可查询" if calls else "无调用明细（空运行或旧版日志）") + '</td></tr>')
    atomic_write(logs_dir / "index.html", '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>写作助手 · 日志首页</title>
<style>body{font:16px system-ui;background:#f4f6fa;color:#172438;margin:5vw}main{max-width:1100px;margin:auto}
table{width:100%;border-collapse:collapse;background:white}td,th{padding:18px;text-align:left;border-bottom:1px solid #e1e5ed}
a{color:#2357b5}p{color:#536176}h1{font-size:32px}</style><main><h1>写作助手 · 运行日志</h1>
<p>选择一次运行查看报告。任务结束后自动更新；已打开的页面请刷新。所有数据保存在本地。</p>
<table><thead><tr><th>运行</th><th>任务数</th><th>模型调用数</th><th>明细状态</th></tr></thead><tbody>''' +
                 ("".join(rows) or '<tr><td colspan="4">暂无运行日志</td></tr>') + '</tbody></table></main></html>')
    return logs_dir / "index.html"

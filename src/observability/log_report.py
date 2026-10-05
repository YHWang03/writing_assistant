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
    '''读取全部原始结构化事件，跳过损坏记录，不丢弃未来页面可能需要的字段。

    paras:
        path: 结构化 JSONL 日志文件的 Path。
    return: 包含事件列表、跳过数量和运行标识的字典。
    '''
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
            events.append(event)
    details = {}
    for event in events:
        for key in ('input_detail', 'output_detail'):
            ref = event.get(key)
            if not isinstance(ref, dict) or not isinstance(ref.get('path'), str):
                continue
            name = ref['path']
            target = (path.parent / name).resolve()
            if name in details or not target.is_relative_to(path.parent.resolve()):
                continue
            try:
                details[name] = json.loads(target.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                pass
    return {"events": events, "skipped": skipped, "run": path.stem, 'details': details}


def atomic_write(path, content):
    '''原子替换报告文件，使读取方只看到旧版本或完整新版本。

    paras:
        path: 报告目标 Path，自动创建父目录。
        content: 要以 UTF-8 写入的完整报告文本；文件异常向上传播。
    '''
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
    '''将安全转义后的日志数据嵌入 HTML 统计模板。

    paras:
        data: 需要嵌入报告的运行统计数据。
    return: 可在浏览器查看的完整 HTML 文本。
    '''
    template = Path(__file__).with_name("log_report.html").read_text(encoding="utf-8")
    # JSON is embedded in a script element: escape markup delimiters, including
    # provider-supplied strings, so logs cannot terminate the element.
    payload = json.dumps(data, ensure_ascii=False).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    return template.replace("__REPORT_DATA__", payload)


def refresh_reports(logs_dir):
    '''根据结构化日志更新运行报告、离线查看器和索引，首次使用时包含旧日志。

    paras:
        logs_dir: 日志根目录，其 structured 子目录保存 JSONL 文件。
    return: 生成的 index.html 的 Path；写入失败时向上传播异常。
    '''
    logs_dir = Path(logs_dir)
    reports = logs_dir / "reports"
    atomic_write(logs_dir / 'viewer.html', render_report({'events': [], 'skipped': 0, 'run': '离线导入'}))
    rows = []
    sources = sorted((logs_dir / "structured").glob("*.jsonl"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    for source in sources:
        destination = reports / (source.stem + ".html")
        data = read_usage(source)
        if not destination.exists() or max(source.stat().st_mtime_ns,
                Path(__file__).stat().st_mtime_ns,
                Path(__file__).with_name('log_report.html').stat().st_mtime_ns) > destination.stat().st_mtime_ns:
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
<p><a href="viewer.html">导入原始 JSONL / 详情文件（无需重跑任务）</a></p>
<table><thead><tr><th>运行</th><th>任务数</th><th>模型调用数</th><th>明细状态</th></tr></thead><tbody>''' +
                 ("".join(rows) or '<tr><td colspan="4">暂无运行日志</td></tr>') + '</tbody></table></main></html>')
    return logs_dir / "index.html"

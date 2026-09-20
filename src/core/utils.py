"""通用工具函数 — 跨模块复用的零依赖辅助函数"""

import json
import re


def extract_json_array(text: str) -> list:
    """从 LLM 返回文本中提取第一个合法 JSON 数组。

    paras:
        text: 可能混有说明文字的 LLM 输出
    return: 解析到的数组；未找到返回空列表
    """
    decoder = json.JSONDecoder()
    for pos, ch in enumerate(text):
        if ch != "[":
            continue
        try:
            value, _ = decoder.raw_decode(text[pos:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            return value
    return []


def extract_python_list(text: str) -> list | None:
    """从 LLM 返回文本中提取 Python 字面量列表（支持 ```python 围栏）。

    paras:
        text: 可能混有说明文字的 LLM 输出
    return: 解析到的列表；解析失败返回 None
    """
    m = re.search(r"```(?:python)?\s*(\[[\s\S]*?\])\s*```", text)
    if m:
        raw = m.group(1)
    else:
        m = re.search(r"\[[\s\S]*\]", text)
        raw = m.group(0) if m else text
    try:
        import ast
        value = ast.literal_eval(raw)
        return value if isinstance(value, list) else None
    except (ValueError, SyntaxError):
        return None


def strip_control_chars(text: str, limit: int = 150) -> str:
    """清洗文本用作单行日志预览：去控制字符、压空白、截断。

    paras:
        text: 原始文本（如工具输出）
        limit: 最大保留字符数
    return: 可安全打印的单行文本；空结果返回 "(empty)"
    """
    cleaned = "".join(c for c in text[:limit] if c in "\n\t" or c >= " ")
    cleaned = cleaned.strip() or "(empty)"
    return cleaned.replace("\n", " ")

"""持久文献库 — reference_library 的 JSON 落盘/读取，跨任务累积，按 cite_key 去重合并"""

import json as json_mod
from pathlib import Path

from .paper import Paper, Source

LIBRARY_FILENAME = "reference_library.json"


def _library_file(dir_path: str) -> Path:
    """拼出库文件完整路径。

    paras:
        dir_path: 文献库目录
    return: 目录下 reference_library.json 的 Path
    """
    return Path(dir_path) / LIBRARY_FILENAME


def load_library(path: str) -> list[Paper]:
    """读取文献库 JSON 为 Paper 列表。

    paras:
        path: 文献库目录
    return: Paper 列表；文件不存在或损坏返回空列表
    """
    if not path:
        return []
    p = _library_file(path)
    if not p.exists():
        return []
    try:
        data = json_mod.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    papers: list[Paper] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            papers.append(_paper_from_dict(item))
        except Exception:
            continue
    return papers


def save_library(path: str, papers: list[Paper]) -> list[Paper]:
    """把 papers 合并进现有库（按 cite_key 去重，幂等）后写回。

    paras:
        path: 文献库目录
        papers: 要合入的 Paper 列表
    return: 合并后的完整 Paper 列表
    """
    existing = load_library(path)
    merged = merge_papers(existing, papers)
    p = _library_file(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = [paper.to_dict() for paper in merged]
    p.write_text(json_mod.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged


def merge_papers(*paper_lists) -> list[Paper]:
    """按 cite_key 去重合并多个 Paper 列表，保留先出现的条目。

    paras:
        paper_lists: 任意个 Paper 列表
    return: 去重后的 Paper 列表
    """
    seen: set[str] = set()
    merged: list[Paper] = []
    for papers in paper_lists:
        for paper in papers or []:
            if paper is None or not getattr(paper, "cite_key", ""):
                continue
            if paper.cite_key in seen:
                continue
            seen.add(paper.cite_key)
            merged.append(paper)
    return merged


def _paper_from_dict(d: dict) -> Paper:
    """从 dict 重建 Paper 对象。

    paras:
        d: Paper.to_dict 的输出
    return: Paper 对象（source 非法回退 LLM，year 非法回退 0）
    """
    src = d.get("source", "llm")
    try:
        source = Source(src)
    except ValueError:
        source = Source.LLM
    try:
        year = int(d.get("year", 0) or 0)
    except (TypeError, ValueError):
        year = 0
    return Paper(
        cite_key=d.get("cite_key", ""),
        title=d.get("title", ""),
        authors=d.get("authors", ""),
        year=year,
        source=source,
        abstract=d.get("abstract", ""),
        journal=d.get("journal", ""),
        volume=d.get("volume", ""),
        number=d.get("number", ""),
        pages=d.get("pages", ""),
        doi=d.get("doi", ""),
        issn=d.get("issn", ""),
        url=d.get("url", ""),
        month=d.get("month", ""),
        publisher=d.get("publisher", ""),
        arxiv_id=d.get("arxiv_id", ""),
        keywords=list(d.get("keywords") or []),
    )

"""持久文献库 — reference_library 的 JSON 落盘/读取，跨任务累积，按 cite_key 去重合并"""

import json as json_mod
import logging
from pathlib import Path
import os
import tempfile

from .paper import Paper, Source

LIBRARY_FILENAME = "reference_library.json"
logger = logging.getLogger(__name__)


def reference_snapshot(papers):
    '''生成不受条目顺序影响的完整文献快照，用于判断进展和持久化一致性。

    paras:
        papers: 结构化文献对象序列。
    return: 按引用键排序的文献字典列表。
    '''
    return sorted((paper.to_dict() for paper in papers), key=lambda item: item['cite_key'])


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
    except (OSError, UnicodeError, json_mod.JSONDecodeError) as exc:
        logger.warning("文献库读取失败 %s: %s", p, exc)
        return []
    if not isinstance(data, list):
        return []
    papers: list[Paper] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            papers.append(_paper_from_dict(item))
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("跳过无效文献记录: %s", exc)
            continue
    return papers


def save_library(path: str, papers: list[Paper], *, replace: bool = False) -> list[Paper]:
    """保存文献。默认传入记录覆盖同名旧记录；replace=True 保存完整快照（含删除）。

    paras:
        path: 文献库目录
        papers: 要合入的 Paper 列表
        replace: True 写入完整快照（允许删除旧记录）；False 合并旧库并覆盖同名记录。
    return: 合并后的完整 Paper 列表
    """
    existing = [] if replace else load_library(path)
    merged = merge_papers(papers, existing)
    p = _library_file(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = [paper.to_dict() for paper in merged]
    atomic_write(p, json_mod.dumps(data, ensure_ascii=False, indent=2))
    return merged


def atomic_write(path: Path, text: str):
    '''以 UTF-8 临时文件原子替换目标，自动创建父目录并清理临时文件。
    要么完整写入新内容，要么保留旧文件

    paras:
        path: 目标文件的 Path，存在时覆盖。
        text: 要写入的完整文本；写入或替换失败时向上传播文件异常。
    '''
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         delete=False) as stream:
            name = stream.name
            stream.write(text)
        os.replace(name, path)
    finally:
        if name and Path(name).exists():
            Path(name).unlink()


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
        entry_type=d.get("entry_type", "article"),
        bib_fields=dict(d.get("bib_fields") or {}),
        source_pdf=d.get("source_pdf", ""),
        source_fingerprint=d.get("source_fingerprint", ""),
        display_label=d.get("display_label", ""),
        metadata_missing=list(d.get("metadata_missing") or []),
        field_sources=dict(d.get("field_sources") or {}),
        completion_status=d.get("completion_status", ""),
    )

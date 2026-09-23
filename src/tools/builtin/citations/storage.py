import json as json_mod
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ...base import Tool
from ....domain.paper import Paper, Source
from ....core.llm import get_tool_llm
from .._safe_path import safe_resolve
from .._cite_key import make_cite_key
from .common import _escape_latex, _escape_bibtex_fields

class WriteBibFileTool(Tool):
    """将 BibTeX 条目去重合并写入 .bib 文件"""

    def __init__(self):
        super().__init__(
            name="write_bib_file",
            description="将多条 BibTeX 条目汇总写入指定的 .bib 文件。"
        )

    def get_parameters(self) -> dict:
        """返回工具参数的 JSON Schema 定义。

        return: input_schema 字典
        """
        return {
            "type": "object",
            "properties": {
                "bibtex_entries": {
                    "type": "array", "items": {"type": "string"},
                    "description": "BibTeX 条目列表"
                },
                "output_path": {"type": "string", "description": "输出 .bib 文件路径"},
            },
            "required": ["bibtex_entries", "output_path"],
        }

    def execute(self, bibtex_entries: list[str], output_path: str) -> str:
        """将 BibTeX 条目去重合并写入 .bib 文件。

        paras:
            bibtex_entries: BibTeX 条目列表，写入前对字段值做 LaTeX 转义防御（上游已转义不会重复转义）
            output_path: 输出 .bib 文件路径
        return: JSON 字符串，含 added/skipped/total 等统计；无条目返回警告字符串
        """
        if not bibtex_entries:
            return "警告: 没有任何 BibTeX 条目可写入"
        try:
            out_path = safe_resolve(output_path)
        except ValueError as e:
            return json_mod.dumps({"error": str(e)})
        out_path.parent.mkdir(parents=True, exist_ok=True)

        key_pattern = re.compile(r'@\w+\{([^,]+),')
        existing_keys: set[str] = set()
        existing_entries: list[str] = []
        if out_path.exists():
            content = out_path.read_text(encoding="utf-8")
            for m in key_pattern.finditer(content):
                existing_keys.add(m.group(1))
            existing_entries = [e.strip() for e in content.strip().split("\n\n") if e.strip()]

        new_entries: list[str] = []
        added: list[str] = []
        skipped: list[str] = []
        for entry in bibtex_entries:
            entry = _escape_bibtex_fields(entry)
            m = key_pattern.search(entry)
            if not m:
                new_entries.append(entry)
                continue
            cite_key = m.group(1)
            if cite_key in existing_keys:
                skipped.append(cite_key)
                continue
            existing_keys.add(cite_key)
            new_entries.append(entry)
            added.append(cite_key)

        if not new_entries:
            return json_mod.dumps({
                "status": "ok", "path": str(out_path),
                "added": 0, "skipped": len(skipped),
                "skipped_keys": skipped,
                "total": len(existing_entries),
            }, ensure_ascii=False)

        all_entries = existing_entries + new_entries
        content = "\n\n".join(all_entries) + "\n"
        out_path.write_text(content, encoding="utf-8")
        return json_mod.dumps({
            "status": "ok", "path": str(out_path),
            "added": len(added), "added_keys": added,
            "skipped": len(skipped), "skipped_keys": skipped,
            "total": len(all_entries),
        }, ensure_ascii=False)


class AddReferenceTool(Tool):
    """将论文元数据生成 Paper 实例并存入文献库（需注入 add_func）"""

    def __init__(self):
        super().__init__(
            name="add_reference",
            description="将一篇论文的元数据存入文献库，供后续引用检查和写作时查询。"
                        "输入论文的标题、作者、年份、摘要、cite_key 等字段。"
        )
        self._add_func = None
        self.provenance = None
        self.verifier = None

    def set_add_func(self, add_func):
        """注入入库回调（context.add_reference）。

        paras:
            add_func: 接收 Paper 对象的可调用对象
        """
        self._add_func = add_func

    def get_parameters(self) -> dict:
        """返回工具参数的 JSON Schema 定义。

        return: input_schema 字典
        """
        return {
            "type": "object",
            "properties": {
                "cite_key": {"type": "string", "description": "引用键"},
                "title": {"type": "string", "description": "论文标题"},
                "authors": {"type": "string", "description": "作者列表"},
                "year": {"type": "integer", "description": "发表年份"},
                "journal": {"type": "string", "description": "期刊名"},
                "volume": {"type": "string", "description": "卷号"},
                "number": {"type": "string", "description": "期号"},
                "pages": {"type": "string", "description": "页码范围"},
                "doi": {"type": "string", "description": "DOI"},
                "abstract": {"type": "string", "description": "论文摘要"},
                "keywords": {"type": "array", "items": {"type": "string"}, "description": "关键词列表"},
                "entry_type": {"type": "string", "description": "BibTeX类型，如article/book/inproceedings"},
                "publisher": {"type": "string"},
                "bib_fields": {"type": "object", "additionalProperties": {"type": "string"}, "description": "扩展字段，如booktitle/series"},
            },
            "required": ["cite_key", "title", "authors", "year"],
        }

    def execute(self, cite_key: str, title: str, authors: str, year: int,
                source: str = "llm", journal: str = "", volume: str = "",
                number: str = "", pages: str = "", doi: str = "",
                abstract: str = "", keywords: list | None = None,
                entry_type: str = "article", publisher: str = "", bib_fields: dict | None = None) -> str:
        """将论文元数据存入文献库。

        paras:
            cite_key: 引用键
            title: 论文标题
            authors: 作者列表
            year: 发表年份
            source: 文献来源 user/llm/online
            journal/volume/number/pages/doi: 可选字段
            abstract: 论文摘要
            keywords: 关键词列表
        return: JSON 字符串 {"status": "ok", "cite_key": ...}；
                未注入回调或 source 无效返回 {"error": ...}
        """
        if self._add_func is None:
            return json_mod.dumps({"error": "add_reference 工具未注入上下文"})
        try:
            source_enum = Source(source)
        except ValueError:
            return json_mod.dumps({"error": f"无效的 source 值: '{source}'，可选: user, llm, online"})
        ref = Paper(
            cite_key=cite_key, title=title, authors=authors, year=year,
            source=source_enum, journal=journal, volume=volume,
            number=number, pages=pages, doi=doi, abstract=abstract,
            keywords=keywords or [],
            entry_type=entry_type, publisher=publisher, bib_fields=bib_fields or {},
        )
        if self.provenance is None or self.verifier is None:
            return "Error: 文献来源门禁未初始化，拒绝入库"
        try:
            ref = self.provenance.admit(ref, self.verifier)
        except ValueError as exc:
            return f"Error: {exc}"
        self._add_func(ref)
        return json_mod.dumps({"status": "ok", "cite_key": cite_key})

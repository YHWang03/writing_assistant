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

class GenerateBibtexTool(Tool):
    """根据论文元数据生成标准 BibTeX 条目"""

    def __init__(self):
        super().__init__(
            name="generate_bibtex",
            description="根据论文元数据生成标准 BibTeX 条目。"
        )

    def get_parameters(self) -> dict:
        """返回工具参数的 JSON Schema 定义。

        return: input_schema 字典
        """
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "论文标题"},
                "authors": {"type": "string", "description": "作者列表，逗号分隔"},
                "year": {"type": "integer", "description": "发表年份"},
                "cite_key": {"type": "string", "description": "引用键"},
                "journal": {"type": "string", "description": "期刊名"},
                "volume": {"type": "string", "description": "卷号"},
                "number": {"type": "string", "description": "期号"},
                "pages": {"type": "string", "description": "页码范围"},
                "doi": {"type": "string", "description": "DOI"},
                "issn": {"type": "string", "description": "ISSN"},
                "url": {"type": "string", "description": "URL"},
                "month": {"type": "string", "description": "月份"},
                "publisher": {"type": "string", "description": "出版商"},
                "arxiv_id": {"type": "string", "description": "arXiv ID"},
                "entry_type": {"type": "string", "description": "条目类型", "default": "article"},
            },
            "required": ["title", "authors", "year"],
        }

    def execute(self, title: str, authors: str, year: int,
                cite_key: str = "", journal: str = "",
                volume: str = "", number: str = "", pages: str = "",
                doi: str = "", issn: str = "", url: str = "",
                month: str = "", publisher: str = "",
                arxiv_id: str = "", entry_type: str = "article") -> str:
        """根据论文元数据生成 BibTeX 条目。

        paras:
            title: 论文标题
            authors: 作者列表（逗号分隔）
            year: 发表年份
            cite_key: 引用键，缺省时自动生成
            journal/volume/number/pages/doi/issn/url/month/publisher: 可选字段
            arxiv_id: arXiv ID，提供时条目类型转为 misc
            entry_type: 条目类型，默认 article
        return: BibTeX 条目文本
        """
        if not cite_key:
            cite_key = make_cite_key(authors, year, title)
        if arxiv_id and entry_type == "article":
            entry_type = "misc"
        authors_formatted = " and ".join(
            a.strip() for a in authors.split(",") if a.strip()
        )
        lines = [f"@{entry_type}{{{cite_key},"]
        lines.append(f"  title = {{{_escape_latex(title)}}},")
        lines.append(f"  author = {{{authors_formatted}}},")
        lines.append(f"  year = {{{year}}},")
        if journal:
            lines.append(f"  journal = {{{_escape_latex(journal)}}},")
        if volume:
            lines.append(f"  volume = {{{_escape_latex(volume)}}},")
        if number:
            lines.append(f"  number = {{{_escape_latex(number)}}},")
        if pages:
            lines.append(f"  pages = {{{_escape_latex(pages)}}},")
        if doi:
            lines.append(f"  doi = {{{_escape_latex(doi)}}},")
        if issn:
            lines.append(f"  issn = {{{_escape_latex(issn)}}},")
        if url:
            lines.append(f"  url = {{{_escape_latex(url)}}},")
        if month:
            lines.append(f"  month = {{{_escape_latex(month)}}},")
        if publisher:
            lines.append(f"  publisher = {{{_escape_latex(publisher)}}},")
        if arxiv_id:
            lines.append(f"  eprint = {{{arxiv_id}}},")
            lines.append(f"  archiveprefix = {{arXiv}},")
        lines[-1] = lines[-1].rstrip(",")
        lines.append("}")
        return "\n".join(lines)


class SummarizePaperTool(Tool):
    """调用 LLM 总结论文核心贡献"""

    def __init__(self):
        super().__init__(
            name="summarize_paper",
            description="根据论文标题和摘要，用一句话总结其核心贡献。"
        )

    def get_parameters(self) -> dict:
        """返回工具参数的 JSON Schema 定义。

        return: input_schema 字典
        """
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "论文标题"},
                "abstract": {"type": "string", "description": "论文摘要"},
            },
            "required": ["title", "abstract"],
        }

    def execute(self, title: str, abstract: str) -> str:
        """用一句话总结论文核心贡献。

        paras:
            title: 论文标题
            abstract: 论文摘要
        return: 一句话总结；失败返回 "Error: ..." 字符串
        """
        prompt = (
            f"请用一句话（不多于50个英文单词）总结以下论文的核心贡献。\n\n"
            f"标题: {title}\n摘要: {abstract[:1000]}\n\n核心贡献（一句话）:"
        )
        try:
            return get_tool_llm().chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=128,
            ).strip()
        except Exception as e:
            return f"Error: LLM 调用失败 — {e}"

class GenerateBibFromRefLibTool(Tool):
    """从 reference_library 批量生成 .bib 文件"""

    def __init__(self):
        super().__init__(
            name="generate_bib_from_ref_library",
            description="从 reference_library 中读取所有文献条目，自动生成 BibTeX 并写入 .bib 文件。"
                        "以结构化库为准更新和删除旧条目，覆盖前保留备份。解析完 PDF 并将论文加入文献库后，"
                        "调用此工具一次性将所有条目写入 .bib 文件。"
        )
        self._reference_library: list[Paper] = []
        self._reference_provider = None

    def set_reference_provider(self, provider):
        """Query an authorized snapshot when the tool executes."""
        self._reference_provider = provider

    def set_persist_callback(self, callback):
        self._persist = callback

    def set_reference_library(self, refs: list[Paper]):
        """注入文献库列表。

        paras:
            refs: Paper 对象列表
        """
        self._reference_library = refs

    def get_parameters(self) -> dict:
        """返回工具参数的 JSON Schema 定义。

        return: input_schema 字典
        """
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "输出 .bib 文件路径"},
            },
            "required": ["path"],
        }

    def execute(self, path: str) -> str:
        """从完整文献库导出 BibTeX；替换旧版本前保留备份。

        paras:
            path: 输出 .bib 文件路径
        return: JSON 字符串，含 added/skipped/total 等统计；库为空返回 {"error": ...}
        """
        references = (self._reference_provider() if self._reference_provider is not None
                      else self._reference_library)
        try:
            out_path = safe_resolve(path)
        except ValueError as e:
            return json_mod.dumps({"error": str(e)})

        if out_path.suffix.lower() != ".bib":
            return "Error: 导出路径必须以 .bib 结尾"

        out_path.parent.mkdir(parents=True, exist_ok=True)

        from ....domain.bibliography import render_bib, parse_bib
        from ....domain.library import atomic_write
        from hashlib import sha256

        try:
            content = render_bib(references)
            old = out_path.read_text(encoding="utf-8") if out_path.exists() else ""
            # 留存每个不同版本，支持恢复旧的手工修正。
            if old and old != content:
                parse_bib(old)
                backup = out_path.with_name(out_path.name + "." + sha256(old.encode()).hexdigest()[:12] + ".bak")
                if not backup.exists():
                    atomic_write(backup, old)
            if getattr(self, "_persist", None):
                self._persist()
            atomic_write(out_path, content)
            old_keys = {key for _, key, _ in parse_bib(old)} if old else set()
            keys = {ref.cite_key for ref in references}
            return json_mod.dumps({"status": "ok", "path": str(out_path),
                                   "added_keys": sorted(keys - old_keys),
                                   "total": len(keys)}, ensure_ascii=False)
        except (ValueError, OSError) as exc:
            return f"Error: {exc}"

    def _paper_to_bibtex(self, ref: Paper) -> str:
        from ....domain.bibliography import render_bib
        return render_bib([ref]).strip()

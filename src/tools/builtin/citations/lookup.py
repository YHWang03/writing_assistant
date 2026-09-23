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

class ScanCitationsTool(Tool):
    """扫描 .tex 中的引用并返回上下文"""

    def __init__(self):
        super().__init__(
            name="scan_citations",
            description="扫描 LaTeX 文件中的 \\cite{...} 引用，提取引用上下文。"
                        "可指定 cite_key 过滤单个引用，不指定则返回全部。"
        )

    def get_parameters(self) -> dict:
        """返回工具参数的 JSON Schema 定义。

        return: input_schema 字典
        """
        return {
            "type": "object",
            "properties": {
                "tex_path": {"type": "string", "description": ".tex 文件路径"},
                "cite_key": {"type": "string", "description": "指定单个引用键，只返回该引用的上下文。不指定则返回全部"},
            },
            "required": ["tex_path"],
        }

    def execute(self, tex_path: str, cite_key: str = "") -> str:
        """扫描 .tex 中的引用（cite 及其变体），返回引用键与上下文。

        paras:
            tex_path: .tex 文件路径
            cite_key: 只返回该引用键的上下文；为空返回全部
        return: JSON 数组字符串，每项含 cite_key 与前后各 200 字符的 context；出错返回 {"error": ...}
        """
        try:
            path = safe_resolve(tex_path)
        except ValueError as e:
            return json_mod.dumps({"error": str(e)})
        if not path.exists():
            return json_mod.dumps({"error": f"文件不存在: {tex_path}"})
        content = path.read_text(encoding="utf-8", errors="replace")

        # \cite/\citep/\citet/\citeauthor/\citeyear 变体，支持可选参数 [..] 与多 key {k1,k2}
        cite_pattern = re.compile(
            r'\\(?:cite|citet|citep|citeauthor|citeyear)'
            r'(?:\s*\[[^\]]*\])*'
            r'\s*\{([^}]+)\}'
        )

        citations = []
        for match in cite_pattern.finditer(content):
            keys_str = match.group(1)
            keys = [k.strip() for k in keys_str.split(",") if k.strip()]
            start = match.start()
            end = match.end()
            ctx_start = max(0, start - 200)
            ctx_end = min(len(content), end + 200)
            context = content[ctx_start:ctx_end].replace("\n", " ").strip()
            for key in keys:
                if cite_key and key != cite_key:
                    continue
                citations.append({
                    "cite_key": key,
                    "context": context,
                })

        if cite_key and not citations:
            return json_mod.dumps({"error": f"未找到 cite_key: {cite_key}", "cite_key": cite_key})

        return json_mod.dumps(citations, ensure_ascii=False, indent=2)


class LookupPaperInfoTool(Tool):
    """从文献库按 cite_key 查询论文元数据（需先 set_reference_library 注入库）"""

    def __init__(self):
        super().__init__(
            name="lookup_paper_info",
            description="从文献库中查找指定论文的标题、摘要等元数据。输入 cite_key。"
        )
        self._reference_library: list[Paper] = []

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
                "cite_key": {"type": "string", "description": "论文的引用键 (cite_key)"},
            },
            "required": ["cite_key"],
        }

    def execute(self, cite_key: str) -> str:
        """按 cite_key 从文献库查询论文元数据。

        paras:
            cite_key: 论文引用键
        return: JSON 字符串（标题、摘要等）；未找到返回 {"error": ...}
        """
        for ref in self._reference_library:
            if ref.cite_key == cite_key:
                return json_mod.dumps({
                    "cite_key": cite_key,
                    "title": ref.title,
                    "authors": ref.authors,
                    "year": ref.year,
                    "abstract": ref.abstract[:1500],
                    "journal": ref.journal,
                    "volume": ref.volume,
                    "pages": ref.pages,
                    "doi": ref.doi,
                    "keywords": ref.keywords,
                    "source": ref.source.value,
                }, ensure_ascii=False)
        return json_mod.dumps({"error": f"文献库中未找到 cite_key '{cite_key}'。请使用 list_cite_keys 工具查看所有可用的引用键。"})

class ListCiteKeysTool(Tool):
    """返回文献库中所有文献的 cite_key 列表"""

    def __init__(self):
        super().__init__(
            name="list_cite_keys",
            description="返回 reference_library 中所有文献的 cite_key，每行一个。"
                        "当 lookup_paper_info 返回 cite_key 不存在时，使用此工具查看可用的引用键。"
        )
        self._reference_library: list[Paper] = []

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
            "properties": {},
            "required": [],
        }

    def execute(self) -> str:
        """列出文献库中所有文献的 cite_key 与标题。

        return: 每行一个 cite_key 的文本；库为空返回提示字符串
        """
        if not self._reference_library:
            return "reference_library 为空，暂无可用文献。"
        lines = [f"共 {len(self._reference_library)} 篇文献，可用 cite_key 如下："]
        for ref in self._reference_library:
            lines.append(f"  {ref.cite_key}  —  {ref.title}")
        return "\n".join(lines)

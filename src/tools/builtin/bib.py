"""BibTeX 生成与引用校验工具 — 条目生成、引用扫描/查询/比对/批量校验、.bib 写入、文献入库。"""

import re
import json as json_mod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from ..base import Tool
from ...core.paper import Paper, Source
from ...core.llm import get_tool_llm
from ._safe_path import safe_resolve
from ._cite_key import make_cite_key


_LATEX_ESCAPES = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\~{}", "^": r"\^{}",
    "\\": r"\textbackslash{}",
}

_LATEX_SPECIAL_RE = re.compile(r'[&%$#_{}~^\\]')


def _escape_latex(text: str) -> str:
    """转义 BibTeX/LaTeX 特殊字符，防止编译错误。

    paras:
        text: 原始文本
    return: 转义后的文本；空文本原样返回
    """
    if not text:
        return text
    # 单遍正则替换；顺序 replace 会造成二次转义（先 \$ 再转 \ 会破坏已转义的 \$）
    return _LATEX_SPECIAL_RE.sub(lambda m: _LATEX_ESCAPES[m.group(0)], text)


def _escape_bibtex_fields(entry: str) -> str:
    """对 BibTeX 条目中所有字段值做 LaTeX 转义，不破坏条目结构语法。

    paras:
        entry: BibTeX 条目文本
    return: 字段值转义后的条目文本
    """
    def _escape_field(match):
        # match 分组：1=缩进+字段名+" = {"，2=字段值，3=} 或 },
        indent = match.group(1)
        value = match.group(2)
        suffix = match.group(3)
        return f"{indent}{_escape_latex(value)}{suffix}"

    return re.sub(
        r'^(\s*\w+\s*=\s*\{)([^}]*)(\},?)$',
        _escape_field,
        entry,
        flags=re.MULTILINE,
    )


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


class CompareCitationTool(Tool):
    """比对引用描述与被引文献内容是否一致（支持单条与批量并行）"""

    def __init__(self):
        super().__init__(
            name="compare_citation",
            description="比对论文中引用某文献的描述与被引文献的原文内容是否一致。"
        )

    def get_parameters(self) -> dict:
        """返回工具参数的 JSON Schema 定义。

        return: input_schema 字典
        """
        return {
            "type": "object",
            "properties": {
                "cite_key": {"type": "string", "description": "引用键（单条模式）"},
                "citation_context": {"type": "string", "description": "引用上下文（单条模式）"},
                "paper_title": {"type": "string", "description": "被引论文标题（单条模式）"},
                "paper_abstract": {"type": "string", "description": "被引论文摘要（单条模式）"},
                "paper_conclusion": {"type": "string", "description": "被引论文结论（可选）"},
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "cite_key": {"type": "string"},
                            "citation_context": {"type": "string"},
                            "paper_title": {"type": "string"},
                            "paper_abstract": {"type": "string"},
                            "paper_conclusion": {"type": "string"},
                        },
                        "required": ["cite_key", "citation_context", "paper_title", "paper_abstract"],
                    },
                    "description": "批量模式：传入多个引用对象，并行比对。传此参数时忽略单条参数。"
                },
            },
            "required": ["cite_key", "citation_context", "paper_abstract"],
        }

    def execute(self, cite_key: str = "", citation_context: str = "",
                paper_title: str = "", paper_abstract: str = "",
                paper_conclusion: str = "", citations: list[dict] | None = None) -> str:
        """比对引用描述与被引文献内容。

        paras:
            cite_key: 引用键（单条模式）
            citation_context: 引用上下文（单条模式）
            paper_title: 被引论文标题（单条模式）
            paper_abstract: 被引论文摘要（单条模式）
            paper_conclusion: 被引论文结论（可选）
            citations: 批量模式的引用对象列表，传入时忽略单条参数
        return: 单条模式返回 LLM 比对文本；批量模式返回 JSON 数组
        """
        if citations:
            return self._execute_batch(citations)

        if not cite_key:
            return json_mod.dumps({"error": "请提供 cite_key 或 citations 参数"})
        return self._check_one(cite_key, citation_context, paper_title,
                               paper_abstract, paper_conclusion)

    def _check_one(self, cite_key: str, citation_context: str,
                   paper_title: str, paper_abstract: str,
                   paper_conclusion: str = "", as_json: bool = False):
        """比对单条引用（带重试）。

        paras:
            cite_key: 引用键
            citation_context: 引用上下文
            paper_title: 被引论文标题
            paper_abstract: 被引论文摘要
            paper_conclusion: 被引论文结论（可选）
            as_json: True 返回结构化 dict（失败返回 None），False 返回自然语言文本
        return: 比对结果文本或 dict；重试耗尽返回 None（as_json 时）或错误字符串
        """
        prompt = (
            "你是学术论文引用审查员。判断引用描述是否与被引文献一致。\n\n"
            f"## 被引文献\n标题: {paper_title}\n摘要: {paper_abstract[:1500]}\n"
        )
        if paper_conclusion:
            prompt += f"结论: {paper_conclusion[:1000]}\n"
        if as_json:
            prompt += (
                f"\n## 引用上下文\n引用键: \\cite{{{cite_key}}}\n{citation_context}\n\n"
                "## 输出格式\n只返回 JSON 对象："
                '{"verdict": "✅/⚠️/❌/❓", "reason": "具体理由", "suggestion": "修改建议（无则留空）"}'
                "不要返回其他内容。"
            )
        else:
            prompt += (
                f"\n## 引用上下文\n引用键: \\cite{{{cite_key}}}\n{citation_context}\n\n"
                "## 输出格式\nVerdict: [✅/⚠️/❌/❓]\nReason: [具体理由]\nSuggestion: [修改建议]"
            )
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = get_tool_llm().chat(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                )
                if result and result.strip():
                    if as_json:
                        return self._parse_verdict_json(result, cite_key)
                    return result
            except Exception as e:
                if attempt == max_retries - 1:
                    return None if as_json else f"Error: {e}"
        return None if as_json else json_mod.dumps({"error": "LLM 连续返回空结果，已重试2次仍失败"})

    @staticmethod
    def _parse_verdict_json(result: str, cite_key: str):
        """解析单条比对的 JSON 结果。

        paras:
            result: LLM 返回文本
            cite_key: 补充进结果的引用键
        return: 结构化 dict；解析失败返回 None
        """
        try:
            s = result.strip()
            if s.startswith("```"):
                lines = s.split("\n")
                s = "\n".join(lines[1:-1])
            obj = json_mod.loads(s)
            if isinstance(obj, dict):
                obj.setdefault("cite_key", cite_key)
                obj.setdefault("suggestion", "")
                return obj
            return None
        except Exception:
            return None

    def _execute_batch(self, citations: list[dict]) -> str:
        """并行比对多条引用。

        paras:
            citations: 引用对象列表
        return: JSON 数组字符串，每项含 cite_key 与 verdict
        """
        results: list[dict] = [None] * len(citations)

        def _check(idx: int, c: dict):
            return idx, self._check_one(
                cite_key=c.get("cite_key", ""),
                citation_context=c.get("citation_context", ""),
                paper_title=c.get("paper_title", ""),
                paper_abstract=c.get("paper_abstract", ""),
                paper_conclusion=c.get("paper_conclusion", ""),
            )

        with ThreadPoolExecutor(max_workers=min(len(citations), 10)) as executor:
            futures = {executor.submit(_check, i, c): i for i, c in enumerate(citations)}
            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    results[idx] = {
                        "cite_key": citations[idx].get("cite_key", ""),
                        "verdict": result,
                    }
                except Exception as e:
                    idx = futures[future]
                    results[idx] = {
                        "cite_key": citations[idx].get("cite_key", ""),
                        "verdict": f"Error: {e}",
                    }

        return json_mod.dumps(results, ensure_ascii=False, indent=2)


class ValidateAllCitationsTool(Tool):
    """合并 scan + lookup + compare 为单次引用校验工作流"""

    def __init__(self):
        super().__init__(
            name="validate_all_citations",
            description="一次性校验 .tex 文件中所有 \\cite 引用的准确性。"
                        "自动扫描引用、查询文献库、比对引用描述与被引文献内容。"
                        "对未入库的文献标记为 missing，不阻断流程。"
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
                "tex_path": {"type": "string", "description": ".tex 文件路径"},
            },
            "required": ["tex_path"],
        }

    def execute(self, tex_path: str) -> str:
        """校验 .tex 中所有引用：扫描 → 查文献库 → 批量比对。

        paras:
            tex_path: .tex 文件路径
        return: JSON 字符串，含 total_in_tex/validated/missing/failed/message；
                未入库文献标记 missing，不阻断流程
        """
        try:
            path = safe_resolve(tex_path)
        except ValueError as e:
            return json_mod.dumps({"error": str(e)})
        if not path.exists():
            return json_mod.dumps({"error": f"文件不存在: {tex_path}"})

        content = path.read_text(encoding="utf-8", errors="replace")
        cite_pattern = re.compile(
            r'\\(?:cite|citet|citep|citeauthor|citeyear)'
            r'(?:\s*\[[^\]]*\])*'
            r'\s*\{([^}]+)\}'
        )
        tex_citations: dict[str, str] = {}
        for match in cite_pattern.finditer(content):
            keys_str = match.group(1)
            keys = [k.strip() for k in keys_str.split(",") if k.strip()]
            ctx_start = max(0, match.start() - 200)
            ctx_end = min(len(content), match.end() + 200)
            context = content[ctx_start:ctx_end].replace("\n", " ").strip()
            for key in keys:
                tex_citations[key] = context

        if not tex_citations:
            return json_mod.dumps({
                "total_in_tex": 0,
                "validated": [],
                "missing": [],
                "failed": [],
                "message": ".tex 文件中未找到任何引用",
            }, ensure_ascii=False)

        library_map: dict[str, Paper] = {
            ref.cite_key: ref for ref in self._reference_library
        }

        tex_keys = set(tex_citations.keys())
        lib_keys = set(library_map.keys())

        to_validate_keys = tex_keys & lib_keys
        missing_keys = tex_keys - lib_keys

        validated = []
        missing = [
            {
                "cite_key": key,
                "reason": "该文献未入库，缺乏相关信息，需要 LiteratureAgent 添加相关信息并入库",
            }
            for key in sorted(missing_keys)
        ]
        failed = []

        if to_validate_keys:
            citations = []
            for key in sorted(to_validate_keys):
                ref = library_map[key]
                citations.append({
                    "cite_key": key,
                    "citation_context": tex_citations[key],
                    "paper_title": ref.title,
                    "paper_abstract": ref.abstract,
                })
            validated, failed = self._batch_compare(citations)

        return json_mod.dumps({
            "total_in_tex": len(tex_keys),
            "validated": validated,
            "missing": missing,
            "failed": failed,
            "message": (
                f"共 {len(tex_keys)} 条引用: "
                f"{len(validated)} 条已校验, "
                f"{len(missing)} 条未入库, "
                f"{len(failed)} 条校验失败"
            ),
        }, ensure_ascii=False, indent=2)

    def _batch_compare(self, citations: list[dict]) -> tuple[list[dict], list[dict]]:
        """单次 LLM 调用批量比对；解析失败降级到逐条比对。

        paras:
            citations: 引用对象列表
        return: (validated, failed) 元组
        """
        prompt = (
            "你是学术论文引用审查员。请逐一判断以下引用描述是否与被引文献一致。\n\n"
        )
        for i, c in enumerate(citations):
            prompt += (
                f"## 文献 {i+1}\n"
                f"引用键: \\cite{{{c['cite_key']}}}\n"
                f"被引文献标题: {c['paper_title']}\n"
                f"被引文献摘要: {c['paper_abstract'][:1500]}\n"
                f"引用上下文: {c['citation_context']}\n\n"
            )
        prompt += (
            "## 输出格式\n"
            "返回一个 JSON 数组，每个元素对应一条文献的校验结果：\n"
            '[{"cite_key": "...", "verdict": "✅/⚠️/❌/❓", "reason": "具体理由", "suggestion": "修改建议（无则留空）"}, ...]\n'
            "只返回 JSON 数组，不要其他内容。"
        )

        try:
            result = get_tool_llm().chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
            )
            result = result.strip()
            if result.startswith("```"):
                lines = result.split("\n")
                result = "\n".join(lines[1:-1])
            parsed = json_mod.loads(result)
            if isinstance(parsed, list):
                return parsed, []
            raise ValueError("LLM 返回非数组")
        except Exception:
            return self._compare_fallback(citations)

    def _compare_fallback(self, citations: list[dict]) -> tuple[list[dict], list[dict]]:
        """批量 JSON 失败时逐条降级比对（复用 _check_one，带重试，失败隔离）。

        paras:
            citations: 引用对象列表
        return: (validated, failed) 元组
        """
        comparator = CompareCitationTool()
        validated = []
        failed = []
        for c in citations:
            result = comparator._check_one(
                cite_key=c["cite_key"],
                citation_context=c["citation_context"],
                paper_title=c["paper_title"],
                paper_abstract=c["paper_abstract"],
                paper_conclusion=c.get("paper_conclusion", ""),
                as_json=True,
            )
            if isinstance(result, dict):
                validated.append(result)
            else:
                failed.append({
                    "cite_key": c["cite_key"],
                    "verdict": "❓",
                    "reason": "批量与逐条校验均失败",
                    "suggestion": "",
                })
        return validated, failed


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
                "source": {
                    "type": "string",
                    "enum": ["user", "llm", "online"],
                    "description": "文献来源：user=用户PDF解析，llm=LLM知识生成(未验证)，online=在线检索",
                },
                "journal": {"type": "string", "description": "期刊名"},
                "volume": {"type": "string", "description": "卷号"},
                "number": {"type": "string", "description": "期号"},
                "pages": {"type": "string", "description": "页码范围"},
                "doi": {"type": "string", "description": "DOI"},
                "abstract": {"type": "string", "description": "论文摘要"},
                "keywords": {"type": "array", "items": {"type": "string"}, "description": "关键词列表"},
            },
            "required": ["cite_key", "title", "authors", "year"],
        }

    def execute(self, cite_key: str, title: str, authors: str, year: int,
                source: str = "llm", journal: str = "", volume: str = "",
                number: str = "", pages: str = "", doi: str = "",
                abstract: str = "", keywords: list | None = None) -> str:
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
        )
        self._add_func(ref)
        return json_mod.dumps({"status": "ok", "cite_key": cite_key})


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


class GenerateBibFromRefLibTool(Tool):
    """从 reference_library 批量生成 .bib 文件"""

    def __init__(self):
        super().__init__(
            name="generate_bib_from_ref_library",
            description="从 reference_library 中读取所有文献条目，自动生成 BibTeX 并写入 .bib 文件。"
                        "自动检测已有条目避免重复添加。解析完 PDF 并将论文加入文献库后，"
                        "调用此工具一次性将所有条目写入 .bib 文件。"
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
                "path": {"type": "string", "description": "输出 .bib 文件路径"},
            },
            "required": ["path"],
        }

    def execute(self, path: str) -> str:
        """从文献库批量生成 BibTeX 并合并写入 .bib 文件（按 cite_key 去重，保留已有条目）。

        paras:
            path: 输出 .bib 文件路径
        return: JSON 字符串，含 added/skipped/total 等统计；库为空返回 {"error": ...}
        """
        if not self._reference_library:
            return json_mod.dumps({"error": "reference_library 为空，无条目可写入"})

        try:
            out_path = safe_resolve(path)
        except ValueError as e:
            return json_mod.dumps({"error": str(e)})

        out_path.parent.mkdir(parents=True, exist_ok=True)

        existing_keys: set[str] = set()
        existing_entries: list[str] = []
        if out_path.exists():
            content = out_path.read_text(encoding="utf-8")
            key_pattern = re.compile(r'@\w+\{([^,]+),')
            for m in key_pattern.finditer(content):
                existing_keys.add(m.group(1))
            existing_entries = content.strip().split("\n\n")
            existing_entries = [e.strip() for e in existing_entries if e.strip()]

        new_entries: list[str] = []
        skipped: list[str] = []
        added: list[str] = []

        for ref in self._reference_library:
            if not ref.cite_key:
                continue
            if ref.cite_key in existing_keys:
                skipped.append(ref.cite_key)
                continue
            bibtex = self._paper_to_bibtex(ref)
            new_entries.append(bibtex)
            added.append(ref.cite_key)

        if not new_entries and not existing_entries:
            return json_mod.dumps({"status": "ok", "path": str(out_path), "added": 0, "skipped": len(skipped)})

        all_entries = existing_entries + new_entries
        content = "\n\n".join(all_entries) + "\n"
        out_path.write_text(content, encoding="utf-8")

        return json_mod.dumps({
            "status": "ok",
            "path": str(out_path),
            "added": len(added),
            "added_keys": added,
            "skipped": len(skipped),
            "skipped_keys": skipped,
            "total": len(existing_entries) + len(new_entries),
        }, ensure_ascii=False)

    def _paper_to_bibtex(self, ref: Paper) -> str:
        """将 Paper 对象转为 BibTeX 字符串。

        paras:
            ref: Paper 对象
        return: BibTeX 条目文本
        """
        authors_formatted = " and ".join(
            a.strip() for a in ref.authors.split(",") if a.strip()
        )
        entry_type = ref.arxiv_id and "misc" or "article"
        lines = [f"@{entry_type}{{{ref.cite_key},"]
        lines.append(f"  title = {{{_escape_latex(ref.title)}}},")
        lines.append(f"  author = {{{authors_formatted}}},")
        lines.append(f"  year = {{{ref.year}}},")
        if ref.journal:
            lines.append(f"  journal = {{{_escape_latex(ref.journal)}}},")
        if ref.volume:
            lines.append(f"  volume = {{{_escape_latex(ref.volume)}}},")
        if ref.number:
            lines.append(f"  number = {{{_escape_latex(ref.number)}}},")
        if ref.pages:
            lines.append(f"  pages = {{{_escape_latex(ref.pages)}}},")
        if ref.doi:
            lines.append(f"  doi = {{{_escape_latex(ref.doi)}}},")
        if ref.issn:
            lines.append(f"  issn = {{{_escape_latex(ref.issn)}}},")
        if ref.url:
            lines.append(f"  url = {{{_escape_latex(ref.url)}}},")
        if ref.month:
            lines.append(f"  month = {{{_escape_latex(ref.month)}}},")
        if ref.publisher:
            lines.append(f"  publisher = {{{_escape_latex(ref.publisher)}}},")
        if ref.arxiv_id:
            lines.append(f"  eprint = {{{ref.arxiv_id}}},")
            lines.append(f"  archiveprefix = {{arXiv}},")
        lines[-1] = lines[-1].rstrip(",")
        lines.append("}")
        return "\n".join(lines)

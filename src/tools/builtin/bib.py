"""
BibTeX 与引用工具
- GenerateBibtexTool: 生成 BibTeX 条目
- SummarizePaperTool: 一句话总结论文贡献
- ScanCitationsTool: 扫描 .tex 中的引用
- LookupPaperInfoTool: 从文献库查询论文信息
- CompareCitationTool: 比对引用与原文
- ValidateAllCitationsTool: 合并 scan+lookup+compare 为单次工作流调用
- WriteBibFileTool: 写入 references.bib 文件
- AddReferenceTool: 将论文元数据存入文献库
- ListCiteKeysTool: 列出所有文献的 cite_key
"""

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
    r"""转义 BibTeX/LaTeX 特殊字符，防止编译错误。

    单遍正则替换：逐字符遍历并一次性替换，避免顺序 replace 造成的二次转义
    （先转 \$ 再转 \ 会破坏已转义的 \$；先转 \ 引入的 $ 又会被后续 \$ 污染）。"""
    if not text:
        return text
    return _LATEX_SPECIAL_RE.sub(lambda m: _LATEX_ESCAPES[m.group(0)], text)


def _escape_bibtex_fields(entry: str) -> str:
    """对 BibTeX 条目中所有字段值做 LaTeX 转义。
    匹配 "field = {value}," 模式，只转义字段值内部的特殊字符，
    不破坏 BibTeX 结构语法（如 @article{cite_key, 和 } 结尾）。"""
    def _escape_field(match):
        indent = match.group(1)      # 缩进 + field_name + " = {"
        value = match.group(2)       # 字段值
        suffix = match.group(3)      # } 或 },
        return f"{indent}{_escape_latex(value)}{suffix}"

    return re.sub(
        r'^(\s*\w+\s*=\s*\{)([^}]*)(\},?)$',
        _escape_field,
        entry,
        flags=re.MULTILINE,
    )


class GenerateBibtexTool(Tool):
    """
    根据论文元数据生成 BibTeX 条目
    return str 如下格式：
    @article{vidale1990finite,
     title = {Finite-difference calculation of traveltimes in three dimensions},
     author = {John E. Vidale},
     year = {1990},
     journal = {GEOPHYSICS},
     volume = {55},
     number = {5},
     pages = {521-526} 
    }    
    """

    def __init__(self):
        super().__init__(
            name="generate_bibtex",
            description="根据论文元数据生成标准 BibTeX 条目。"
        )

    def get_parameters(self) -> dict:
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
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "论文标题"},
                "abstract": {"type": "string", "description": "论文摘要"},
            },
            "required": ["title", "abstract"],
        }

    def execute(self, title: str, abstract: str) -> str:
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
    """
    扫描 .tex 中的引用
    找到所有包含 '\cite' 等变体 的引用并返回上下文
    """

    def __init__(self):
        super().__init__(
            name="scan_citations",
            description="扫描 LaTeX 文件中的 \\cite{...} 引用，提取引用上下文。"
                        "可指定 cite_key 过滤单个引用，不指定则返回全部。"
        )

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "tex_path": {"type": "string", "description": ".tex 文件路径"},
                "cite_key": {"type": "string", "description": "指定单个引用键，只返回该引用的上下文。不指定则返回全部"},
            },
            "required": ["tex_path"],
        }

    def execute(self, tex_path: str, cite_key: str = "") -> str:
        try:
            path = safe_resolve(tex_path)
        except ValueError as e:
            return json_mod.dumps({"error": str(e)})
        if not path.exists():
            return json_mod.dumps({"error": f"文件不存在: {tex_path}"})
        content = path.read_text(encoding="utf-8", errors="replace")

        # 匹配所有常见引用变体: \cite, \citep, \citet, \citeauthor, \citeyear
        # 支持可选参数: \citep[page 3]{key} 或 \citet[p.42]{key}
        cite_pattern = re.compile(
            r'\\(?:cite|citet|citep|citeauthor|citeyear)'
            r'(?:\s*\[[^\]]*\])*'   # 零个或多个可选参数 [...]
            r'\s*\{([^}]+)\}'        # 必选参数 {key1,key2,...}
        )

        citations = []
        for match in cite_pattern.finditer(content):
            keys_str = match.group(1)
            keys = [k.strip() for k in keys_str.split(",") if k.strip()]
            start = match.start()
            end = match.end()
            # 提取上下文：前后各 200 字符
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
    """
    从文献库中查询指定论文的详细信息
    接受cite_key参数，返回该论文的标题、摘要等元数据
    使用工具前需先将context中引用文献信息 同步至 工具
    """

    def __init__(self):
        super().__init__(
            name="lookup_paper_info",
            description="从文献库中查找指定论文的标题、摘要等元数据。输入 cite_key。"
        )
        self._reference_library: list[Paper] = []

    def set_reference_library(self, refs: list[Paper]):
        self._reference_library = refs

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "cite_key": {"type": "string", "description": "论文的引用键 (cite_key)"},
            },
            "required": ["cite_key"],
        }

    def execute(self, cite_key: str) -> str:
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
    """
    比对引用与原文
    接受 被引论文摘要，cite_key所在上下文，返回比对结果
    包含 是否一致， 原因描述， 修改建议
    """

    def __init__(self):
        super().__init__(
            name="compare_citation",
            description="比对论文中引用某文献的描述与被引文献的原文内容是否一致。"
        )

    def get_parameters(self) -> dict:
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

        # 批量模式：并行处理
        if citations:
            return self._execute_batch(citations)

        # 单条模式
        if not cite_key:
            return json_mod.dumps({"error": "请提供 cite_key 或 citations 参数"})
        return self._check_one(cite_key, citation_context, paper_title,
                               paper_abstract, paper_conclusion)

    def _check_one(self, cite_key: str, citation_context: str,
                   paper_title: str, paper_abstract: str,
                   paper_conclusion: str = "", as_json: bool = False):
        """比对单条引用（带重试）。
        as_json=False 返回自然语言文本（默认，兼容现有调用）；
        as_json=True 返回结构化 dict {cite_key, verdict, reason, suggestion}，失败返回 None。"""
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
        """解析单条比对的 JSON 结果；解析失败返回 None"""
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
        """并行比对多条引用"""
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
    """合并 scan_citations + list_cite_keys + lookup + compare_citation 为一个工作流"""

    def __init__(self):
        super().__init__(
            name="validate_all_citations",
            description="一次性校验 .tex 文件中所有 \\cite 引用的准确性。"
                        "自动扫描引用、查询文献库、比对引用描述与被引文献内容。"
                        "对未入库的文献标记为 missing，不阻断流程。"
        )
        self._reference_library: list[Paper] = []

    def set_reference_library(self, refs: list[Paper]):
        self._reference_library = refs

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "tex_path": {"type": "string", "description": ".tex 文件路径"},
            },
            "required": ["tex_path"],
        }

    def execute(self, tex_path: str) -> str:

        # ---- Step 1: scan citations from .tex ----
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
        tex_citations: dict[str, str] = {}  # cite_key → context
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

        # ---- Step 2: build library cite_key → Paper map ----
        library_map: dict[str, Paper] = {
            ref.cite_key: ref for ref in self._reference_library
        }

        tex_keys = set(tex_citations.keys())
        lib_keys = set(library_map.keys())

        # ---- Step 3: classify ----
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

        # ---- Step 4: batch compare via LLM ----
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
        """批量比对引用，返回 (validated, failed)。
        快速路径：单次 LLM 调用返回 JSON 数组；解析失败时降级到逐条比对（失败隔离）。"""
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
            # 降级：逐条比对，失败隔离，不让整批一起挂
            return self._compare_fallback(citations)

    def _compare_fallback(self, citations: list[dict]) -> tuple[list[dict], list[dict]]:
        """批量 JSON 失败时逐条降级，复用 CompareCitationTool._check_one（带重试）"""
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
    """将 BibTeX 条目写入文件"""

    def __init__(self):
        super().__init__(
            name="write_bib_file",
            description="将多条 BibTeX 条目汇总写入指定的 .bib 文件。"
        )

    def get_parameters(self) -> dict:
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
        if not bibtex_entries:
            return "警告: 没有任何 BibTeX 条目可写入"
        try:
            out_path = safe_resolve(output_path)
        except ValueError as e:
            return json_mod.dumps({"error": str(e)})
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # 解析已有 .bib 文件中的 cite_key，避免重复
        key_pattern = re.compile(r'@\w+\{([^,]+),')
        existing_keys: set[str] = set()
        existing_entries: list[str] = []
        if out_path.exists():
            content = out_path.read_text(encoding="utf-8")
            for m in key_pattern.finditer(content):
                existing_keys.add(m.group(1))
            existing_entries = [e.strip() for e in content.strip().split("\n\n") if e.strip()]

        # 解析新条目的 cite_key，只添加不重复的
        new_entries: list[str] = []
        added: list[str] = []
        skipped: list[str] = []
        for entry in bibtex_entries:
            # 对 BibTeX 字段值做 LaTeX 转义（防御层，即使上游已转义也不会重复）
            entry = _escape_bibtex_fields(entry)
            m = key_pattern.search(entry)
            if not m:
                # 无法解析 cite_key，直接添加
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
    """
    将论文元数据添加到文献库
    接收cite_key, title等参数，生成paper实例，添加到文献库
    """

    def __init__(self):
        super().__init__(
            name="add_reference",
            description="将一篇论文的元数据存入文献库，供后续引用检查和写作时查询。"
                        "输入论文的标题、作者、年份、摘要、cite_key 等字段。"
        )
        self._add_func = None  # 注入 context.add_reference

    def set_add_func(self, add_func):
        self._add_func = add_func

    def get_parameters(self) -> dict:
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
    """返回 reference_library 中所有文献的 cite_key 列表"""

    def __init__(self):
        super().__init__(
            name="list_cite_keys",
            description="返回 reference_library 中所有文献的 cite_key，每行一个。"
                        "当 lookup_paper_info 返回 cite_key 不存在时，使用此工具查看可用的引用键。"
        )
        self._reference_library: list[Paper] = []

    def set_reference_library(self, refs: list[Paper]):
        self._reference_library = refs

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def execute(self) -> str:
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
        self._reference_library = refs

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "输出 .bib 文件路径"},
            },
            "required": ["path"],
        }

    def execute(self, path: str) -> str:
        if not self._reference_library:
            return json_mod.dumps({"error": "reference_library 为空，无条目可写入"})

        try:
            out_path = safe_resolve(path)
        except ValueError as e:
            return json_mod.dumps({"error": str(e)})

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # 解析已有 .bib 文件中的 cite_key，避免重复
        existing_keys: set[str] = set()
        existing_entries: list[str] = []
        if out_path.exists():
            content = out_path.read_text(encoding="utf-8")
            key_pattern = re.compile(r'@\w+\{([^,]+),')
            for m in key_pattern.finditer(content):
                existing_keys.add(m.group(1))
            # 保留已有条目内容
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

        # 合并已有条目和新条目
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
        """将 Paper 对象转为 BibTeX 字符串"""
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
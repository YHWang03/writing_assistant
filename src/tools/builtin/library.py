"""文献库工具（library-first 流程）
- ListPaperFilesTool      : 列出 seed/refs 目录的 PDF，从文件名解析 year + title_query
- FindRelevantPapersTool  : 从持久文献库检索与主题相关的文献，并加入 reference_library
- WriteLibraryTool        : 把 reference_library 合并写回持久文献库 JSON
"""

import json as json_mod
import re
from pathlib import Path

from ..base import Tool
from ...core.paper import Paper
from ...core.library import load_library, save_library
from ...core.llm import get_tool_llm


# ---- 文件名解析 ----

_UNICODE_HYPHENS = ("‐", "‑", "–", "—", "‒", "―")


def parse_paper_filename(filename: str) -> dict:
    """从论文 PDF 文件名解析出可靠的元数据。

    只提取可靠信息：开头 4 位年份 + 其余部分（_/- → 空格）作为标题候选串。
    作者边界（单作者/双作者）不可靠，不在此处切分，交给联网检索解析。
    """
    stem = Path(filename).stem
    for ch in _UNICODE_HYPHENS:
        stem = stem.replace(ch, "-")
    year = 0
    rest = stem
    m = re.match(r"^(\d{4})[-_]", stem)
    if m:
        year = int(m.group(1))
        rest = stem[m.end():]
    title_query = re.sub(r"[-_]+", " ", rest).strip()
    return {"year": year, "title_query": title_query}


class ListPaperFilesTool(Tool):
    """列出所有论文 PDF，并从文件名解析 year + title_query（不做 PDF 解析）。"""

    def __init__(self):
        super().__init__(
            name="list_paper_files",
            description="列出所有可用的论文 PDF（seed 目录 + reference 目录），"
                        "并从文件名解析出 year 和 title_query（用于后续联网检索）。"
                        "不解析 PDF 内容。",
        )
        self._pdf_paths: list[str] = []

    def set_pdf_paths(self, seed_paths, ref_paths):
        self._pdf_paths = list(seed_paths or []) + list(ref_paths or [])

    def get_parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    def execute(self) -> str:
        if not self._pdf_paths:
            return json_mod.dumps({
                "total": 0, "papers": [],
                "note": "未注入 PDF 路径（seed/refs 目录可能为空）",
            }, ensure_ascii=False)
        papers = []
        for p in self._pdf_paths:
            info = parse_paper_filename(Path(p).name)
            papers.append({"path": p, **info})
        return json_mod.dumps(
            {"total": len(papers), "papers": papers},
            ensure_ascii=False, indent=2,
        )


class FindRelevantPapersTool(Tool):
    """
    从持久文献库检索与主题相关的文献，并自动加入 reference_library
    输入参数topic及library_path, 提取所有文献的title,keywords,abstract(截断)字段
    交由llm判断相关性，对相关文献进行入库
    """

    def __init__(self):
        super().__init__(
            name="find_relevant_papers",
            description="从持久文献库中检索与当前论文主题相关的文献，并把相关文献加入 "
                        "reference_library。返回找到的相关文献数量，以及是否达到 min_relevant 阈值。"
                        "若 found 达到 min_relevant（sufficient=true），则无需再解析 PDF 或联网搜索。",
        )
        self._library_path = ""
        self._add_func = None
        self._min_relevant = 15

    def set_library_path(self, path: str):
        self._library_path = path

    def set_add_func(self, add_func):
        self._add_func = add_func

    def set_min_relevant(self, n: int):
        self._min_relevant = int(n)

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "我们要写的论文主题/创新点描述（用于相关性匹配）",
                },
                "min_relevant": {
                    "type": "integer",
                    "description": "最少需要的相关文献数",
                    "default": 15,
                },
            },
            "required": ["topic"],
        }

    def execute(self, topic: str, min_relevant: int | None = None) -> str:
        if self._add_func is None:
            return json_mod.dumps({"error": "find_relevant_papers 未注入 add_reference"})
        threshold = int(min_relevant or self._min_relevant)
        candidates = load_library(self._library_path) if self._library_path else []
        if not candidates:
            return json_mod.dumps({
                "found": 0, "sufficient": False,
                "min_relevant": threshold, "total_in_library": 0,
                "note": "持久文献库为空或不存在，请用 list_paper_files + search_papers 补充文献。",
            }, ensure_ascii=False)

        relevant = self._match_relevant(topic, candidates)
        for paper in relevant:
            self._add_func(paper)

        return json_mod.dumps({
            "found": len(relevant),
            "sufficient": len(relevant) >= threshold,
            "min_relevant": threshold,
            "total_in_library": len(candidates),
            "papers": [
                {"cite_key": p.cite_key, "title": p.title, "year": p.year}
                for p in relevant
            ],
        }, ensure_ascii=False)

    # ---- 相关性匹配 ----

    def _match_relevant(self, topic: str, candidates: list[Paper]) -> list[Paper]:
        """用 flash LLM 做主题相关性筛选；失败则降级为关键词重叠。"""
        lines = []
        for i, p in enumerate(candidates):
            abstract = (p.abstract or "").replace("\n", " ")[:400]
            kw = ", ".join(p.keywords or [])[:120]
            lines.append(f"[{i}] {p.title} ({p.year}) | {kw} | {abstract}")
        prompt = (
            "你是学术文献筛选助手。给定一篇待写论文的主题，从候选文献中"
            "筛选出「相关」文献（主题接近、可被本论文引用）。\n\n"
            f"## 待写论文主题\n{topic}\n\n"
            "## 候选文献（格式：[序号] 标题 (年份) | 关键词 | 摘要）\n"
            + "\n".join(lines)
            + "\n\n## 输出要求\n"
            "只返回一个 JSON 数组，包含所有相关文献的序号（整数），例如 [0,3,7]；"
            "无相关则返回 []。不要输出其他内容。"
        )
        try:
            raw = get_tool_llm().chat(
                messages=[{"role": "user", "content": prompt}], max_tokens=2048,
            ).strip()
            idxs = self._parse_index_list(raw)
            return [candidates[i] for i in idxs if 0 <= i < len(candidates)]
        except Exception:
            return self._keyword_match(topic, candidates)

    @staticmethod
    def _parse_index_list(raw: str) -> list[int]:
        # 防止llm返回结果包含其余无关信息
        # 仅保留方括号中的内容，如 [0,3,7]
        m = re.search(r"\[[\s\S]*\]", raw)
        if m:
            raw = m.group(0)
        data = json_mod.loads(raw)
        if not isinstance(data, list):
            raise ValueError("not a list")
        out: list[int] = []
        for x in data:
            if isinstance(x, int):
                out.append(x)
            elif isinstance(x, str) and x.strip().isdigit():
                out.append(int(x.strip()))
        return out

    @staticmethod
    def _keyword_match(topic: str, candidates: list[Paper]) -> list[Paper]:
        """降级：主题与标题/摘要/关键词的 token 重叠评分。"""
        topic_tokens = set(re.findall(r"[a-zA-Z0-9]+", topic.lower()))
        scored: list[tuple[int, Paper]] = []
        for p in candidates:
            text = f"{p.title} {p.abstract} {' '.join(p.keywords or [])}".lower()
            tokens = set(re.findall(r"[a-zA-Z0-9]+", text))
            overlap = len(topic_tokens & tokens)
            if overlap > 0:
                scored.append((overlap, p))
        scored.sort(key=lambda x: -x[0])
        return [p for _, p in scored]


class WriteLibraryTool(Tool):
    """把 reference_library 合并写回持久文献库 JSON。"""

    def __init__(self):
        super().__init__(
            name="write_library",
            description="把 reference_library 中的文献合并写入持久文献库 JSON（按 cite_key 去重），"
                        "使文献库跨任务累积。应在写 .bib 之前调用。",
        )
        self._library_path = ""
        self._reference_library: list[Paper] = []

    def set_library_path(self, path: str):
        self._library_path = path

    def set_reference_library(self, refs: list[Paper]):
        self._reference_library = refs

    def get_parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    def execute(self) -> str:
        if not self._library_path:
            return json_mod.dumps({"error": "write_library 未注入文献库路径"})
        before_keys = {p.cite_key for p in load_library(self._library_path) if p.cite_key}
        merged = save_library(self._library_path, self._reference_library)
        added = sum(1 for p in merged if p.cite_key and p.cite_key not in before_keys)
        return json_mod.dumps({
            "status": "ok",
            "path": self._library_path,
            "total": len(merged),
            "added": added,
        }, ensure_ascii=False)

"""PDF 解析工具 — 元数据提取（单篇/批量并行）与原文获取。"""

import html
import json as json_mod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from ..base import Tool
from ...core.llm import get_tool_llm
from ._cite_key import make_cite_key


class ParsePDFTool(Tool):
    """解析 PDF 提取元数据，支持单篇与批量并行"""

    def __init__(self):
        super().__init__(
            name="parse_pdf",
            description="解析 PDF 论文，提取标题、作者、摘要、正文等信息。"
                        "支持单篇（pdf_path）和批量并行（pdf_paths 传入列表）。"
                        "批量模式下，多个 PDF 并发处理，大幅提升效率。"
        )

    def get_parameters(self) -> dict:
        """返回工具参数的 JSON Schema 定义。

        return: input_schema 字典
        """
        return {
            "type": "object",
            "properties": {
                "pdf_path": {
                    "type": "string",
                    "description": "PDF 文件路径（单篇）。与 pdf_paths 二选一",
                },
                "pdf_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "PDF 文件路径列表（批量并行）。与 pdf_path 二选一，传入后并发处理",
                },
                "max_workers": {
                    "type": "integer",
                    "description": "并行处理的最大线程数（默认 4）。仅在 pdf_paths 模式下生效",
                    "default": 4,
                },
            },
            "required": [],
        }

    def execute(self, pdf_path: str = "", pdf_paths: list[str] | None = None,
                max_workers: int = 4) -> str:
        """解析 PDF 提取元数据。

        paras:
            pdf_path: 单篇模式文件路径
            pdf_paths: 批量模式路径列表
            max_workers: 批量模式最大线程数
        return: JSON 字符串；单篇 LLM 解析失败时返回 fallback 指示（回退 get_paper_text）
        """
        if pdf_path:
            return self._parse_single(pdf_path)

        if pdf_paths:
            return self._parse_batch(pdf_paths, max_workers)

        return json_mod.dumps({"error": "请提供 pdf_path 或 pdf_paths 参数"})

    def _parse_single(self, pdf_path: str) -> str:
        """解析单篇 PDF，LLM 解析失败时返回 fallback 指示。

        paras:
            pdf_path: PDF 文件路径
        return: JSON 字符串（元数据或 fallback 指示）
        """
        path = Path(pdf_path)
        if not path.exists():
            return json_mod.dumps({"error": f"文件不存在: {pdf_path}", "file": pdf_path})

        try:
            import fitz
            doc = fitz.open(str(path))
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            doc.close()
        except ImportError:
            return json_mod.dumps({"error": "PyMuPDF 未安装", "file": pdf_path})
        except Exception as e:
            return json_mod.dumps({"error": f"PDF 解析失败: {e}", "file": pdf_path})

        if not full_text.strip():
            return json_mod.dumps({"error": "PDF 内容为空", "file": pdf_path})

        result = self._extract_with_llm(full_text[:3000], pdf_path)
        try:
            parsed = json_mod.loads(result)
            if "error" in parsed:
                return json_mod.dumps({
                    "fallback": "get_paper_text",
                    "message": "parse_pdf 的 LLM 解析失败，已自动 fallback 到 get_paper_text",
                    "text": full_text[:50000],
                    "file": pdf_path,
                }, ensure_ascii=False)
            parsed["file"] = pdf_path
            return json_mod.dumps(parsed, ensure_ascii=False)
        except json_mod.JSONDecodeError:
            return json_mod.dumps({
                "fallback": "get_paper_text",
                "message": "parse_pdf 的 LLM 解析失败（返回非 JSON），已自动 fallback 到 get_paper_text",
                "text": full_text[:50000],
                "file": pdf_path,
            }, ensure_ascii=False)

    def _parse_batch(self, pdf_paths: list[str], max_workers: int) -> str:
        """并行解析多篇 PDF。

        paras:
            pdf_paths: PDF 路径列表
            max_workers: 最大线程数
        return: JSON 字符串，含 total/success/failed/results/errors
        """
        results = []
        errors = []
        workers = min(max_workers, len(pdf_paths))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._parse_one_worker, p): p
                for p in pdf_paths
            }
            for future in as_completed(futures):
                pdf_path = futures[future]
                try:
                    result = future.result()
                    if "error" in result:
                        errors.append(result)
                    else:
                        results.append(result)
                except Exception as e:
                    errors.append({"error": str(e), "file": pdf_path})

        output = {
            "total": len(pdf_paths),
            "success": len(results),
            "failed": len(errors),
            "results": results,
        }
        if errors:
            output["errors"] = errors

        return json_mod.dumps(output, ensure_ascii=False, indent=2)

    def _parse_one_worker(self, pdf_path: str) -> dict:
        """单 worker 解析一篇 PDF（提取文本 + LLM 提取元数据）。

        paras:
            pdf_path: PDF 文件路径
        return: 元数据 dict；失败返回含 error 的 dict
        """
        path = Path(pdf_path)
        if not path.exists():
            return {"error": f"文件不存在: {pdf_path}", "file": pdf_path}

        try:
            import fitz
            doc = fitz.open(str(path))
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            doc.close()
        except ImportError:
            return {"error": "PyMuPDF 未安装", "file": pdf_path}
        except Exception as e:
            return {"error": f"PDF 解析失败: {e}", "file": pdf_path}

        if not full_text.strip():
            return {"error": "PDF 内容为空", "file": pdf_path}

        result = self._extract_with_llm(full_text[:3000], pdf_path)
        try:
            parsed = json_mod.loads(result)
            parsed["file"] = pdf_path
            return parsed
        except json_mod.JSONDecodeError:
            return {"error": "LLM 返回非 JSON", "file": pdf_path, "raw": result[:200]}

    def _extract_with_llm(self, text: str, pdf_path: str = "") -> str:
        """用 LLM 从 PDF 文本中提取论文元数据。

        paras:
            text: PDF 文本（前 3000 字符）
            pdf_path: PDF 文件路径（用于错误报告）
        return: JSON 字符串；失败返回 {"error": ...}
        """
        prompt = (
            "从以下 PDF 文本中提取论文元数据，以 JSON 格式返回。提取不到的字段设为空字符串。\n\n"
            "返回格式（只返回 JSON，不要其他内容）:\n"
            '{"title": "", "authors": "", "year": 0, "journal": "", "volume": "", '
            '"number": "", "pages": "", "doi": "", "abstract": "", "keywords": []}\n\n'
            f"PDF 文本:\n{text}"
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
            # LLM 可能输出 HTML 实体（如 &amp;），解码还原为原始字符
            result = html.unescape(result)
            return result
        except Exception as e:
            return json_mod.dumps({"error": f"LLM 提取失败: {e}", "file": pdf_path})


class ParseAndStoreTool(Tool):
    """解析 PDF 并自动入库，返回简要摘要而不返回完整元数据"""

    def __init__(self):
        super().__init__(
            name="parse_and_store",
            description="解析 PDF 论文并自动将元数据存入文献库（reference_library）。"
                        "支持单篇（pdf_path）和批量（pdf_paths）。"
                        "返回简要摘要（入库数、失败数），不返回完整元数据以节省上下文。"
        )
        self._parse_tool = ParsePDFTool()
        self._add_func = None
        # 已成功入库的 PDF 路径（跨 dispatch 保留）；失败不记录以允许重试
        self._attempted_files: set[str] = set()

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
                "pdf_path": {
                    "type": "string",
                    "description": "PDF 文件路径（单篇）。与 pdf_paths 二选一",
                },
                "pdf_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "PDF 文件路径列表（批量并行）。与 pdf_path 二选一",
                },
                "max_workers": {
                    "type": "integer",
                    "description": "并行处理的最大线程数（默认 4）",
                    "default": 4,
                },
            },
            "required": [],
        }

    def execute(self, pdf_path: str = "", pdf_paths: list[str] | None = None,
                max_workers: int = 4) -> str:
        """解析 PDF 并将元数据入库（幂等：已成功入库的文件跳过）。

        paras:
            pdf_path: 单篇模式文件路径
            pdf_paths: 批量模式路径列表
            max_workers: 批量模式最大线程数
        return: JSON 字符串，含 stored/failed/skipped/failures 等统计；
                必备字段（title/authors/year）缺失时不入库，只计入失败报告
        """
        if self._add_func is None:
            return json_mod.dumps({"error": "parse_and_store 工具未注入上下文"})

        files = [pdf_path] if pdf_path else list(pdf_paths or [])
        new_files = [f for f in files if f not in self._attempted_files]
        skipped = [f for f in files if f in self._attempted_files]

        if not new_files:
            return json_mod.dumps({
                "stored": 0, "failed": 0, "skipped": len(skipped),
                "skipped_files": skipped,
                "note": "所有传入文件均已处理过，已跳过（避免重复解析）。请直接调用 generate_bib_from_ref_library 生成 .bib。",
            }, ensure_ascii=False)

        raw = self._parse_tool.execute(
            pdf_path=(new_files[0] if pdf_path else ""),
            pdf_paths=(new_files if pdf_paths else None),
            max_workers=max_workers,
        )

        try:
            data = json_mod.loads(raw)
        except json_mod.JSONDecodeError:
            return json_mod.dumps({"error": "parse_pdf 返回非 JSON", "raw": raw[:500]})

        if isinstance(data, dict) and "results" in data:
            papers = data.get("results", [])
            errors = data.get("errors", [])
        elif isinstance(data, dict) and "title" in data:
            papers = [data]
            errors = []
        elif isinstance(data, dict) and "fallback" in data:
            return json_mod.dumps({
                "stored": 0, "failed": 1,
                "failed_files": [data.get("file", "")],
                "reason": "LLM 解析失败，fallback 到原始文本，未入库",
            }, ensure_ascii=False)
        elif isinstance(data, dict) and "error" in data:
            return json_mod.dumps({
                "stored": 0, "failed": 1,
                "failed_files": [data.get("file", "")],
                "reason": data["error"],
            }, ensure_ascii=False)
        else:
            papers = []
            errors = []

        stored_keys = []
        failed_files = []
        failures = []
        for p in papers:
            missing = self._missing_fields(p)
            if missing:
                file = p.get("file", "unknown")
                failed_files.append(file)
                failures.append({
                    "file": file,
                    "reason": "元数据不完整",
                    "missing": missing,
                    "have": {
                        "title": p.get("title", ""),
                        "authors": p.get("authors", ""),
                        "year": p.get("year", 0) or 0,
                        "doi": p.get("doi", ""),
                    },
                })
                continue
            cite_key = make_cite_key(
                p.get("authors", ""), p.get("year", 0) or 0, p.get("title", "")
            )
            try:
                from ...core.paper import Paper, Source
                ref = Paper(
                    cite_key=cite_key,
                    title=p.get("title", ""),
                    authors=p.get("authors", ""),
                    year=p.get("year", 0) or 0,
                    source=Source.USER,
                    abstract=p.get("abstract", ""),
                    journal=p.get("journal", ""),
                    volume=p.get("volume", ""),
                    number=p.get("number", ""),
                    pages=p.get("pages", ""),
                    doi=p.get("doi", ""),
                    keywords=p.get("keywords", []),
                )
                self._add_func(ref)
                stored_keys.append(cite_key)
                if p.get("file"):
                    self._attempted_files.add(p["file"])
            except Exception as e:
                failed_files.append(p.get("file", cite_key or "unknown"))
                failures.append({
                    "file": p.get("file", cite_key or "unknown"),
                    "reason": f"入库异常: {e}",
                })

        for e in errors:
            file = e.get("file", "unknown")
            failed_files.append(file)
            failures.append({
                "file": file,
                "reason": e.get("error", "解析失败"),
            })

        output = {
            "stored": len(stored_keys),
            "stored_keys": stored_keys,
            "failed": len(failed_files),
            "failed_files": failed_files,
        }
        if skipped:
            output["skipped"] = len(skipped)
            output["skipped_files"] = skipped
        if failures:
            output["failures"] = failures
        return json_mod.dumps(output, ensure_ascii=False)

    @staticmethod
    def _missing_fields(p: dict) -> list[str]:
        """检测必备字段（title/authors/year），返回缺失字段列表。

        paras:
            p: 单篇解析结果 dict
        return: 缺失字段名列表；journal 不作为必备字段（预印本首页常无期刊名，缺失不应丢弃整条文献）
        """
        return [f for f in ("title", "authors", "year") if not p.get(f)]


class GetPaperTextTool(Tool):
    """获取 PDF 论文的纯文本"""

    def __init__(self):
        super().__init__(
            name="get_paper_text",
            description="获取 PDF 论文的原始文本内容。"
        )

    def get_parameters(self) -> dict:
        """返回工具参数的 JSON Schema 定义。

        return: input_schema 字典
        """
        return {
            "type": "object",
            "properties": {
                "pdf_path": {"type": "string", "description": "PDF 文件路径"},
                "start_page": {"type": "integer", "description": "起始页", "default": 1},
                "end_page": {"type": "integer", "description": "结束页（-1=末页）", "default": -1},
                "max_chars": {"type": "integer", "description": "最大返回字符数", "default": 50000},
            },
            "required": ["pdf_path"],
        }

    def execute(self, pdf_path: str, start_page: int = 1,
                end_page: int = -1, max_chars: int = 50000) -> str:
        """获取 PDF 指定页范围的纯文本（带分页标记）。

        paras:
            pdf_path: PDF 文件路径
            start_page: 起始页（从 1 起）
            end_page: 结束页，-1 表示末页
            max_chars: 最大返回字符数
        return: 分页文本，超长截断；失败返回 "Error: ..." 字符串
        """
        path = Path(pdf_path)
        if not path.exists():
            return f"Error: 文件不存在: {pdf_path}"
        try:
            import fitz
            doc = fitz.open(str(path))
            pages = []
            total = doc.page_count
            if end_page < 0:
                end_page = total
            start_page = max(1, start_page)
            end_page = min(total, end_page)
            for i in range(start_page - 1, end_page):
                pages.append(f"\n--- Page {i+1} ---\n{doc[i].get_text()}")
            doc.close()
            result = "\n".join(pages)
        except ImportError:
            return "Error: PyMuPDF 未安装"
        except Exception as e:
            return f"Error: {e}"

        if len(result) > max_chars:
            result = result[:max_chars] + "\n\n... (已截断)"
        return result

"""PDF 解析工具 — 元数据提取（单篇/批量并行）与原文获取。"""

import html
import logging
import json as json_mod
from concurrent.futures import ThreadPoolExecutor, as_completed
from ...observability.telemetry import submit_with_context
from pathlib import Path
from ..base import Tool
from ...core.llm import get_tool_llm
from ._cite_key import make_cite_key
from ...context.pdf_cache import PDFCache
from ...core.llm import IncompleteResponseError
from anthropic import APIConnectionError, APIStatusError


class ParsePDFTool(Tool):
    """解析 PDF 提取元数据，支持单篇与批量并行"""

    def __init__(self, cache_dir=None):
        super().__init__(
            name="parse_pdf",
            description="解析 PDF 论文，提取标题、作者、摘要、正文等信息。"
                        "支持单篇（pdf_path）和批量并行（pdf_paths 传入列表）。"
                        "批量模式下，多个 PDF 并发处理，大幅提升效率。"
        )
        self._cache = PDFCache(cache_dir)

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
        return json_mod.dumps(self._parse_one_worker(pdf_path), ensure_ascii=False)

    def _parse_batch(self, pdf_paths: list[str], max_workers: int) -> str:
        """并行解析多篇 PDF。

        paras:
            pdf_paths: PDF 路径列表
            max_workers: 最大线程数
        return: JSON 字符串，含 total/success/failed/results/errors
        """
        results = []
        errors = []
        workers = max(1, min(max_workers, len(pdf_paths), 8))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                submit_with_context(executor, self._parse_one_worker, p): p
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
        """单篇/批量共用：指纹缓存、失败分类、最多两次临时错误尝试。"""
        path = Path(pdf_path)
        try:
            content = path.read_bytes()
        except OSError as exc:
            return {"status": "failed", "error": str(exc), "error_kind": "file_unavailable",
                    "file": pdf_path, "attempts": 0, "cache_hit": False, "retryable": False}
        key = self._cache.key(content)
        with self._cache.lock(key):
            cached = self._cache.read(key)
            if cached and cached.get("status") == "success" and ParseAndStoreTool._missing_fields(cached):
                cached = None
            if cached:
                cached.update(file=pdf_path, cache_hit=True)
                self._log_result(cached)
                return cached
            result = {"fingerprint": key, "file": pdf_path, "status": "failed",
                      "cache_hit": False, "attempts": 0, "retryable": False}
            try:
                import fitz
                with fitz.open(stream=content, filetype="pdf") as doc:
                    if doc.needs_pass:
                        raise ValueError("PDF 已加密，需要密码")
                    full_text = "".join(page.get_text() for page in doc)
            except Exception as exc:
                result.update(error=f"PDF 读取失败: {exc}", error_kind="pdf_unreadable")
            else:
                if not full_text.strip():
                    result.update(error="PDF 无可提取文本，可能需要 OCR", error_kind="empty_text")
                else:
                    for attempt in range(1, 3):
                        result["attempts"] = attempt
                        extracted = self._extract_with_llm(full_text[:3000], pdf_path)
                        try:
                            parsed = json_mod.loads(extracted)
                            if not isinstance(parsed, dict):
                                raise ValueError("元数据必须为对象")
                            if "error" in parsed:
                                result.update(error=parsed["error"],
                                              error_kind=parsed.get("error_kind", "llm_error"),
                                              retryable=parsed.get("retryable", False))
                                if result["retryable"] and attempt < 2:
                                    continue
                                break
                            missing = ParseAndStoreTool._missing_fields(parsed)
                            if missing:
                                raise ValueError("缺失或无效字段: " + ", ".join(missing))
                            # 只接受书目字段，避免模型覆盖状态/指纹。
                            names = ("title", "authors", "year", "journal", "volume",
                                     "number", "pages", "doi", "abstract", "keywords")
                            result.update({name: parsed[name] for name in names if name in parsed})
                            result.pop("error", None)
                            result.pop("error_kind", None)
                            result.update(status="success", retryable=False)
                        except (ValueError, TypeError) as exc:
                            result.update(error=f"元数据解析失败: {exc}",
                                          error_kind="invalid_metadata", retryable=False)
                        break
            if result["status"] == "failed":
                result["retry_exhausted"] = bool(result["retryable"])
                result["next_action"] = "本进程不再重复解析；请补充元数据或修复问题后重启。"
            try:
                self._cache.write(key, result)
            except OSError as exc:
                result["cache_warning"] = f"缓存保存失败: {exc}"
            self._log_result(result)
            return result

    @staticmethod
    def _log_result(result):
        fields = {k: result[k] for k in ("file", "fingerprint", "status", "attempts",
                  "cache_hit", "error_kind", "retry_exhausted") if k in result}
        logging.getLogger(__name__).info("PDF parse %s", fields,
                                        extra={"event": "pdf_parse", **fields})

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
            # JSON 解析后再解码实体，避免 &quot; 破坏 JSON 字符串边界。
            parsed = json_mod.loads(result)
            if isinstance(parsed, dict):
                parsed = {k: html.unescape(v) if isinstance(v, str) else v for k, v in parsed.items()}
            result = json_mod.dumps(parsed, ensure_ascii=False)
            return result
        except IncompleteResponseError as e:
            return json_mod.dumps({"error": str(e), "error_kind": "incomplete_output", "retryable": False})
        except Exception as e:
            transient = isinstance(e, APIConnectionError) or (
                isinstance(e, APIStatusError) and (e.status_code in (408, 409, 429) or e.status_code >= 500))
            return json_mod.dumps({"error": f"LLM 提取失败: {e}", "file": pdf_path,
                                   "error_kind": "transient_api" if transient else "extraction_failed",
                                   "retryable": transient})


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
        new_files = list(dict.fromkeys(files))
        allowed = getattr(self, "allowed_pdf_paths", None)
        if allowed is not None and any(str(Path(p).resolve()) not in allowed for p in new_files):
            return "Error: PDF不在用户配置的输入列表中，不能作为用户来源导入"
        skipped = []

        if not new_files:
            return json_mod.dumps({
                "stored": 0, "failed": 0, "skipped": len(skipped),
                "skipped_files": skipped,
                "note": "请提供 pdf_path 或 pdf_paths。",
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
            papers, errors = [], [data]
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
                from ...domain.paper import Paper, Source
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
                    source_pdf=str(Path(p.get("file", "")).resolve()),
                    source_fingerprint=p.get("fingerprint", ""),
                )
                from ...domain.reference_provenance import title_key
                ref.provenance_kind = "pdf"
                ref.provenance_title = title_key(ref.title)
                provenance = getattr(self, "provenance", None)
                if provenance is not None:
                    provenance.record(ref.title, "pdf")
                self._add_func(ref)
                stored_keys.append(cite_key)
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
                **{k: e[k] for k in ("error_kind", "attempts", "cache_hit", "retryable",
                                      "retry_exhausted", "next_action") if k in e},
                "file": file,
                "reason": e.get("error", "解析失败"),
            })

        output = {
            "cached": sum(bool(p.get("cache_hit")) for p in papers + errors),
            "files": [{k: p[k] for k in ("file", "status", "fingerprint", "attempts",
                       "cache_hit", "error_kind", "retry_exhausted", "cache_warning") if k in p}
                      for p in papers + errors],
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
        missing = [f for f in ("title", "authors")
                   if not isinstance(p.get(f), str) or not p[f].strip()]
        year = p.get("year")
        if not isinstance(year, int) or isinstance(year, bool) or year <= 0:
            missing.append("year")
        for key in ("journal", "volume", "number", "pages", "doi", "abstract"):
            if key in p and not isinstance(p[key], str):
                missing.append(key)
        if "keywords" in p and (not isinstance(p["keywords"], list)
                               or not all(isinstance(k, str) for k in p["keywords"])):
            missing.append("keywords")
        return missing


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

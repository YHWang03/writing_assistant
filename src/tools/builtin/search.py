"""文献搜索工具 — OpenAlex / Semantic Scholar / arXiv 在线搜索与论文验证。

网络统一走 _http_get（指数退避 + 全抖动 + Retry-After + 分主机限流），结果落盘缓存；
auto 后端按 OpenAlex → Semantic Scholar → arXiv 依次回退。
"""

import json as json_mod
import os
import random
import re
import time
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from ..base import Tool

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 5
_BASE_BACKOFF = 1.0
_MAX_BACKOFF = 30.0
_TIMEOUT = 20

# 分主机限流的最小请求间隔（秒）：OpenAlex 礼貌池 ~10 req/s，S2 匿名 1 req/s，arXiv 官方要求 ≥3s
_HOST_MIN_INTERVAL = {
    "api.openalex.org": 0.15,
    "api.semanticscholar.org": 1.1,
    "export.arxiv.org": 3.0,
}
_host_last_request: dict[str, float] = {}

_search_cache: dict | None = None
_search_cache_path: Path | None = None


class _NetworkError(Exception):
    """网络请求重试耗尽，携带最后一次底层异常。"""


class _SearchError(Exception):
    """搜索失败（含可读的 reason / hint），供上层转成结构化返回。"""

    def __init__(self, reason: str, hint: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.hint = hint


def _s2_api_key() -> str:
    """读取 Semantic Scholar API key（.env 由 main.py 加载到 os.environ）。

    return: API key 字符串，未配置返回空串
    """
    return os.getenv("S2_API_KEY", "")


def _s2_headers() -> dict:
    """构造 Semantic Scholar 请求头（有 API key 时附带）。

    return: 请求头 dict
    """
    headers = {"User-Agent": "WritingAssistant/1.0"}
    key = _s2_api_key()
    if key:
        headers["x-api-key"] = key
    return headers


def _openalex_headers() -> dict:
    """构造 OpenAlex 请求头（User-Agent 带 mailto，可用 .env 的 OPENALEX_MAILTO 配置）。

    return: 请求头 dict
    """
    mailto = os.getenv("OPENALEX_MAILTO", "")
    ua = "WritingAssistant/1.0"
    if mailto:
        ua += f" (mailto:{mailto})"
    return {"User-Agent": ua}


def _reconstruct_abstract(inv: dict | None) -> str:
    """还原 OpenAlex 的倒排索引摘要（{词: [位置]}）为原文。

    paras:
        inv: 倒排索引 dict
    return: 摘要文本；无索引返回空串
    """
    if not inv:
        return ""
    pos_to_word = {}
    for word, positions in inv.items():
        for p in positions:
            pos_to_word[p] = word
    return " ".join(pos_to_word[i] for i in sorted(pos_to_word))


def _arxiv_id_from_doi(doi: str) -> str:
    """从 DOI 提取 arxiv id（OpenAlex 对 arXiv 论文的 DOI 形如 10.48550/arXiv.1706.03762）。

    paras:
        doi: DOI 字符串
    return: arxiv id（去掉版本号后缀）；无法提取返回空串
    """
    if not doi:
        return ""
    m = re.search(r"10\.48550/arxiv\.(\S+)", doi, re.IGNORECASE)
    if not m:
        return ""
    return re.sub(r"v\d+$", "", m.group(1))


def _get_cache_path() -> Path:
    """返回搜索缓存文件路径（沿用 s2_cache.json 文件名以保留历史缓存）。

    return: 缓存文件 Path
    """
    global _search_cache_path
    if _search_cache_path is None:
        _search_cache_path = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "example" / "output" / "s2_cache.json"
        )
    return _search_cache_path


def _load_cache() -> dict:
    """加载搜索缓存（首次调用时从磁盘读取）。

    return: 缓存 dict
    """
    global _search_cache
    if _search_cache is None:
        path = _get_cache_path()
        if path.exists():
            try:
                _search_cache = json_mod.loads(path.read_text(encoding="utf-8"))
            except Exception:
                _search_cache = {}
        else:
            _search_cache = {}
    return _search_cache


def _save_cache(cache: dict):
    """保存搜索缓存到磁盘。

    paras:
        cache: 缓存 dict
    """
    path = _get_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_mod.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_cache_key(text: str) -> str:
    """归一化缓存键。

    paras:
        text: 原始查询文本
    return: 去首尾空白、转小写、截断到 250 字符的键
    """
    return text.strip().lower()[:250]


def _throttle(host: str):
    """限流：确保同一主机两次请求至少间隔配置的最小时间（S2 有 key 时 0.1s）。

    paras:
        host: 主机名
    """
    if host == "api.semanticscholar.org" and _s2_api_key():
        min_interval = 0.1
    else:
        min_interval = _HOST_MIN_INTERVAL.get(host, 1.0)
    now = time.time()
    last = _host_last_request.get(host, 0.0)
    wait = min_interval - (now - last)
    if wait > 0:
        time.sleep(wait)
    _host_last_request[host] = time.time()


def _retry_delay(attempt: int, retry_after: str | None = None) -> float:
    """计算重试等待时间：指数退避 + 全抖动，响应带 Retry-After 时优先尊重。

    paras:
        attempt: 当前重试轮次（从 0 起）
        retry_after: 服务端返回的 Retry-After 值
    return: 等待秒数
    """
    if retry_after:
        try:
            return min(float(retry_after), _MAX_BACKOFF)
        except ValueError:
            pass
    return min(_BASE_BACKOFF * (2 ** attempt), _MAX_BACKOFF) * random.uniform(0.2, 1.0)


def _http_get(url: str, host: str, headers: dict | None = None,
              timeout: int = _TIMEOUT, max_retries: int = _MAX_RETRIES) -> str:
    """带指数退避 + 全抖动 + Retry-After 的 GET 请求。

    paras:
        url: 请求地址
        host: 主机名（用于限流）
        headers: 请求头
        timeout: 单次请求超时秒数
        max_retries: 最大重试次数
    return: 解码后的响应文本；失败抛 _NetworkError
    """
    headers = dict(headers or {})
    headers.setdefault("User-Agent", "WritingAssistant/1.0")
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        _throttle(host)
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code not in _RETRYABLE_STATUS:
                raise _NetworkError(f"HTTP {e.code} {e.reason}") from e
            last_exc = e
            time.sleep(_retry_delay(attempt, e.headers.get("Retry-After")))
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_exc = e
            time.sleep(_retry_delay(attempt))
    raise _NetworkError(f"连续 {max_retries} 次失败: {last_exc}") from last_exc


def _ok(backend: str, results: list, note: str = "") -> str:
    """构造成功返回。

    paras:
        backend: 后端名
        results: 结果列表
        note: 附加说明
    return: JSON 字符串 {"ok": True, ...}
    """
    return json_mod.dumps({
        "ok": True, "backend": backend, "results": results, "note": note,
    }, ensure_ascii=False, indent=2)


def _err(backend: str, error: str, hint: str = "") -> str:
    """构造失败返回。

    paras:
        backend: 后端名
        error: 错误信息
        hint: 处理建议
    return: JSON 字符串 {"ok": False, ...}
    """
    return json_mod.dumps({
        "ok": False, "backend": backend, "error": error, "hint": hint,
    }, ensure_ascii=False)


class SearchPapersTool(Tool):
    """在线搜索论文（多后端，带缓存与限流）"""

    def __init__(self):
        super().__init__(
            name="search_papers",
            description="在线搜索论文。返回 JSON：{\"ok\": true, \"backend\": ..., \"results\": [...]} "
                        "或 {\"ok\": false, \"error\": ..., \"hint\": ...}。"
                        "backend 支持 auto（默认，按 OpenAlex → Semantic Scholar → arXiv 依次回退）、"
                        "openalex、semantic_scholar、arxiv。"
        )
        self.max_searches = 8
        self._search_count = 0

    def reset(self):
        """重置搜索计数（每个任务开始时调用，防止跨 dispatch 累积）。"""
        self._search_count = 0

    def get_parameters(self) -> dict:
        """返回工具参数的 JSON Schema 定义。

        return: input_schema 字典
        """
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "max_results": {"type": "integer", "description": "最大返回结果数", "default": 5},
                "backend": {
                    "type": "string",
                    "description": "搜索后端: auto(默认)/openalex/semantic_scholar/arxiv",
                    "default": "auto",
                },
            },
            "required": ["query"],
        }

    def execute(self, query: str, max_results: int = 5,
                backend: str = "auto") -> str:
        """在线搜索论文。

        paras:
            query: 搜索关键词
            max_results: 最大返回结果数
            backend: auto/openalex/semantic_scholar/arxiv
        return: JSON 字符串 {"ok": True, "results": [...]} 或 {"ok": False, "error": ...}；
                超过搜索次数上限返回带 stop 标记的错误
        """
        self._search_count += 1
        if self._search_count > self.max_searches:
            return json_mod.dumps({
                "ok": False, "backend": backend, "stop": True,
                "error": f"搜索次数已达上限（{self.max_searches} 次），已停止联网检索。",
                "hint": "改用 get_paper_text 读 PDF 页眉获取元数据，或把无法确定的文献标记为 unresolved 上报；不要再调用 search_papers。",
            }, ensure_ascii=False)
        try:
            if backend == "arxiv":
                return self._search_arxiv(query, max_results)
            if backend == "semantic_scholar":
                return self._search_semantic_scholar(query, max_results)
            if backend == "openalex":
                return self._search_openalex(query, max_results)
            if backend == "auto":
                return self._search_auto(query, max_results)
            return _err(backend, error=f"未知后端 '{backend}'",
                        hint="可选 auto / openalex / semantic_scholar / arxiv")
        except _SearchError as e:
            return _err(backend, error=e.reason, hint=e.hint)
        except Exception as e:
            return _err(backend, error=f"搜索失败: {e}")

    def _search_auto(self, query: str, max_results: int) -> str:
        """按 OpenAlex → Semantic Scholar → arXiv 依次尝试，失败自动回退。

        paras:
            query: 搜索关键词
            max_results: 最大返回结果数
        return: JSON 字符串；全部失败返回 {"ok": False, ...}
        """
        attempts = [
            ("openalex", self._search_openalex),
            ("semantic_scholar", self._search_semantic_scholar),
            ("arxiv", self._search_arxiv),
        ]
        failed = []
        for name, fn in attempts:
            try:
                obj = json_mod.loads(fn(query, max_results))
            except _SearchError as e:
                failed.append(f"{name}({e.reason})")
                continue
            except Exception as e:
                failed.append(f"{name}({e})")
                continue
            if failed:
                note = f"{' → '.join(f.split('(')[0] for f in failed)} 失败，已回退到 {name}"
                if obj.get("note"):
                    note += f"；{obj['note']}"
                obj["note"] = note
            return json_mod.dumps(obj, ensure_ascii=False, indent=2)
        return _err("auto", error="；".join(failed),
                    hint="稍后重试，或检查网络 / 设置 S2_API_KEY")

    def _search_semantic_scholar(self, query: str, max_results: int) -> str:
        """用 Semantic Scholar 搜索（带缓存）。

        paras:
            query: 搜索关键词
            max_results: 最大返回结果数
        return: JSON 字符串；失败抛 _SearchError
        """
        cache = _load_cache()
        cache_key = f"search:{_normalize_cache_key(query)}"
        if cache_key in cache:
            try:
                papers = json_mod.loads(cache[cache_key])
            except Exception:
                papers = []
            return _ok("semantic_scholar", papers, note="cache hit")

        try:
            encoded = urllib.parse.quote(query)
            url = (
                f"https://api.semanticscholar.org/graph/v1/paper/search?"
                f"query={encoded}&limit={max_results}"
                f"&fields=title,authors,year,abstract,externalIds,url"
            )
            data = json_mod.loads(_http_get(url, "api.semanticscholar.org",
                                            headers=_s2_headers()))
            papers = []
            for item in data.get("data", []):
                authors = [a.get("name", "") for a in item.get("authors", [])]
                papers.append({
                    "title": item.get("title", ""),
                    "authors": ", ".join(authors),
                    "year": item.get("year", 0),
                    "abstract": item.get("abstract", "") or "",
                    "doi": item.get("externalIds", {}).get("DOI", ""),
                    "arxiv_id": item.get("externalIds", {}).get("ArXiv", ""),
                    "url": item.get("url", ""),
                    "source": "semantic_scholar",
                })
            cache[cache_key] = json_mod.dumps(papers, ensure_ascii=False)
            _save_cache(cache)
            return _ok("semantic_scholar", papers)
        except _NetworkError as e:
            raise _SearchError(
                reason=f"Semantic Scholar 请求失败: {e}",
                hint="可换用 backend=arxiv，或稍后重试",
            ) from e
        except Exception as e:
            raise _SearchError(reason=f"Semantic Scholar 解析失败: {e}") from e

    def _search_openalex(self, query: str, max_results: int) -> str:
        """用 OpenAlex 搜索（带缓存）。

        paras:
            query: 搜索关键词
            max_results: 最大返回结果数
        return: JSON 字符串；失败抛 _SearchError
        """
        cache = _load_cache()
        cache_key = f"openalex:{_normalize_cache_key(query)}"
        if cache_key in cache:
            try:
                papers = json_mod.loads(cache[cache_key])
            except Exception:
                papers = []
            return _ok("openalex", papers, note="cache hit")

        try:
            encoded = urllib.parse.quote(query)
            url = (
                f"https://api.openalex.org/works?"
                f"search={encoded}&per-page={max_results}"
            )
            data = json_mod.loads(_http_get(url, "api.openalex.org",
                                            headers=_openalex_headers()))
            papers = []
            for item in data.get("results", []):
                authors = [a.get("author", {}).get("display_name", "")
                           for a in item.get("authorships", [])]
                doi = item.get("doi") or ""
                if doi.startswith("https://doi.org/"):
                    doi = doi[len("https://doi.org/"):]
                papers.append({
                    "title": item.get("display_name", ""),
                    "authors": ", ".join(a for a in authors if a),
                    "year": item.get("publication_year", 0),
                    "abstract": _reconstruct_abstract(item.get("abstract_inverted_index")),
                    "doi": doi,
                    "arxiv_id": _arxiv_id_from_doi(doi),
                    "url": f"https://doi.org/{doi}" if doi else item.get("id", ""),
                    "source": "openalex",
                })
            cache[cache_key] = json_mod.dumps(papers, ensure_ascii=False)
            _save_cache(cache)
            return _ok("openalex", papers)
        except _NetworkError as e:
            raise _SearchError(
                reason=f"OpenAlex 请求失败: {e}",
                hint="可换用 backend=arxiv，或稍后重试",
            ) from e
        except Exception as e:
            raise _SearchError(reason=f"OpenAlex 解析失败: {e}") from e

    def _search_arxiv(self, query: str, max_results: int) -> str:
        """用 arXiv 搜索（带缓存，解析 Atom XML）。

        paras:
            query: 搜索关键词
            max_results: 最大返回结果数
        return: JSON 字符串；失败抛 _SearchError
        """
        import xml.etree.ElementTree as ET

        cache = _load_cache()
        cache_key = f"arxiv:{_normalize_cache_key(query)}"
        if cache_key in cache:
            try:
                papers = json_mod.loads(cache[cache_key])
            except Exception:
                papers = []
            return _ok("arxiv", papers, note="cache hit")

        try:
            encoded = urllib.parse.quote(query)
            url = (
                f"https://export.arxiv.org/api/query?"
                f"search_query=all:{encoded}&start=0&max_results={max_results}"
            )
            raw = _http_get(url, "export.arxiv.org")
            root = ET.fromstring(raw)
            ns = {"atom": "http://www.w3.org/2005/Atom",
                  "arxiv": "http://arxiv.org/schemas/atom"}

            papers = []
            for entry in root.findall("atom:entry", ns):
                title = entry.find("atom:title", ns)
                title_text = title.text.strip() if title is not None else ""
                authors = []
                for author in entry.findall("atom:author", ns):
                    name = author.find("atom:name", ns)
                    if name is not None:
                        authors.append(name.text.strip())
                summary = entry.find("atom:summary", ns)
                summary_text = summary.text.strip() if summary is not None else ""
                arxiv_id = ""
                for link in entry.findall("atom:link", ns):
                    href = link.get("href", "")
                    if "/abs/" in href:
                        arxiv_id = href.split("/abs/")[-1]
                        break
                    if "/pdf/" in href:
                        arxiv_id = href.split("/pdf/")[-1]
                        break
                # 去掉版本号后缀 v1/v2，得到规范 arxiv id（如 2105.02723）
                arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
                published = entry.find("atom:published", ns)
                year = int(published.text[:4]) if published is not None else 0
                papers.append({
                    "title": title_text, "authors": ", ".join(authors),
                    "year": year, "abstract": summary_text,
                    "arxiv_id": arxiv_id, "source": "arxiv",
                })

            cache[cache_key] = json_mod.dumps(papers, ensure_ascii=False)
            _save_cache(cache)
            return _ok("arxiv", papers)
        except _NetworkError as e:
            raise _SearchError(
                reason=f"arXiv 请求失败: {e}",
                hint="可换用 backend=semantic_scholar，或稍后重试",
            ) from e
        except Exception as e:
            raise _SearchError(reason=f"arXiv 解析失败: {e}") from e


class VerifyPaperTool(Tool):
    """验证论文是否真实存在（复用 SearchPapersTool 的级联搜索）"""

    def __init__(self):
        super().__init__(
            name="verify_paper",
            description="验证论文是否真实存在（OpenAlex → Semantic Scholar → arXiv 依次查询）。"
        )
        self._searcher = SearchPapersTool()

    def get_parameters(self) -> dict:
        """返回工具参数的 JSON Schema 定义。

        return: input_schema 字典
        """
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "论文标题"},
                "authors": {"type": "string", "description": "第一作者姓名（可选）"},
            },
            "required": ["title"],
        }

    def execute(self, title: str, authors: str = "") -> str:
        """验证论文是否真实存在。

        paras:
            title: 论文标题
            authors: 第一作者姓名（可选）
        return: JSON 字符串 {"verified": bool, ...}，命中时含论文元数据
        """
        try:
            obj = json_mod.loads(self._searcher._search_auto(title, 1))
        except Exception as e:
            return json_mod.dumps({"verified": False, "reason": f"验证失败: {e}"})

        if not obj.get("ok"):
            return json_mod.dumps({"verified": False,
                                   "reason": obj.get("error") or "未找到匹配论文"})
        papers = obj.get("results", [])
        if not papers:
            return json_mod.dumps({"verified": False, "reason": "未找到匹配论文"})

        p = papers[0]
        return json_mod.dumps({
            "verified": True,
            "title": p.get("title", ""),
            "authors": p.get("authors", ""),
            "year": p.get("year", 0),
            "abstract": p.get("abstract", ""),
            "doi": p.get("doi", ""),
        }, ensure_ascii=False)

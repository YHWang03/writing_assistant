import json as json_mod

from ...base import Tool
from ....domain.paper import Paper, Source

class AddReferenceTool(Tool):
    """将论文元数据生成 Paper 实例并存入文献库（需注入 add_func）"""

    def __init__(self):
        '''初始化结构化文献入库工具及入库回调。
        '''
        super().__init__(
            name="add_reference",
            description="将一篇论文的元数据存入文献库，供后续引用检查和写作时查询。"
                        "输入论文的标题、作者、年份、摘要、cite_key 等字段。"
        )
        self._add_func = None
        self.search_results = {}
        self.allowed_pdf_paths = set()

    def set_add_func(self, add_func):
        """注入入库回调（context.add_reference）。

        paras:
            add_func: 接收 Paper 对象的可调用对象
        """
        self._add_func = add_func

    def get_parameters(self) -> dict:
        """返回工具参数的 JSON Schema 定义。"""
        return {
            "type": "object",
            "properties": {
                "cite_key": {"type": "string", "description": "引用键"},
                "pdf_path": {"type": "string", "description": "来自用户PDF时必须填写其路径，尤其是批量解析失败后补录。只做本地来源检查，失败也不联网。"},
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
                journal: str = "", volume: str = "",
                number: str = "", pages: str = "", doi: str = "",
                abstract: str = "", keywords: list | None = None,
                entry_type: str = "article", publisher: str = "", bib_fields: dict | None = None,
                pdf_path: str = "") -> str:
        """将论文元数据存入文献库。

        paras:
            cite_key: 引用键
            title: 论文标题
            authors: 作者列表
            year: 发表年份
            pdf_path: 配置中的用户 PDF 路径；不提供时从实际搜索结果取元数据
            journal/volume/number/pages/doi: 可选字段
            abstract: 论文摘要
            keywords: 关键词列表
            entry_type: 用户 PDF 的 BibTeX 条目类型，默认 article。
            publisher: 用户 PDF 的出版商。
            bib_fields: 用户 PDF 的扩展 BibTeX 字段，如 booktitle；None 使用空字典。
                搜索来源的书目信息以实际搜索记录为准，不接受上述字段替换。
        return: JSON 字符串 {"status": "ok", "cite_key": ...}；
                无合法输入来源时返回错误；不执行联网存在性验证
        """
        if self._add_func is None:
            return json_mod.dumps({"error": "add_reference 工具未注入上下文"})
        ref = Paper(
            cite_key=cite_key, title=title, authors=authors, year=year,
            source=Source.USER if pdf_path else Source.ONLINE, journal=journal, volume=volume,
            number=number, pages=pages, doi=doi, abstract=abstract,
            keywords=keywords or [],
            entry_type=entry_type, publisher=publisher, bib_fields=bib_fields or {},
        )
        try:
            if pdf_path:
                from pathlib import Path
                from ....context.pdf_cache import PDFCache
                path = Path(pdf_path).resolve()
                if str(path) not in self.allowed_pdf_paths:
                    return 'Error: PDF 不在用户配置的输入列表中'
                ref.source_pdf = str(path)
                ref.source_fingerprint = PDFCache().key(path.read_bytes())
            else:
                record = self.search_results.get(title.strip().casefold())
                if record is None:
                    return 'Error: 只能导入用户 PDF 或搜索工具实际返回的文献，禁止凭模型记忆新建文献。'
                # 搜索来源以工具实际返回的数据为准，不接受模型替换作者或摘要。
                names = ('title', 'authors', 'year', 'journal', 'volume', 'number',
                         'pages', 'doi', 'abstract', 'keywords', 'url', 'arxiv_id')
                values = {'title': record['title'], 'authors': '', 'year': 0}
                values.update({name: record[name] for name in names if record.get(name) is not None})
                ref = Paper(cite_key=cite_key, source=Source.ONLINE, **values)
        except (ValueError, OSError, TypeError) as exc:
            return f"Error: {exc}"
        self._add_func(ref)
        return json_mod.dumps({"status": "ok", "cite_key": cite_key})

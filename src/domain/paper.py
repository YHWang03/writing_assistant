"""Paper 数据类 — 一篇学术论文的完整元数据"""

from dataclasses import dataclass, field, asdict
from enum import Enum


class Source(Enum):
    """文献来源：user（用户提供 PDF，可信）/ llm（仅兼容旧数据，不作为新增来源）/ online（在线检索，可信）"""
    USER = "user"
    LLM = "llm"
    ONLINE = "online"


@dataclass
class Paper:
    """一篇学术论文的完整元数据"""

    cite_key: str
    title: str
    authors: str
    year: int
    source: Source = Source.LLM
    abstract: str = ""
    journal: str = ""
    volume: str = ""
    number: str = ""
    pages: str = ""
    doi: str = ""
    issn: str = ""
    url: str = ""
    month: str = ""
    publisher: str = ""
    arxiv_id: str = ""
    keywords: list[str] = field(default_factory=list)
    entry_type: str = "article"  # BibTeX 条目类型
    bib_fields: dict[str, str] = field(default_factory=dict) # 额外的BibTeX字段
    source_pdf: str = ""  # 来源 PDF 文件路径
    source_fingerprint: str = ""  # 来源 PDF 文件的指纹（哈希），用于去重
    display_label: str = ""  # 显示标签，用于在用户界面中显示
    metadata_missing: list[str] = field(default_factory=list)  # 缺失的元数据字段
    field_sources: dict[str, str] = field(default_factory=dict)  # 每个字段的来源
    completion_status: str = ""  # 完成状态

    def to_dict(self) -> dict:
        """转为 dict。"""
        d = asdict(self)
        d["source"] = self.source.value
        return d

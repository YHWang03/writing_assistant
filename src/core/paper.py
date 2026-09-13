"""Paper 数据类 — 表示一篇学术论文的元数据"""

from dataclasses import dataclass, field, asdict
from enum import Enum


class Source(Enum):
    """文献来源"""
    USER = "user"      # 用户提供的 PDF 解析而来，可信
    LLM = "llm"        # LLM 凭自身知识生成，需后续在线验证
    ONLINE = "online"  # 在线检索获得，可信


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

    def to_dict(self) -> dict:
        """转为 dict，source 字段输出为字符串值"""
        d = asdict(self)
        d["source"] = self.source.value
        return d
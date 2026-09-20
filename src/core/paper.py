"""Paper 数据类 — 一篇学术论文的完整元数据"""

from dataclasses import dataclass, field, asdict
from enum import Enum


class Source(Enum):
    """文献来源：user（用户提供 PDF，可信）/ llm（LLM 生成，需验证）/ online（在线检索，可信）"""
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

    def to_dict(self) -> dict:
        """转为 dict。

        paras: 无
        return: 全字段 dict，source 输出为字符串值
        """
        d = asdict(self)
        d["source"] = self.source.value
        return d

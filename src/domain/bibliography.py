"""严格读取项目生成的 braced BibTeX；不支持的语法拒绝导入，绝不静默丢字段。"""

import re
from copy import deepcopy
from .paper import Paper

FIELDS = {"title": "title", "author": "authors", "year": "year",
          **{k: k for k in ("journal", "volume", "number", "pages", "doi", "issn",
                           "url", "month", "publisher")}, "eprint": "arxiv_id"}


def parse_bib(text):
    entries = []
    pos = 0
    def skip():
        nonlocal pos
        while pos < len(text) and text[pos].isspace():
            pos += 1
    def braced():
        nonlocal pos
        if pos >= len(text) or text[pos] != "{":
            raise ValueError("仅支持花括号字段值，请先规范化 BibTeX")
        pos += 1
        start, depth = pos, 1
        while pos < len(text):
            ch = text[pos]
            if ch == "\\":
                pos += 2
                continue
            depth += (ch == "{") - (ch == "}")
            if depth == 0:
                value = text[start:pos]
                pos += 1
                return value
            pos += 1
        raise ValueError("BibTeX 花括号未闭合")
    keys = set()
    while True:
        skip()
        if pos == len(text):
            return entries
        match = re.match(r"@(\w+)\s*\{\s*([^,\s{}]+)\s*,", text[pos:])
        if not match:
            raise ValueError(f"不支持的 BibTeX 语法，位置 {pos}")
        kind, key = match.groups()
        if key in keys:
            raise ValueError(f"重复引用键: {key}")
        keys.add(key)
        pos += match.end()
        fields = {}
        while True:
            skip()
            if pos < len(text) and text[pos] == "}":
                pos += 1
                break
            field = re.match(r"(\w+)\s*=\s*", text[pos:])
            if not field:
                raise ValueError(f"无效 BibTeX 字段，位置 {pos}")
            name = field[1].lower()
            if name in fields:
                raise ValueError(f"重复字段: {name}")
            pos += field.end()
            fields[name] = braced()
            skip()
            if pos < len(text) and text[pos] == ",":
                pos += 1
            elif pos >= len(text) or text[pos] != "}":
                raise ValueError("字段间缺少逗号")
        entries.append((kind, key, fields))


def merge_bib(papers, text):
    """显式迁移：已有摘要/来源不变，以 bib 的书目字段为准，不删除库中其他记录。"""
    refs = {p.cite_key: deepcopy(p) for p in papers}
    for kind, key, fields in parse_bib(text):
        ref = refs.get(key, Paper(key, "", "", 0))
        ref.entry_type = kind
        for bib_name, attr in FIELDS.items():
            value = fields.get(bib_name, "")
            setattr(ref, attr, int(value or 0) if attr == "year" else value)
        ref.bib_fields = {k: v for k, v in fields.items() if k not in FIELDS}
        refs[key] = ref
    return list(refs.values())


def render_bib(papers):
    entries = []
    for ref in papers:
        fields = dict(ref.bib_fields)
        for bib_name, attr in FIELDS.items():
            value = getattr(ref, attr)
            if value:
                if bib_name == "author" and " and " not in str(value):
                    value = " and ".join(a.strip() for a in str(value).split(",") if a.strip())
                fields[bib_name] = str(value)
        if ref.arxiv_id:
            fields.setdefault("archiveprefix", "arXiv")
        kind = "misc" if ref.arxiv_id and ref.entry_type == "article" else ref.entry_type
        if not re.fullmatch(r"[A-Za-z]+", kind) or not re.fullmatch(r"[^,\s{}]+", ref.cite_key):
            raise ValueError("无效的 BibTeX 类型或引用键")
        if any(not re.fullmatch(r"\w+", name) for name in fields):
            raise ValueError("无效的 BibTeX 字段名")
        # 保留已有 LaTeX 命令和保护大小写的括号，仅转义尚未转义的特殊字符。
        fields = {name: re.sub(r"(?<!\\)([&%$#_])", r"\\\1", value)
                  for name, value in fields.items()}
        body = ",\n".join(f"  {name} = {{{value}}}" for name, value in fields.items())
        entries.append(f"@{kind}{{{ref.cite_key},\n{body}\n}}")
    text = "\n\n".join(entries) + "\n"
    parse_bib(text)  # 拒绝不平衡括号等结构污染，再写入文件。
    return text

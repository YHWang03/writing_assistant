"""严格读取项目生成的 braced BibTeX；不支持的语法拒绝导入，绝不静默丢字段。"""

import re
from copy import deepcopy
from .paper import Paper

FIELDS = {"title": "title", "author": "authors", "year": "year",
          **{k: k for k in ("journal", "volume", "number", "pages", "doi", "issn",
                           "url", "month", "publisher")}, "eprint": "arxiv_id"}


def format_entry(kind, key, fields):
    '''拼接单条 BibTeX，字段转义和有效性校验由调用方负责。

    paras:
        kind: 条目类型，如 article。
        key: 文献引用键。
        fields: 已转义的字段名到字段值的映射，保持传入顺序。
    return: 完整的单条 BibTeX 文本。
    '''
    body = ",\n".join(f"  {name} = {{{value}}}" for name, value in fields.items())
    return f"@{kind}{{{key},\n{body}\n}}"


def parse_bib(text):
    '''严格解析花括号字段形式的 BibTeX，拒绝重复键及不支持的语法。

    paras:
        text: 待解析或扫描的文本。
    return: 由条目类型、引用键和字段字典组成的三元组列表。
    '''
    entries = []
    pos = 0
    def skip():
        '''推进解析游标，跳过当前位置后的空白字符。
        '''
        nonlocal pos
        while pos < len(text) and text[pos].isspace():
            pos += 1
    def braced():
        '''读取当前花括号字段，处理嵌套括号和转义并推进游标。

        return: 不含最外层括号的字段内容；结构非法时抛出 ValueError。
        '''
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
    '''将 BibTeX 书目字段合并到文献副本，保留已有摘要、来源及库中其他记录。

    paras:
        papers: 原有 Paper 序列，不原地修改。
        text: 待迁移的 BibTeX 文本；非法结构或年份转换失败时抛出 ValueError。
    return: 合并后的 Paper 列表；同键书目字段以 BibTeX 为准，缺失书目字段也会被清空。
    '''
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
    '''将结构化文献渲染为 BibTeX，转义特殊字符并检查最终结构。

    paras:
        papers: 结构化文献对象序列。
    return: 包含全部文献条目的 BibTeX 文本。
    '''
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
        entries.append(format_entry(kind, ref.cite_key, fields))
    text = "\n\n".join(entries) + "\n"
    parse_bib(text)  # 拒绝不平衡括号等结构污染，再写入文件。
    return text

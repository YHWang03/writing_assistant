"""cite_key 生成（authorYearFirstWord 格式），供 bib.py 与 pdf.py 共用。"""

import re


def make_cite_key(authors: str, year: int, title: str) -> str:
    """生成 cite_key：第一作者姓 + 年份 + 标题首词（小写，如 vidale1988finite）。

    paras:
        authors: 作者列表（逗号分隔）
        year: 发表年份
        title: 论文标题
    return: cite_key；缺作者或年份时返回空串，调用方据此判定无法生成
    """
    if not authors or not year:
        return ""
    try:
        first_author = authors.split(",")[0].strip()
        last_name = first_author.split()[-1].lower()
        last_name = re.sub(r'[^a-z]', '', last_name)
    except (IndexError, ValueError):
        last_name = "unknown"
    try:
        first_word = re.findall(r'[a-zA-Z]+', title)[0].lower()
    except IndexError:
        first_word = "paper"
    return f"{last_name}{year}{first_word}"

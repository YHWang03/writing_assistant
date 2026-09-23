"""Shared BibTeX escaping helpers."""

import re

_LATEX_ESCAPES = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\~{}", "^": r"\^{}",
    "\\": r"\textbackslash{}",
}

_LATEX_SPECIAL_RE = re.compile(r'[&%$#_{}~^\\]')


def _escape_latex(text: str) -> str:
    """转义 BibTeX/LaTeX 特殊字符，防止编译错误。

    paras:
        text: 原始文本
    return: 转义后的文本；空文本原样返回
    """
    if not text:
        return text
    # 单遍正则替换；顺序 replace 会造成二次转义（先 \$ 再转 \ 会破坏已转义的 \$）
    return _LATEX_SPECIAL_RE.sub(lambda m: _LATEX_ESCAPES[m.group(0)], text)


def _escape_bibtex_fields(entry: str) -> str:
    """对 BibTeX 条目中所有字段值做 LaTeX 转义，不破坏条目结构语法。

    paras:
        entry: BibTeX 条目文本
    return: 字段值转义后的条目文本
    """
    def _escape_field(match):
        # match 分组：1=缩进+字段名+" = {"，2=字段值，3=} 或 },
        indent = match.group(1)
        value = match.group(2)
        suffix = match.group(3)
        return f"{indent}{_escape_latex(value)}{suffix}"

    return re.sub(
        r'^(\s*\w+\s*=\s*\{)([^}]*)(\},?)$',
        _escape_field,
        entry,
        flags=re.MULTILINE,
    )

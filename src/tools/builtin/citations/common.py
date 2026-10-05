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

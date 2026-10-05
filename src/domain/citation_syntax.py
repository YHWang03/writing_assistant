"""Shared citation syntax; callers choose their existing scan scope."""
import re

_OPTIONS_AND_KEYS = r'(?:\s*\[[^\]]*\])*\s*\{([^}]+)\}'
STANDARD_CITATION = re.compile(r'\\(?:cite|citet|citep|citeauthor|citeyear)' + _OPTIONS_AND_KEYS)
PROTECTED_CITATION = re.compile(r'\\(?:[A-Za-z]*cite[A-Za-z]*|nocite)\*?' + _OPTIONS_AND_KEYS)


def citation_keys(match):
    '''拆分引用正则匹配中的键列表并去除首尾空白。

    paras:
        match: 正则表达式匹配对象。
    return: 引用键列表。
    '''
    return [key.strip() for key in match[1].split(',') if key.strip()]

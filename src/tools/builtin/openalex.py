import os

def openalex_headers() -> dict:
    """构造 OpenAlex 请求头（User-Agent 带 mailto，可用 .env 的 OPENALEX_MAILTO 配置）。

    return: 请求头 dict
    """
    mailto = os.getenv("OPENALEX_MAILTO", "")
    ua = "WritingAssistant/1.0"
    if mailto:
        ua += f" (mailto:{mailto})"
    return {"User-Agent": ua}

def reconstruct_abstract(inv: dict | None) -> str:
    """还原 OpenAlex 的倒排索引摘要（{词: [位置]}）为原文。

    paras:
        inv: 倒排索引 dict
    return: 摘要文本；无索引返回空串
    """
    if not inv:
        return ""
    pos_to_word = {}
    for word, positions in inv.items():
        for p in positions:
            pos_to_word[p] = word
    return " ".join(pos_to_word[i] for i in sorted(pos_to_word))

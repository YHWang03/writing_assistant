"""token 计数 — DeepSeek V4 官方 BPE 词表精确计数，不可用时回退 CJK/ASCII 启发式"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_TOKENIZER_PATH = Path(__file__).resolve().parent.parent.parent / "resources" / "ds-v4" / "tokenizer.json"

# CJK 及全角字符（fallback 估算：1 token/字，ASCII 4 字符/token）
_CJK_RE = re.compile(
    r'[一-鿿㐀-䶿'
    r'぀-ヿ'
    r'　-〿＀-￯'
    r']'
)

# 每条消息的 chat 模板开销
_TOKENS_PER_MESSAGE = 4

_tokenizer = None
_load_error: str | None = None
_load_attempted = False


def _get_tokenizer():
    """懒加载 tokenizer 单例，失败返回 None 且不再重试。"""
    global _tokenizer, _load_error, _load_attempted
    if _load_attempted:
        return _tokenizer
    _load_attempted = True

    try:
        from tokenizers import Tokenizer
    except ImportError as e:
        _load_error = f"tokenizers 库未安装: {e}"
        logger.warning(f"[token_counter] {_load_error}，回退到启发式估算")
        return None

    if not _TOKENIZER_PATH.exists():
        _load_error = f"词表文件不存在: {_TOKENIZER_PATH}"
        logger.warning(f"[token_counter] {_load_error}，回退到启发式估算")
        return None

    try:
        _tokenizer = Tokenizer.from_file(str(_TOKENIZER_PATH))
        logger.info(
            f"[token_counter] 已加载 DeepSeek V4 tokenizer: {_TOKENIZER_PATH.name} "
            f"({_TOKENIZER_PATH.stat().st_size // 1024} KB)",
            extra={"event": "tokenizer_loaded", "path": str(_TOKENIZER_PATH)},
        )
        return _tokenizer
    except Exception as e:
        _load_error = f"加载 tokenizer 失败: {e}"
        logger.warning(f"[token_counter] {_load_error}，回退到启发式估算")
        return None


def is_available() -> bool:
    """tokenizer 是否可用。"""
    return _get_tokenizer() is not None


def count_tokens(text: str) -> int:
    """精确计数单段文本 token 数。

    paras:
        text: 原始文本
    return: token 数；tokenizer 不可用返回 -1
    """
    tok = _get_tokenizer()
    if tok is None or not text:
        return -1
    return len(tok.encode(text).ids)


def _legacy_estimate_text(text: str) -> int:
    """启发式估算单段文本：CJK 1 token/字，ASCII 4 字符/token

    paras:
        text: 原始文本
    return: token 估算值
    """
    if not text:
        return 0
    n_cjk = len(_CJK_RE.findall(text))
    n_ascii = len(text) - n_cjk
    if n_ascii <= 0:
        return n_cjk
    return n_cjk + max(1, (n_ascii + 3) // 4)


def estimate_tokens(messages: list) -> int:
    """估算消息列表 token 数（精确优先，启发式回退）

    paras:
        messages: dict 或 Message 消息列表
    return: token 估算值
    """
    tok = _get_tokenizer()

    def count(text):
        '''按已选后端计数单段文本；回退时使用原启发式规则。

        paras:
            text: 待计数的字符串。
        return: 此段文本的 token 数或估算值。
        '''
        return len(tok.encode(text).ids) if tok is not None else _legacy_estimate_text(text)

    total = 0
    n_messages = 0
    for msg in messages:
        n_messages += 1
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if isinstance(content, str):
            total += count(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                for key in ("text", "thinking"):
                    v = block.get(key, "")
                    if isinstance(v, str) and v:
                        total += count(v)
                for key, default in (("input", {}), ("content", "")):
                    value = block.get(key, default if tok is None else None)
                    # 保留旧回退规则：缺失 input 按 {} 计数，显式 None 按文本计数。
                    if value is not None or tok is None:
                        text = str(value)
                        if text:
                            total += count(text)

    return total + (n_messages * _TOKENS_PER_MESSAGE if tok is not None else 0)


def get_status() -> dict:
    """返回计数器状态供诊断。"""
    return {
        "available": _tokenizer is not None,
        "attempted": _load_attempted,
        "error": _load_error,
        "tokenizer_path": str(_TOKENIZER_PATH),
        "tokenizer_exists": _TOKENIZER_PATH.exists(),
    }

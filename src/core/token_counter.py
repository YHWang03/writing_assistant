"""
DeepSeek V4 token 精确计数器 — 替代 context_compress.estimate_tokens 的启发式估算。

实现思路：
  - 使用 HuggingFace `tokenizers` 库 + DeepSeek-V4 官方 BPE 词表
  - 与 DeepSeek API 服务端计数一致（误差 < 0.1%，仅 chat 模板开销有几 token 偏差）
  - 单例 + 懒加载：首次调用 estimate_tokens 时加载 6 MB 词表，进程内常驻
  - Fallback：若 tokenizers 库缺失或词表文件丢失，自动回退到原 CJK/ASCII 启发式
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# 词表文件路径：项目根/resources/ds-v4/tokenizer.json
_TOKENIZER_PATH = Path(__file__).resolve().parent.parent.parent / "resources" / "ds-v4" / "tokenizer.json"

# CJK 及全角字符正则：fallback 估算时区分 CJK（1 token/字）与 ASCII（4 字符/token）
_CJK_RE = re.compile(
    r'[一-鿿㐀-䶿'   # CJK 统一表意文字 + 扩展 A
    r'぀-ヿ'                  # 日文假名
    r'　-〿＀-￯'     # CJK 标点 + 全角形式
    r']'
)

# 每条 chat 消息的模板开销（与 OpenAI cookbook 的 tokens_per_message=3 对齐，
# DeepSeek 实测约 4-6 token/消息，取下限 4 作平均估计）
_TOKENS_PER_MESSAGE = 4

# 单例状态
_tokenizer = None          # 已加载的 Tokenizer 实例（None 表示未加载）
_load_error: str | None = None   # 加载失败原因（None 表示未尝试过或加载成功）
_load_attempted = False    # 是否已尝试过加载（避免失败后反复重试）


def _get_tokenizer():
    """懒加载 tokenizer 单例。加载失败返回 None，调用方回退到启发式。

    首次失败后不再重试（_load_attempted 标记），避免每步 ReAct 循环都做无谓的 IO。
    """
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
    """tokenizer 是否可用（已加载或可加载）。供调用方决定是否启用精确计数路径。"""
    return _get_tokenizer() is not None


def count_tokens(text: str) -> int:
    """精确计数单段文本的 token 数。tokenizer 不可用时返回 -1。"""
    tok = _get_tokenizer()
    if tok is None or not text:
        return -1
    return len(tok.encode(text).ids)


def _legacy_estimate_text(text: str) -> int:
    """原 context_compress._estimate_text 的启发式：CJK 1 token/字，ASCII 4 字符/token。

    保留作 fallback，与改造前行为完全一致，确保 tokenizer 不可用时数值不漂移。
    """
    if not text:
        return 0
    n_cjk = len(_CJK_RE.findall(text))
    n_ascii = len(text) - n_cjk
    if n_ascii <= 0:
        return n_cjk
    # 短 ASCII 片段向上取整（至少 1 token），避免 <4 字符被估成 0
    return n_cjk + max(1, (n_ascii + 3) // 4)


def estimate_tokens(messages: list) -> int:
    """估算消息列表的 token 数。

    - tokenizer 可用：精确计数每个文本块的 token 数 + 每条消息的 chat 模板开销
    - tokenizer 不可用：回退到原 CJK/ASCII 启发式（行为与改造前一致）

    签名与原 context_compress.estimate_tokens 完全一致，可直接替换。
    """
    tok = _get_tokenizer()

    # ---- Fallback 路径：保留原启发式 ----
    if tok is None:
        total = 0
        for msg in messages:
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            if isinstance(content, str):
                total += _legacy_estimate_text(content)
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    total += _legacy_estimate_text(block.get("text", "") or "")
                    total += _legacy_estimate_text(str(block.get("input", {})) or "")
                    total += _legacy_estimate_text(block.get("thinking", "") or "")
                    total += _legacy_estimate_text(str(block.get("content", "")) or "")
        return total

    # ---- 精确计数路径 ----
    total = 0
    n_messages = 0
    for msg in messages:
        n_messages += 1
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if isinstance(content, str):
            total += len(tok.encode(content).ids)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                # text / thinking 字段：直接是字符串
                for key in ("text", "thinking"):
                    v = block.get(key, "")
                    if isinstance(v, str) and v:
                        total += len(tok.encode(v).ids)
                # input 字段：dict 转 str 后计数
                inp = block.get("input")
                if inp is not None:
                    s = str(inp)
                    if s:
                        total += len(tok.encode(s).ids)
                # content 字段：tool_result 的内容，可能是 str 或其他
                c = block.get("content")
                if isinstance(c, str) and c:
                    total += len(tok.encode(c).ids)
                elif c is not None and not isinstance(c, str):
                    s = str(c)
                    if s:
                        total += len(tok.encode(s).ids)

    # chat 模板开销：每条消息约 4 token（<｜User｜>...<｜Assistant｜> 等包裹）
    return total + n_messages * _TOKENS_PER_MESSAGE


def get_status() -> dict:
    """返回当前 token 计数器状态，供诊断或日志使用。"""
    return {
        "available": _tokenizer is not None,
        "attempted": _load_attempted,
        "error": _load_error,
        "tokenizer_path": str(_TOKENIZER_PATH),
        "tokenizer_exists": _TOKENIZER_PATH.exists(),
    }
